import numpy as np
import pdb
import torch
import copy
from typing import Dict, List, Optional, Tuple, Any, Union
import rustworkx as rx
import gc
from collections import OrderedDict
import multiprocessing

from environments.base_env import BaseEnv
from environments.base_task_config import BaseTaskConfig

import threading
import time
import logging
import wandb

from utils.general_utils import calculate_fitness_metrics
from utils.visualization import log_node_data_to_wandb
from utils import file_logger

from algorithms.common import gaussian_mutation, polynomial_mutation
from algorithms.common import sbx, iso_dd, regression_monet
from algorithms.common import AUCTracker

log = logging.getLogger(__name__)


_worker_envs = {}  # per-process cache: task_id -> env instance


def _worker_init(task_configs, task_env_class):
    """Pool initializer: build the env cache once per worker process."""
    global _worker_envs
    _worker_envs = {cfg.task_id: task_env_class(cfg) for cfg in task_configs}


def _eval_standalone(args):
    """Module-level worker function for parallel evaluation (must be picklable).
    Uses the per-worker env cache so only (task_id, solution) crosses the pipe."""
    task_id, solution = args
    fitness = _worker_envs[task_id].evaluate_solution(solution)
    return fitness, (0.0, 0.0)


class MetricsLogger:
    def __init__(self, use_wandb):
        self.use_wandb = use_wandb
        file_logger.init(project="monet", config={}, name="monet_run") # file logger no matter what

    def log(self, data, step=None):
        if self.use_wandb:
            wandb.log(data, step=step)
        file_logger.log(data, step=step)
        log.info(data)

    def finish(self):
        if self.use_wandb:
            wandb.finish()
        file_logger.finish()

def run_monet(monet, cfg, task_config, use_wandb):
    """Run MONET algorithm.

    Args:
        monet: MONET instance to run
        cfg: Configuration object containing algorithm parameters
        task_config: Task configuration object
        use_wandb: Boolean flag for using wandb logging
    """
    metrics_logger = MetricsLogger(use_wandb)

    # Derived hyperparameters
    num_init = int(cfg.algorithm.num_init_percent * task_config.num_tasks)
    num_neighbors = max(1, int(cfg.algorithm.neighbor_percentage * task_config.num_tasks))
    strategy = cfg.algorithm.strategy
    threshold = cfg.algorithm.threshold

    log.info("building neighbor cache")
    # building neighbor cache - use actual task_ids, not indices
    for task_id in monet.task_ids:
        monet.get_task_neighbors(task_id, threshold=cfg.algorithm.threshold,
                                            num_neighbors=num_neighbors)
    log.info("caching done..")

    mean_fitness = []
    median_fitness = []
    qd_scores = []
    avg_cluster_sizes = []
    avg_coverages = []
    mean_fitness_auc = AUCTracker()

    log.info(f"running init loop for {num_init} tasks")
    # init num_task nodes
    n_eval = 0
    steps = 0
    metrics = calculate_fitness_metrics(monet)
    mean_fitness.append(metrics["mean_fitness"])
    median_fitness.append(metrics["median_fitness"])
    qd_scores.append(metrics["qd_score"])
    avg_cluster_sizes.append(metrics["avg_cluster_size"])
    avg_coverages.append(metrics["coverage"])
    mean_fitness_auc.update(n_eval, metrics["mean_fitness"])

    for _ in range(num_init):
        solution = np.random.uniform(low=task_config.min_solution, high=task_config.max_solution, size=task_config.solution_dim)
        # Sample from actual task_ids, not assuming sequential integers
        task_id = np.random.choice(monet.task_ids)
        node = monet.get_node(task_id)
        # we create a temp node to call eval node
        temp_node = {"task_config": node["task_config"], "solution": solution}
        fitness, behavior = monet.evaluate_node(temp_node)

        # Only update if better or no solution exists (Elitism during init)
        current_fitness = node.get('fitness', -float('inf'))
        if node['solution'] is None or fitness > current_fitness:
            monet.add_node(task_id=task_id, solution=solution, fitness=fitness,
                                behavior=behavior,
                                n_eval=n_eval, solution_id=n_eval)
        n_eval += 1
        steps += 1

        # Only calculate metrics at intervals to reduce overhead
        if steps % max(1, num_init // 10) == 0 or steps == num_init:
            metrics = calculate_fitness_metrics(monet)
            mean_fitness_auc.update(n_eval, metrics["mean_fitness"])
            metrics_logger.log({"step": steps,
                                "n_evals": n_eval,
                                "mean_fitness": metrics["mean_fitness"],
                                "median_fitness": metrics["median_fitness"],
                                "qd_score": metrics["qd_score"],
                                "avg_cluster_size": metrics["avg_cluster_size"],
                                "solution_id_count": metrics["solution_id_count"],
                                "coverage": metrics["coverage"],
                                "mean_fitness_auc": mean_fitness_auc.auc})
            mean_fitness.append(metrics["mean_fitness"])
            median_fitness.append(metrics["median_fitness"])
            qd_scores.append(metrics["qd_score"])
            avg_cluster_sizes.append(metrics["avg_cluster_size"])
            avg_coverages.append(metrics["coverage"])

    log.info(f"Mean Fitness: {mean_fitness[-1]}, Median Fitness: {median_fitness[-1]}, QD-Score: {qd_scores[-1]}, Average Cluster Size: {avg_cluster_sizes[-1]}, Coverage: {avg_coverages[-1]}")
    log.info("init done, starting main loop")

    no_solution = 0
    none_neighbor = 0
    skip_individual_learning = 0
    skip_social_learning = 0
    success_individual_learning = 0
    success_social_learning = 0
    fail_individual_learning = 0
    fail_social_learning = 0
    try:
        log_node_data_to_wandb(monet, n_eval, use_wandb)
    except Exception as e:
        log.warning(f"Logging to wandb failed at init with error: {e}")

    n_workers = getattr(cfg.algorithm, 'n_workers', 1)

    def _log_metrics(n_eval, steps):
        metrics = calculate_fitness_metrics(monet)
        mean_fitness_auc.update(n_eval, metrics["mean_fitness"])
        metrics_logger.log({"step": steps, "n_evals": n_eval,
                             "mean_fitness": metrics["mean_fitness"],
                             "median_fitness": metrics["median_fitness"],
                             "qd_score": metrics["qd_score"],
                             "solution_id_count": metrics["solution_id_count"],
                             "avg_cluster_size": metrics["avg_cluster_size"],
                             "coverage": metrics["coverage"],
                             "mean_fitness_auc": mean_fitness_auc.auc,
                             "no_solution": no_solution,
                             "none_neighbor": none_neighbor,
                             "skip_individual_learning": skip_individual_learning,
                             "skip_social_learning": skip_social_learning,
                             "success_individual_learning": success_individual_learning,
                             "success_social_learning": success_social_learning,
                             "fail_individual_learning": fail_individual_learning,
                             "fail_social_learning": fail_social_learning})
        mean_fitness.append(metrics["mean_fitness"])
        median_fitness.append(metrics["median_fitness"])
        qd_scores.append(metrics["qd_score"])
        avg_cluster_sizes.append(metrics["avg_cluster_size"])
        avg_coverages.append(metrics["coverage"])
        log.info(f"Step: {steps}, N_eval: {n_eval}, "
                 f"Mean Fitness: {metrics['mean_fitness']}, "
                 f"Median Fitness: {metrics['median_fitness']}, "
                 f"QD-Score: {metrics['qd_score']}, "
                 f"Average Cluster Size: {metrics['avg_cluster_size']}, "
                 f"Coverage: {metrics['coverage']}, "
                 f"Mean Fitness AUC: {mean_fitness_auc.auc}")
        return metrics

    if n_workers <= 1:
        # --- Sequential loop (original behaviour) ---
        while n_eval < cfg.algorithm.max_evals:
            steps += 1
            chosen_task_id = np.random.choice(monet.task_ids)
            if np.random.rand() < cfg.algorithm.p_ind:
                node = monet.get_node(chosen_task_id)
                res = monet.individual_learning(task_id=chosen_task_id, n_eval=n_eval, cfg=cfg, p_min=task_config.min_solution, p_max=task_config.max_solution)
                if res == "success":
                    success_individual_learning += 1
                    n_eval += 1
                elif res == "failed":
                    fail_individual_learning += 1
                    n_eval += 1
                else:
                    skip_individual_learning += 1
            else:
                neighbor_id = monet.get_candidate_neighbor(task_id=chosen_task_id, strategy=strategy,
                                                            threshold=threshold, num_neighbors=num_neighbors)
                if neighbor_id is None:
                    none_neighbor += 1
                    continue
                res = monet.social_learning(task_id=chosen_task_id, neighbor_id=neighbor_id, n_eval=n_eval, cfg=cfg, p_min=task_config.min_solution, p_max=task_config.max_solution, num_neighbors=num_neighbors)
                if res == "success":
                    success_social_learning += 1
                    n_eval += 1
                elif res == "failed":
                    fail_social_learning += 1
                    n_eval += 1
                else:
                    skip_social_learning += 1

            if n_eval % cfg.log_interval == 0:
                try:
                    log_node_data_to_wandb(monet, n_eval, use_wandb)
                except Exception as e:
                    log.warning(f"Logging to wandb failed at n_eval={n_eval} with error: {e}")
            if n_eval % cfg.log_frequency == 0:
                _log_metrics(n_eval, steps)

    else:
        # --- Parallel loop (batched evaluations) ---
        log.info(f"Running parallel MONET with {n_workers} workers")
        p_min = task_config.min_solution
        p_max = task_config.max_solution
        with multiprocessing.Pool(n_workers,
                                  initializer=_worker_init,
                                  initargs=(monet.task_configs, monet.task_env_class)) as pool:
            while n_eval < cfg.algorithm.max_evals:
                steps += 1
                # Prepare a batch of candidates from the current archive snapshot
                batch = []  # list of (task_id, solution, current_fitness)
                for _ in range(n_workers):
                    task_id = np.random.choice(monet.task_ids)
                    solution, current_fitness = monet._prepare_candidate(
                        task_id, cfg, p_min, p_max, num_neighbors, strategy, threshold)
                    batch.append((task_id, solution, current_fitness))

                # Evaluate all candidates in parallel — only (task_id, solution) crosses the pipe
                results = pool.map(_eval_standalone,
                                   [(task_id, sol) for task_id, sol, _ in batch])

                # Commit results to archive (elitism: only update if improvement)
                for (task_id, solution, current_fitness), (fitness, behavior) in zip(batch, results):
                    if fitness >= current_fitness:
                        monet.add_node(task_id, solution, fitness=fitness,
                                       behavior=behavior, n_eval=n_eval, solution_id=n_eval)
                        success_individual_learning += 1
                    else:
                        fail_individual_learning += 1
                    n_eval += 1

                if n_eval % cfg.log_interval == 0:
                    try:
                        log_node_data_to_wandb(monet, n_eval, use_wandb)
                    except Exception as e:
                        log.warning(f"Logging to wandb failed at n_eval={n_eval} with error: {e}")
                if n_eval % cfg.log_frequency == 0:
                    _log_metrics(n_eval, steps)

    # log final metrics
    log.info("Finished running MONET.")
    metrics = calculate_fitness_metrics(monet)
    mean_fitness_auc.update(n_eval, metrics["mean_fitness"])
    metrics_logger.log({"step": steps,
                        "n_evals": n_eval,
                        "mean_fitness": metrics["mean_fitness"],
                        "median_fitness": metrics["median_fitness"],
                        "qd_score": metrics["qd_score"],
                        "solution_id_count": metrics["solution_id_count"],
                        "avg_cluster_size": metrics["avg_cluster_size"],
                        "coverage": metrics["coverage"],
                        "mean_fitness_auc": mean_fitness_auc.auc,
                        "no_solution": no_solution,
                        "none_neighbor": none_neighbor,
                        "skip_individual_learning": skip_individual_learning,
                        "skip_social_learning": skip_social_learning,
                        "success_individual_learning": success_individual_learning,
                        "success_social_learning": success_social_learning,
                        "fail_individual_learning": fail_individual_learning,
                        "fail_social_learning": fail_social_learning
                        })
    # Final log
    log_node_data_to_wandb(monet, n_eval, use_wandb)
    log.info(f"Final Mean Fitness AUC: {mean_fitness_auc.auc}")

    # Finish logging
    metrics_logger.finish()

    # return mean_fitness
    print(f"final_mean_fitness: {metrics['mean_fitness']}")
    print(f"final_auc: {mean_fitness_auc.auc}")

class MONET:
    """
    Multi-Task Optimization Network (MONET).

    Uses rustworkx PyGraph for efficient graph operations with task nodes.
    Supports similarity-based and random neighbor selection strategies.

    Args:
        task_configs: List of task configurations
        task_env: Environment class for task evaluation
        solution_dim: Dimensionality of solution vectors
        seed: Random seed for reproducibility
        threshold: Similarity threshold for edge creation (ignored if use_random_neighborhood=True)
        use_random_neighborhood: If True, randomly sample neighbors instead of using similarity edges
        max_neighbors_per_node: Maximum neighbors to keep per node (for edge pruning)
        store_similarity_matrix: If True, keep similarity matrix in memory after initialization
        distance_proportional_sampling: If True, sample neighbors proportional to similarity
    """

    def __init__(self, task_configs: List[BaseTaskConfig], task_env: BaseEnv,
                 solution_dim: int = 10,
                 seed: int = 42,
                 threshold: float = 0.0,
                 use_random_neighborhood: bool = False,
                 max_neighbors_per_node: Optional[int] = None,
                 store_similarity_matrix: bool = False,
                 distance_proportional_sampling: bool = False):

        self.task_configs = task_configs
        self.task_env_class = task_env
        self.solution_dim = solution_dim
        self.seed = seed
        self.use_random_neighborhood = use_random_neighborhood
        self.distance_proportional_sampling = distance_proportional_sampling
        self.max_neighbors_per_node = max_neighbors_per_node
        self.store_similarity_matrix = store_similarity_matrix

        # Use rustworkx graph as the primary data structure
        self.graph = rx.PyGraph()
        self.environments = {}

        # Compute similarity matrix only if needed
        if self.use_random_neighborhood or self.store_similarity_matrix:
            self.similarity_matrix = self._compute_task_similarity_matrix()
        else:
            self.similarity_matrix = None
        self.neighbors_cache = {}

        # Mapping from task_id -> index in task_configs (for similarity lookup)
        self.task_index = {cfg.task_id: i for i, cfg in enumerate(self.task_configs)}
        # List of task IDs for sampling/iteration
        self.task_ids = [cfg.task_id for cfg in self.task_configs]

        # Thread-safe locks
        self.node_lock = threading.RLock()
        self.cache_lock = threading.Lock()

        # Build the graph structure
        self._initialize_graph(threshold)

        # Optional pruning and similarity matrix cleanup
        if not self.use_random_neighborhood and self.max_neighbors_per_node is not None:
            #self._prune_edges(self.max_neighbors_per_node) not yet working correctly, its pruning too many edges...
            pass
        if not self.use_random_neighborhood and not self.store_similarity_matrix:
            self.similarity_matrix = None
            gc.collect()

        # Initialize environments
        for config in task_configs:
            self.environments[config.task_id] = task_env(config)

    def _initialize_graph(self, threshold: float = 0.0):
        """Initialize the graph with task nodes and similarity-based edges.

        Creates a node for each task and adds edges between tasks with
        similarity >= threshold (unless using random neighborhood mode).

        Args:
            threshold: Minimum similarity for edge creation (default: 0.0)
        """
        # Add nodes for each task with their data
        self.task_to_node_idx = {}
        self.node_idx_to_task = {}

        for i, config in enumerate(self.task_configs):
            node_data = {
                'timestamp': time.time(),
                'individual_learning_success': 0,
                'individual_learning_fail': 0,
                'social_learning_success': 0,
                'social_learning_fail': 0,
                'task_id': config.task_id,
                'solution': None,
                'solution_id': config.task_id,  # Initially, solution_id is the same as task_id
                'fitness': -float('inf'),
                'behavior': None,
                'task_config': config.to_dict(),
                'last_updated': [],              # FIX: was int sentinel; now a list
            }
            node_idx = self.graph.add_node(node_data)
            self.task_to_node_idx[config.task_id] = node_idx
            self.node_idx_to_task[node_idx] = config.task_id

        # Add similarity edges (only if not using random neighbors)
        if not self.use_random_neighborhood:
            n = len(self.task_configs)
            for i in range(n):
                for j in range(i + 1, n):
                    similarity = self.similarity_matrix[i, j] if self.similarity_matrix is not None else self._pair_similarity(i, j)
                    if similarity >= threshold:
                        node_i = self.task_to_node_idx[self.task_configs[i].task_id]
                        node_j = self.task_to_node_idx[self.task_configs[j].task_id]
                        self.graph.add_edge(node_i, node_j, similarity)

    def _pair_similarity(self, i: int, j: int) -> float:
        """Compute similarity between two tasks on demand.

        Uses Euclidean distance between task vectors with similarity = 1/(1+distance).

        Args:
            i: Index of first task in task_configs
            j: Index of second task in task_configs

        Returns:
            Similarity score in range (0, 1]
        """
        vi = np.array(self.task_configs[i].task_vec)
        vj = np.array(self.task_configs[j].task_vec)
        dist = np.linalg.norm(vi - vj)
        return 1.0 / (1.0 + dist)

    def _prune_edges(self, k: int):
        """Keep only top-k highest-similarity edges per node.

        Args:
            k: Number of top edges to keep per node. If k <= 0, removes all edges.
        """
        if k is None:
            return
        if k <= 0:
            for (u, v) in list(self.graph.edge_list()):
                self.graph.remove_edge(u, v)
            with self.cache_lock:
                self.neighbors_cache.clear()
            return
        to_remove = set()
        for node_idx in range(len(self.graph)):
            nbrs = [(nbr_idx, self.graph.get_edge_data(node_idx, nbr_idx)) for nbr_idx in self.graph.neighbors(node_idx)]
            if len(nbrs) <= k:
                continue
            nbrs.sort(key=lambda x: x[1], reverse=True)
            for drop_nbr, _ in nbrs[k:]:
                a, b = (node_idx, drop_nbr) if node_idx < drop_nbr else (drop_nbr, node_idx)
                to_remove.add((a, b))
        for (u, v) in to_remove:
            try:
                self.graph.remove_edge(u, v)
            except Exception:
                pass
        with self.cache_lock:
            self.neighbors_cache.clear()

    # Expose graph-like interface
    def __len__(self):
        """Number of nodes in the graph"""
        return len(self.graph)

    def __contains__(self, task_id):
        """Check if task_id is in the graph"""
        return task_id in self.task_to_node_idx

    def __getitem__(self, task_id):
        """Get node data for a task"""
        return self.get_node(task_id)

    def __setitem__(self, task_id, value):
        """Set node data for a task"""
        if isinstance(value, dict) and 'solution' in value:
            self.add_node(task_id, value['solution'], value['fitness'], value['behavior'], value['n_eval'])

    def nodes(self):
        """Iterator over all task IDs"""
        return self.task_to_node_idx.keys()

    def neighbors(self, task_id: int, threshold: float = 0.0, num_neighbors: int = 8):
        """Get neighbors - more Pythonic interface"""
        return self.get_task_neighbors(task_id, threshold, num_neighbors)

    def degree(self, task_id: int, threshold: float = 0.0):
        """Get degree of a task node"""
        if task_id not in self.task_to_node_idx:
            return 0
        return len(self.get_task_neighbors(task_id, threshold))

    def edges(self, task_id: Optional[int] = None):
        """Get edges, optionally for a specific task"""
        if (task_id is not None):
            if task_id not in self.task_to_node_idx:
                return []
            node_idx = self.task_to_node_idx[task_id]
            neighbor_indices = self.graph.neighbors(node_idx)
            edges_list = []
            for neighbor_idx in neighbor_indices:
                edge_data = self.graph.get_edge_data(node_idx, neighbor_idx)
                edges_list.append((task_id,
                                 self.node_idx_to_task[neighbor_idx],
                                 edge_data))
            return edges_list
        else:
            edges_list = []
            edge_list = self.graph.edge_list()
            for (node_a, node_b) in edge_list:
                edge_data = self.graph.get_edge_data(node_a, node_b)
                edges_list.append((self.node_idx_to_task[node_a],
                                 self.node_idx_to_task[node_b],
                                 edge_data))
            return edges_list

    def _compute_task_similarity_matrix(self) -> np.ndarray:
        """Compute similarity matrix between all tasks once"""
        n_tasks = len(self.task_configs)

        # Compute actual task similarities based on task vectors
        similarity_matrix = np.zeros((n_tasks, n_tasks))
        for i in range(n_tasks):
            for j in range(n_tasks):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Compute similarity based on task vectors
                    task_i_vec = np.array(self.task_configs[i].task_vec)
                    task_j_vec = np.array(self.task_configs[j].task_vec)
                    # Use cosine similarity or Euclidean distance
                    distance = np.linalg.norm(task_i_vec - task_j_vec)
                    similarity_matrix[i, j] = 1.0 / (1.0 + distance)
        return similarity_matrix

    def get_task_neighbors(self, task_id: int, threshold: float = 0.0, num_neighbors: int = 8) -> List[int]:
        """Get neighboring tasks for a given task.

        Behavior depends on use_random_neighborhood flag:
        - If True: Randomly samples num_neighbors tasks, then sorts by similarity
        - If False: Returns up to num_neighbors graph neighbors with similarity >= threshold

        Args:
            task_id: ID of the focal task
            threshold: Minimum similarity threshold (only used if not random neighborhood)
            num_neighbors: Maximum number of neighbors to return

        Returns:
            List of neighbor task IDs, sorted by similarity (descending)
        """
        if self.use_random_neighborhood:
            available = [cfg.task_id for cfg in self.task_configs if cfg.task_id != task_id]
            if not available:
                return []
            k = min(num_neighbors, len(available))
            sampled = np.random.choice(available, size=k, replace=False).tolist()

            # Sort sampled neighbors by similarity to task_id
            if task_id in self.task_index and self.similarity_matrix is not None:
                t_idx = self.task_index[task_id]
                sampled.sort(key=lambda tid: self.similarity_matrix[t_idx, self.task_index[tid]], reverse=True)
            return sampled
        else:
            if task_id not in self.task_to_node_idx:
                return []
            cache_key = (task_id, threshold, num_neighbors)
            with self.cache_lock:
                if cache_key in self.neighbors_cache:
                    return self.neighbors_cache[cache_key]

            node_idx = self.task_to_node_idx[task_id]
            neighbor_indices = self.graph.neighbors(node_idx)  # rustworkx does NOT include self
            neighbors_with_similarity = []
            for neighbor_idx in neighbor_indices:
                neighbor_task_id = self.node_idx_to_task[neighbor_idx]
                edge_similarity = self.graph.get_edge_data(node_idx, neighbor_idx)
                if edge_similarity >= threshold:
                    neighbors_with_similarity.append((neighbor_task_id, edge_similarity))
            neighbors_with_similarity.sort(key=lambda x: x[1], reverse=True)
            if self.distance_proportional_sampling:
                k = min(num_neighbors, len(neighbors_with_similarity))
                if k == 0:
                    neighbors = []
                elif k == len(neighbors_with_similarity):
                    # If we want all available neighbors, just return them
                    neighbors = [nid for nid, _ in neighbors_with_similarity]
                else:
                    total_similarity = sum(sim for _, sim in neighbors_with_similarity)
                    if total_similarity == 0:
                        probabilities = [1.0 / len(neighbors_with_similarity)] * len(neighbors_with_similarity)
                    else:
                        probabilities = [sim / total_similarity for _, sim in neighbors_with_similarity]
                    neighbors = np.random.choice(
                        [nid for nid, _ in neighbors_with_similarity],
                        size=k,
                        replace=False,
                        p=probabilities
                    ).tolist()
            else:
                neighbors = [nid for nid, _ in neighbors_with_similarity[:num_neighbors]]
            with self.cache_lock:
                self.neighbors_cache[cache_key] = neighbors
            return neighbors

    def add_node(self, task_id: int, solution: np.ndarray, fitness: float,
                 behavior: Tuple[float, float], n_eval: int, solution_id: Optional[int] = None):
        """Add or update solution for a task node.

        Thread-safe update of node data. Appends to last_updated list.

        Args:
            task_id: ID of task to update
            solution: Solution vector (will be copied)
            fitness: Fitness value for the solution
            behavior: Behavior descriptor tuple (currently placeholder)
            n_eval: Current evaluation count
            solution_id: Optional solution identifier (defaults to n_eval)

        Raises:
            ValueError: If task_id not found in graph
        """
        with self.node_lock:
            if task_id not in self.task_to_node_idx:
                raise ValueError(f"Task {task_id} not found in graph")
            node_idx = self.task_to_node_idx[task_id]
            node_data = self.graph[node_idx]

            # Deep copy only if solution is not None to avoid unnecessary overhead
            sol = solution.copy() if solution is not None else None

            # Maintain last_updated as a list
            lu = node_data.get('last_updated')
            if not isinstance(lu, list):
                lu = []
            lu.append(n_eval)

            node_data.update({
                'solution': sol,
                'fitness': fitness,
                'behavior': behavior,
                'last_updated': lu
            })
            node_data['solution_id'] = n_eval if solution_id is None else solution_id

    def get_node(self, task_id: int) -> Dict[str, Any]:
        """Get node data for a task (thread-safe copy).

        Args:
            task_id: ID of task to retrieve

        Returns:
            Dictionary containing node data (copy), or empty dict if task not found
        """
        with self.node_lock:
            if task_id not in self.task_to_node_idx:
                return {}
            node_idx = self.task_to_node_idx[task_id]
            return self.graph[node_idx].copy()

    def get_candidate_neighbor(self, task_id: int, strategy: str = "best_fitness",
                               threshold: float = 0.0, num_neighbors: int = 50,
                               top_k: int = 3) -> Optional[Union[int, List[int]]]:
        """Select a neighbor task ID according to specified strategy.

        Args:
            task_id: Focal task ID
            strategy: Selection strategy - one of:
                - 'best_fitness': Select neighbor with highest fitness
                - 'most_similar': Random choice from top_k most similar neighbors
                - 'least_similar': Random choice from top_k least similar neighbors
                - 'random': Uniformly random neighbor
                - 'fitness_proportional': Sample proportional to fitness
                - 'plane_dd': Returns list [best_fitness, random] for plane directional diversity
            threshold: Similarity threshold for neighbor filtering
            num_neighbors: Max neighbors to consider
            top_k: Number of top candidates for most/least similar strategies

        Returns:
            Single neighbor task_id, or list of [best, random] task_ids for plane_dd strategy,
            or None if no valid neighbors exist
        """
        neighbors = self.get_task_neighbors(task_id, threshold=threshold,
                                            num_neighbors=num_neighbors)
        if not neighbors:
            return None

        if strategy == "best_fitness":
            best_neighbor = max(
                neighbors,
                key=lambda nid: self.get_node(nid).get('fitness', -float('inf'))
            )
            return best_neighbor

        elif strategy == "most_similar":
            # neighbors already ordered by similarity descending (unless random neighborhood mode)
            k = min(top_k, len(neighbors))
            candidate_pool = neighbors[:k]
            return int(np.random.choice(candidate_pool))

        elif strategy == "least_similar":
            k = min(top_k, len(neighbors))
            candidate_pool = neighbors[-k:]
            return int(np.random.choice(candidate_pool))

        elif strategy == "random":
            return int(np.random.choice(neighbors))

        elif strategy == "fitness_proportional":
            # Filter out neighbors with invalid fitness values
            valid_neighbors = []
            valid_fitnesses = []
            for nid in neighbors:
                fitness = self.get_node(nid).get('fitness', -float('inf'))
                if np.isfinite(fitness) and fitness > -float('inf'):
                    valid_neighbors.append(nid)
                    valid_fitnesses.append(fitness)

            if not valid_neighbors:
                # Fallback to random if no valid fitness values
                return int(np.random.choice(neighbors)) if neighbors else None

            valid_fitnesses = np.array(valid_fitnesses)

            # Check if all fitness values are the same
            if np.allclose(valid_fitnesses, valid_fitnesses[0]):
                # If all fitness values are equal, use uniform random selection
                return int(np.random.choice(valid_neighbors))

            # Shift to positive values if needed
            min_fitness = np.min(valid_fitnesses)
            if min_fitness <= 0:
                valid_fitnesses = valid_fitnesses - min_fitness + 1e-6

            # Calculate probabilities
            total_fitness = np.sum(valid_fitnesses)
            if total_fitness > 0 and np.isfinite(total_fitness):
                probabilities = valid_fitnesses / total_fitness
                # Normalize to handle floating point errors
                probabilities = probabilities / np.sum(probabilities)
                chosen_neighbor = np.random.choice(valid_neighbors, p=probabilities)
                return int(chosen_neighbor)

            # Final fallback to random selection
            return int(np.random.choice(valid_neighbors))
        elif strategy == "plane_dd":
            best_neighbor = max(
                neighbors,
                key=lambda nid: self.get_node(nid).get('fitness', -float('inf'))
            )
            random_neighbor = int(np.random.choice(neighbors))
            return [best_neighbor, random_neighbor]
        else:
            raise ValueError(f"Unknown neighbor selection strategy: {strategy}")

    def _prepare_candidate(self, task_id: int, cfg, p_min, p_max,
                           num_neighbors: int, strategy: str, threshold: float):
        """Prepare a candidate solution (prepare-only, no evaluation).

        Mirrors the individual/social learning logic but returns the candidate
        without evaluating it, so evaluations can be batched and run in parallel.

        Returns:
            (solution, current_fitness)
        """
        node = self.get_node(task_id)
        current_fitness = node.get('fitness', -float('inf'))

        if node.get('solution') is None:
            # No solution yet: copy from neighbor or init randomly
            if np.random.rand() >= cfg.algorithm.p_ind:
                neighbor_id = self.get_candidate_neighbor(
                    task_id, strategy=strategy, threshold=threshold, num_neighbors=num_neighbors)
                neighbor_node = self.get_node(neighbor_id) if neighbor_id is not None else None
                if (cfg.algorithm.init_strategy == 'copy'
                        and neighbor_node is not None
                        and neighbor_node.get('solution') is not None):
                    solution = neighbor_node['solution'].copy()
                else:
                    solution = np.random.uniform(low=p_min, high=p_max, size=self.solution_dim)
            else:
                solution = np.random.uniform(low=p_min, high=p_max, size=self.solution_dim)
            return solution, current_fitness

        if np.random.rand() < cfg.algorithm.p_ind:
            # Individual learning
            op = np.random.choice(list(cfg.algorithm.individual_learning))
            if op == 'gaussian_mutation':
                solution = gaussian_mutation(node['solution'], cfg.algorithm.mutation_std, p_min, p_max)
            elif op == 'polynomial_mutation':
                solution = polynomial_mutation(node['solution'], p_min, p_max)
            else:
                raise ValueError(f"Unknown individual learning op: {op}")
        else:
            # Social learning
            neighbor_id = self.get_candidate_neighbor(
                task_id, strategy=strategy, threshold=threshold, num_neighbors=num_neighbors)
            neighbor_node = self.get_node(neighbor_id) if neighbor_id is not None else None

            if neighbor_node is None or neighbor_node.get('solution') is None:
                solution = gaussian_mutation(node['solution'], cfg.algorithm.mutation_std, p_min, p_max)
            else:
                op = np.random.choice(list(cfg.algorithm.social_learning))
                if op == 'sbx':
                    solution = sbx(node['solution'], neighbor_node['solution'], p_min, p_max)
                elif op == 'iso_dd':
                    solution = iso_dd(node['solution'], neighbor_node['solution'], p_min, p_max,
                                      cfg.algorithm.iso_sigma, cfg.algorithm.line_sigma)
                elif op == 'regression':
                    neighbors = self.get_task_neighbors(task_id, threshold=0.0, num_neighbors=num_neighbors)
                    neighbor_nodes = [self.get_node(nid) for nid in neighbors
                                      if self.get_node(nid).get('solution') is not None]
                    if len(neighbor_nodes) < 2:
                        solution = gaussian_mutation(node['solution'], cfg.algorithm.mutation_std, p_min, p_max)
                    else:
                        solution = regression_monet(
                            task=self.task_configs[self.task_index[task_id]].task_vec,
                            neighbors=neighbor_nodes)
                else:
                    raise ValueError(f"Unknown social learning op: {op}")

        return solution, current_fitness

    def individual_learning(self, task_id: int, n_eval: int, cfg, p_min: np.ndarray,
                           p_max: np.ndarray) -> str:
        """Apply individual learning (mutation) to a task's solution.

        Mutates the current solution and updates if improvement found.
        Initializes random solution if none exists.

        Args:
            task_id: ID of task to apply learning to
            n_eval: Current evaluation count
            cfg: Configuration object with algorithm settings
            p_min: Minimum bounds for solution space
            p_max: Maximum bounds for solution space

        Returns:
            "success" if fitness improved, "failed" otherwise
        """
        current_node = self.get_node(task_id)

        if not current_node:
            raise ValueError(f"Task {task_id} not found in graph")
        if current_node.get('solution') is None:
            # Initialize with random solution and evaluate it
            init_sol = np.random.uniform(low=p_min, high=p_max, size=self.solution_dim)
            temp_node = {
                'solution': init_sol,
                "task_config": self.task_configs[self.task_index[task_id]].to_dict()
            }
            init_fitness, init_behavior = self.evaluate_node(temp_node)
            self.add_node(
                task_id=task_id,
                solution=init_sol,
                fitness=init_fitness,
                behavior=init_behavior,
                n_eval=n_eval,
                solution_id=n_eval,
            )
            # The node was just initialized, which counts as one evaluation step.
            # Return "success" as a new solution was added.
            return "success"

        individual_strategy = np.random.choice(list(cfg.algorithm.individual_learning))

        if individual_strategy == 'gaussian_mutation':
            y = gaussian_mutation(
                current_node['solution'], # function makes its own copy
                mutation_std=cfg.algorithm.mutation_std,
                p_min=p_min,
                p_max=p_max
            )
        elif individual_strategy == 'polynomial_mutation':
            y = polynomial_mutation(
                current_node['solution'], # function makes its own copy
                p_min=p_min,
                p_max=p_max
            )
        else:
            raise ValueError(f"Unknown individual learning type: {cfg.algorithm.individual_learning}")

        new_fitness, new_behavior = self.evaluate_node(
            {
                'solution': y,
                "task_config": self.task_configs[self.task_index[task_id]].to_dict()
            }
        )
        if new_fitness >= current_node.get('fitness', -float('inf')):
            self.add_node(task_id,
                          y,
                          fitness=new_fitness,
                          behavior=new_behavior,
                          n_eval=n_eval,
                          solution_id=n_eval)
            # update success count
            individual_learning_success = current_node.get('individual_learning_success', 0) + 1
            with self.node_lock:
                node_idx = self.task_to_node_idx[task_id]
                node_data = self.graph[node_idx]
                node_data['individual_learning_success'] = individual_learning_success
            return "success"
        else:
            # update fail count
            individual_learning_fail = current_node.get('individual_learning_fail', 0) + 1
            with self.node_lock:
                node_idx = self.task_to_node_idx[task_id]
                node_data = self.graph[node_idx]
                node_data['individual_learning_fail'] = individual_learning_fail
            return "failed"

    def social_learning(self, task_id: int, neighbor_id: int, n_eval: int, cfg,
                       p_min: np.ndarray, p_max: np.ndarray, num_neighbors: int) -> Optional[str]:
        """Apply social learning (crossover) using a neighbor's solution.

        Combines current solution with neighbor's solution. If current node
        has no solution, initializes based on cfg.algorithm.init_strategy.

        Args:
            task_id: ID of task to apply learning to
            neighbor_id: ID of neighbor task to learn from
            n_eval: Current evaluation count
            cfg: Configuration object with algorithm settings
            p_min: Minimum bounds for solution space
            p_max: Maximum bounds for solution space

        Returns:
            "success" if fitness improved, "failed" if no improvement,
            None if neighbor has no valid solution
        """
        current_node = self.get_node(task_id)
        neighbor_node = self.get_node(neighbor_id) if neighbor_id is not None else None
        current_fitness = current_node.get('fitness', -float('inf'))
        has_no_solution = current_node.get('solution') is None

        if has_no_solution:
            if cfg.algorithm.init_strategy == 'copy':
                if neighbor_id is None or neighbor_node is None or neighbor_node.get('solution') is None:
                    # no neighbor solution to copy from, initialize randomly
                    solution = np.random.uniform(low=p_min, high=p_max, size=self.solution_dim)
                else:
                    solution = neighbor_node['solution'].copy()  # initialize from neighbor
            else:
                # random initialization as alternative
                solution = np.random.uniform(low=p_min, high=p_max, size=self.solution_dim)

            # Evaluate the initialized solution
            new_fitness, new_behavior = self.evaluate_node({
                'solution': solution,
                "task_config": self.task_configs[self.task_index[task_id]].to_dict()
            })
            self.add_node(
                    task_id=task_id,
                    solution=solution,
                    fitness=new_fitness,
                    behavior=new_behavior,
                    n_eval=n_eval,
                    solution_id=n_eval,
                )
            return "success"

        if neighbor_id is None or neighbor_node is None or neighbor_node.get('solution') is None:
            # cannot perform social learning without a valid neighbor solution
            return None

        social_strategy = np.random.choice(list(cfg.algorithm.social_learning))

        # perform social learning if neighbor solution exists
        if social_strategy == 'sbx':
            y = sbx(
                current_node['solution'],
                neighbor_node['solution'],
                p_min=p_min,
                p_max=p_max
            )
        elif social_strategy == 'iso_dd':
            y = iso_dd(
                current_node['solution'],
                neighbor_node['solution'],
                p_min=p_min,
                p_max=p_max,
                iso_sigma=cfg.algorithm.iso_sigma,
                line_sigma=cfg.algorithm.line_sigma
            )
        elif social_strategy == 'regression':
            neighbors = self.get_task_neighbors(task_id,
                                                threshold=0.0,
                                                num_neighbors=num_neighbors)
            neighbor_nodes = [self.get_node(nid) for nid in neighbors if self.get_node(nid).get('solution') is not None]
            if len(neighbor_nodes) < 2:
                # Not enough neighbors with solutions to perform regression
                return None
            y = regression_monet(
                task=self.task_configs[self.task_index[task_id]].task_vec,
                neighbors=neighbor_nodes
            )
        else:
            raise ValueError(f"Unknown social learning type: {cfg.algorithm.social_learning}")

        new_fitness, new_behavior = self.evaluate_node({
            'solution': y,
            "task_config": self.task_configs[self.task_index[task_id]].to_dict()
        })

        if new_fitness >= current_fitness:
            self.add_node(task_id,
                          y,
                          fitness=new_fitness,
                          behavior=new_behavior,
                          n_eval=n_eval,
                          solution_id=n_eval)

            # update success count
            social_learning_success = current_node.get('social_learning_success', 0) + 1
            with self.node_lock:
                node_idx = self.task_to_node_idx[task_id]
                node_data = self.graph[node_idx]
                node_data['social_learning_success'] = social_learning_success
            return "success"
        else:
            # update fail count
            social_learning_fail = current_node.get('social_learning_fail', 0) + 1
            with self.node_lock:
                node_idx = self.task_to_node_idx[task_id]
                node_data = self.graph[node_idx]
                node_data['social_learning_fail'] = social_learning_fail
            return "failed"

    def evaluate_node(self, node: Dict[str, Any]) -> Tuple[float, Tuple[float, float]]:
        """Evaluate a solution on its assigned task.

        Args:
            node: Dictionary with 'task_config' and 'solution' keys

        Returns:
            Tuple of (fitness, behavior_descriptor)
        """
        task = self.environments[node["task_config"]["task_id"]]
        fitness = task.evaluate_solution(node["solution"])
        behavior = (0.0, 0.0)  # TODO: placeholder for behavior descriptor
        return fitness, behavior

    def cleanup(self):
        """Clean up resources including environments and graph data."""
        for env in self.environments.values():
            env.close()
        # Clear the graph
        self.graph.clear()

    def _get_graph_metrics(self) -> Dict[str, float]:
        """Calculate detailed graph structure metrics.

        Returns:
            Dictionary with keys:
            - graph/total_nodes: Number of nodes in graph
            - graph/total_edges: Number of edges in graph
            - graph/nodes_with_solutions: Count of nodes with valid solutions
            - graph/solution_coverage: Fraction of nodes with solutions
            - graph/avg_degree: Mean node degree
            - graph/max_degree: Maximum node degree
            - graph/min_degree: Minimum node degree
            - graph/degree_std: Standard deviation of node degrees
        """
        total_nodes = len(self.graph)
        total_edges = len(self.graph.edge_list())

        # Count nodes with solutions
        nodes_with_solutions = sum(1 for task_id in self.nodes()
                            if self.get_node(task_id).get('solution') is not None)

        # Calculate average degree
        degrees = [self.degree(task_id) for task_id in self.nodes()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0

        return {
            "graph/total_nodes": total_nodes,
            "graph/total_edges": total_edges,
            "graph/nodes_with_solutions": nodes_with_solutions,
            "graph/solution_coverage": nodes_with_solutions / total_nodes if total_nodes > 0 else 0,
            "graph/avg_degree": avg_degree,
            "graph/max_degree": max_degree,
            "graph/min_degree": min_degree,
            "graph/degree_std": np.std(degrees) if degrees else 0
        }

    def _get_task_metrics(self) -> Dict[str, float]:
        """Calculate task-specific performance metrics.

        Returns:
            Dictionary with metrics about task fitness and update frequency
        """
        task_fitnesses = {}
        task_update_counts = {}

        for task_id in self.nodes():
            node_data = self.get_node(task_id)
            fitness = node_data.get('fitness', -float('inf'))
            last_updated = node_data.get('last_updated', [])

            if fitness > -float('inf'):
                task_fitnesses[task_id] = fitness
            task_update_counts[task_id] = len(last_updated)

        update_counts = list(task_update_counts.values())

        metrics = {
            "tasks/total_tasks": len(list(self.nodes())),
            "tasks/tasks_with_fitness": len(task_fitnesses),
            "tasks/avg_updates_per_task": np.mean(update_counts) if update_counts else 0,
            "tasks/std_updates_per_task": np.std(update_counts) if update_counts else 0,
            "tasks/max_updates_per_task": max(update_counts) if update_counts else 0,
            "tasks/min_updates_per_task": min(update_counts) if update_counts else 0
        }

        # Add per-task fitness distribution
        if task_fitnesses:
            fitness_values = list(task_fitnesses.values())
            metrics.update({
                "tasks/fitness_range": max(fitness_values) - min(fitness_values),
                "tasks/fitness_variance": np.var(fitness_values)
            })

        return metrics

    def get_graph_statistics(self, n_eval: int = 0) -> Dict[str, Any]:
        """Get comprehensive graph statistics for logging.

        Args:
            n_eval: Current evaluation count

        Returns:
            Dictionary containing:
            - size: Number of nodes with valid solutions
            - fitness: Dict with max, mean, median, std of fitness values
            - n_eval: Current evaluation count
        """
        # Get all nodes with solutions
        nodes_with_solutions = []
        for task_id in self.nodes():
            node_data = self.get_node(task_id)
            if node_data.get('solution') is not None and node_data.get('fitness', -float('inf')) > -float('inf'):
                nodes_with_solutions.append(node_data)

        if not nodes_with_solutions:
            return {
                'size': 0,
                'fitness': {'max': 0, 'mean': 0, 'median': 0, 'std': 0},
                'n_eval': n_eval
            }

        fitness_values = [node['fitness'] for node in nodes_with_solutions]

        return {
            'size': len(nodes_with_solutions),
            'fitness': {
                'max': max(fitness_values),
                'mean': np.mean(fitness_values),
                'median': np.median(fitness_values),
                'std': np.std(fitness_values)
            },
            'n_eval': n_eval
        }