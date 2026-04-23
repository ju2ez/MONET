import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
from scipy.spatial import distance, cKDTree, Delaunay
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import multiprocessing
import logging
import math
import pdb
from environments.base_env import BaseEnv
from environments.base_task_config import BaseTaskConfig
import wandb
from utils.visualization import log_mtme_archive_to_wandb, log_ptme_archive_to_wandb
from algorithms.common import AUCTracker

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# hardcoded values
HEXAPOD_MEAN = np.array([0.1408228 , 0.14108632, 0.1398777 , 0.13988906, 0.13929189,
       0.13985942, 0.08455675, 0.08512714, 0.08525372, 0.08463292,
       0.08441107, 0.08493869])
HEXAPOD_STD = np.array([0.0231802 , 0.02303754, 0.02299927, 0.02356229, 0.02290068,
       0.02325215, 0.02292695, 0.02274292, 0.02246502, 0.02314502,
       0.02315888, 0.02281904])

def cvt(k: int, dim: int, coef: int = 10, verbose: bool = False, seed: int = 42) -> np.ndarray:
    """Generate k centroids using Centroidal Voronoi Tessellation."""
    try:
        np.random.seed(seed)
        x = np.random.rand(k * coef, dim)
        k_means = KMeans(init='k-means++', n_clusters=k, n_init=1, verbose=verbose, random_state=seed)
        k_means.fit(x)
        return k_means.cluster_centers_
    except Exception as e:
        logger.error(f"Error in CVT generation: {e}")
        return np.random.random((k, dim))

class Archive:
    def __init__(self, task_config, task_env, solution_dim, n_cells=200,lower_bound=0.0, upper_bound=1.0, ptme=None, seed=42) -> None:
        """Initialize the archive with configuration."""
        self.solution_dim = solution_dim
        self.task_dim = len(task_config[0].task_vec)
        self.task_config = task_config
        self.task_env = task_env
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.k_closest2parent = 1
        self.bandit_values = [1, 5, 10, 50, 100, 500]
        self.bandit_closest2parent = Bandit(self.bandit_values)
        self.ptme = ptme
        self.seed = seed

        self.samples: List[Dict[str, Any]] = []
        self.it = 0
        self.n_cells = n_cells

        logger.info(f"Initializing archive with {self.n_cells} cells")
        self.create_centroids()
        self.tree = cKDTree(self.centroids, leafsize=2)
        self.elites: List[Optional[Dict[str, Any]]] = [None for _ in range(self.n_cells)]
        self._initialize_elites()

    def create_centroids(self) -> None:
        """Create centroids and compute neighborhood structure."""
        try:
            if self.n_cells <= 10_000:
                logger.info(f"Generating {self.n_cells} centroids using CVT of dimension {self.task_dim}")
                self.centroids = cvt(self.n_cells, self.task_dim, seed=self.seed)
            else:
                logger.info(f"Generating {self.n_cells} random centroids of dimension {self.task_dim}")
                self.centroids = np.random.random((self.n_cells, self.task_dim))

            if self.task_dim == 12:
                logger.info("Using precomputed centroids for Hexapod task space")
                self.centroids = [t.task_vec for t in self.task_config]
                logger.info(f"Found {len(self.centroids)} centroids from task configuration")

            if self.task_dim > 10:
                logger.info("Computing centroid neighborhood structure using k-NN")
                # Use k-NN instead of Delaunay for high dimensions
                k_neighbors = min(15, self.n_cells - 1)  # Adjust k as needed
                tree = cKDTree(self.centroids)
                neighbors = [list(tree.query(self.centroids[i], k=k_neighbors+1)[1])
                            for i in range(self.n_cells)]
                self.centroid_neighbors = neighbors
                logger.info(f"Centroid neighborhood structure computed using {k_neighbors}-NN")
            else:
                logger.info("Computing centroid neighborhood structure using Delaunay triangulation")
                delauney = Delaunay(self.centroids)
                neighbors = [[i] for i in range(self.n_cells)]
                for neighborhood in delauney.simplices:
                    for i in neighborhood:
                        for j in neighborhood:
                            neighbors[i].append(j)
                            neighbors[j].append(i)
                self.centroid_neighbors = [list(set(n)) for n in neighbors]
                logger.info("Centroid neighborhood structure computed")

        except Exception as e:
            logger.error(f"Error creating centroids: {e}")
            # Fallback to random centroids
            self.centroids = np.random.random((self.n_cells, self.task_dim))
            self.centroid_neighbors = [[i] for i in range(self.n_cells)]

    def _initialize_elites(self) -> None:
        """Initialize elites with random solutions."""
        logger.info("Initializing elites with random solutions")
        for i, task in enumerate(self.task_config):
            # Generate random solution (command), not using centroid
            random_solution = np.random.rand(self.solution_dim)
            # Set task vector to centroid for evaluation
            env = self.task_env(task=task)
            r = env.evaluate_solution(random_solution)  # Pass solution, not centroid

            evaluation = {"reward": r}
            evaluation["command"] = random_solution
            evaluation["situation"] = task.task_vec
            evaluation["id"] = i
            evaluation["kind"] = "random"
            evaluation["it"] = self.it

            self.samples.append(evaluation)
            self.it += 1
            self.elites[i] = evaluation

            # Log statistics
            if self.ptme is not None:
                stats = self.ptme.calculate_archive_stats(self)
                self.ptme.global_archive_stats.append(stats)
                logger.info(f"Iteration {self.it}: "
                            f"Archive size={len([e for e in self.elites if e is not None])}/{self.n_cells}, "
                            f"Coverage={stats['coverage']:.2%}, "
                            f"Mean fitness={stats['mean_fitness_elites']:.4f}, "
                            f"Max fitness={stats['max_fitness_elites']:.4f}")

            if self.ptme is not None and getattr(self.ptme, 'use_wandb', False):
                wandb.log({
                    "iteration": self.it,
                    "n_evals" : self.it,
                    "archive_size": len([e for e in self.elites if e is not None]),
                    "coverage": stats['coverage'],
                    "mean_fitness": stats['mean_fitness_elites'],
                    "median_fitness": stats['median_fitness_elites'],
                    "max_fitness": stats['max_fitness_elites'],
                    "mean_fitness_samples": stats['mean_fitness_samples'],
                    "median_fitness_samples": stats['median_fitness_samples'],
                    "qd_score": stats['qd_score'],
                    "k_closest2parent": self.k_closest2parent,
                    "success_variation": 0,
                    "fail_variation": 0
                })

        logger.info(f"Archive initialized with {self.n_cells} random elites")

    def add_evaluation(self, evaluation: Dict[str, Any]) -> bool:
        """Add evaluation to archive and update elite if better."""
        try:
            _, index = self.tree.query(evaluation["situation"], k=1)
            evaluation["it"] = self.it
            self.samples.append(evaluation)
            self.it += 1

            current_elite = self.elites[index]
            is_elite = (current_elite is None or
                       evaluation["reward"] >= current_elite["reward"])

            if is_elite:
                self.elites[index] = evaluation

            if "closest2parent" in evaluation.get("kind", ""):
                self.k_closest2parent = self.bandit_closest2parent.update(
                    self.k_closest2parent, is_elite
                )

            return is_elite
        except Exception as e:
            logger.error(f"Error adding evaluation: {e}")
            return False

class Bandit:
    def __init__(self, values: List[Union[int, float]]) -> None:
        """Initialize multi-armed bandit."""
        self.successes = defaultdict(int)
        self.selected = defaultdict(int)
        self.log: List[Union[int, float]] = []
        self.values = values

    def update(self, key: Union[int, float], success: bool) -> Union[int, float]:
        """Update bandit with success/failure and return next arm to pull."""
        self.successes[key] += int(success)
        self.selected[key] += 1

        total_selections = sum(self.selected.values())

        # Exploration phase: try all arms at least once
        if len(self.selected.keys()) < len(self.values):
            unselected = [v for v in self.values if v not in self.selected]
            selected_arm = random.choice(unselected)
        else:
            # UCB1 algorithm
            ucb_values = []
            for arm in self.values:
                n_arm = self.selected[arm]
                mean_reward = self.successes[arm] / n_arm
                confidence = math.sqrt(2 * math.log(total_selections) / n_arm)
                ucb_values.append(mean_reward + confidence)

            selected_arm = self.values[np.argmax(ucb_values)]

        self.log.append(selected_arm)
        return selected_arm


class PTME:
    def __init__(self,
                 task_configs: List[BaseTaskConfig],
                 task_env: BaseEnv,
                 solution_dim: int = 10,
                 seed: int = 42,
                 budget: int = 1_000_000,
                 n_cells: int = 200,
                 proba_regression: float = 0.5,
                 variation_operator: str = "sbx",
                 lingreg_sigma: float = 1.,
                 wandb: bool = False,
                 log_frequency: int = 10_000,
                 verbose: bool = False,
                 ptme = None,
                ) -> None:
        """Initialize PT-ME algorithm with variation operators."""

        self.variation_operators = {
            #"iso-line-dd": self.iso_line_dd, # TODO: add iso-line-dd
            "sbx": self.sbx,
        }
        self.global_archive_stats: List[Dict[str, float]] = []
        self.task_configs = task_configs
        self.task_env = task_env
        self.solution_dim = solution_dim
        self.task_dim = len(task_configs[0].task_vec)
        self.seed = seed

        # Config variables
        self.budget = budget
        self.n_cells = n_cells
        self.proba_regression = proba_regression
        self.variation_operator = variation_operator
        self.linreg_sigma = lingreg_sigma
        self.use_wandb = wandb
        self.log_frequency = log_frequency
        self.verbose = verbose

        logger.info(f"PTME initialized with solution_dim={solution_dim}, seed={seed}, "
                   f"budget={budget}, n_cells={n_cells}")

    def calculate_archive_stats(self, archive: Archive) -> Dict[str, float]:
        """
        Calculate statistics of the archive.
        Mean and median fitness of elites and of all samples.
        """
        elite_rewards = [elite["reward"] for elite in archive.elites if elite is not None]
        sample_rewards = [sample["reward"] for sample in archive.samples]

        if not elite_rewards:
            logger.warning("No elites in archive for stats calculation")
            elite_rewards = [0.0]
        if not sample_rewards:
            logger.warning("No samples in archive for stats calculation")
            sample_rewards = [0.0]

        return {
            "mean_fitness_elites": np.mean(elite_rewards),
            "median_fitness_elites": np.median(elite_rewards),
            "mean_fitness_samples": np.mean(sample_rewards),
            "median_fitness_samples": np.median(sample_rewards),
            "max_fitness_elites": np.max(elite_rewards),
            "qd_score": np.sum(elite_rewards),
            "coverage": len([e for e in archive.elites if e is not None]) / archive.n_cells
        }
    def closest2parent_tournament(self, s, archive, config):
        k = archive.k_closest2parent
        # tasks = np.random.random((k, config["task_dim"]))
        tasks = np.array([self.sample_task() for _ in range(k)])
        _, indexes = archive.tree.query(tasks, k=1)
        situations = [archive.elites[i]["situation"] for i in indexes]
        distances = distance.cdist(situations, [s], "euclidean")
        selected_task = tasks[np.argmin(distances)]
        return selected_task

    def regression(self, s, archive, config):
        """ local linear regression  """
        _, idx = archive.tree.query(s, k=1)
        indexes = archive.centroid_neighbors[idx]  # find the direct neighbors using the precomputed delauney from the centroids
        X = [archive.elites[i]["situation"] for i in indexes]
        Y = [archive.elites[i]["command"] for i in indexes]
        reg = LinearRegression().fit(X, Y)
        c = reg.predict(np.array([s]))[0]
        dim = len(c)
        return np.clip(c + np.random.normal(0, config["linreg_sigma"]) * np.std(Y, axis=0), np.zeros(dim), np.ones(dim))

    def sbx(self, x, y):
        '''
        SBX (cf Deb 2001, p 113) Simulated Binary Crossover

        A large value ef eta gives a higher probablitity for
        creating a `near-parent' solutions and a small value allows
        distant solutions to be selected as offspring.
        '''
        eta = 10 # command_config["eta"]
        xl = 0
        xu = 1
        z = x.copy()
        r1 = np.random.random(size=len(x))
        r2 = np.random.random(size=len(x))

        for i in range(0, len(x)):
            if abs(x[i] - y[i]) > 1e-15:

                x1 = min(x[i], y[i])
                x2 = max(x[i], y[i])

                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                rand = r1[i]

                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)

                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, xl), xu)
                c2 = min(max(c2, xl), xu)

                if r2[i] <= 0.5:
                    z[i] = c2
                else:
                    z[i] = c1
        return z

    def sample_task(self):
        if self.task_dim == 12:
            task = np.random.normal(HEXAPOD_MEAN, HEXAPOD_STD)
            task = np.clip(task, 0.0, 1.0)
            return task
        else:
            return np.random.random(self.task_dim)

    def run(self) -> Dict[str, Any]:
        logger.info("Starting PTME computation...")
        logger.info(f"Budget: {self.budget}, n_cells: {self.n_cells}")

        config = {
            "task_dim": len(self.task_configs[0].task_vec),
            "n_cells": self.n_cells,
            "budget": self.budget,
            "proba_regression": self.proba_regression,
            "variation_operator": self.variation_operator,
            "linreg_sigma": self.linreg_sigma,
            "verbose": self.verbose
        }
        archive = Archive(task_config=self.task_configs, task_env=self.task_env, solution_dim=self.solution_dim, n_cells=self.n_cells, ptme=self, seed=self.seed)

        success_variation = 0
        fail_variation = 0
        mean_fitness_auc = AUCTracker()

        for current_it in tqdm(range(archive.it, self.budget + 1), ncols=150, smoothing=0.01, mininterval=1.):
            if np.random.random() < self.proba_regression:
                # s = np.random.random(self.task_dim)
                s = self.sample_task()
                c = self.regression(s, archive, config)
                selected_operator = "regression"
            else:
                x, y = archive.elites[np.random.randint(archive.n_cells)], archive.elites[np.random.randint(archive.n_cells)]
                c = self.variation_operators[self.variation_operator](x["command"], y["command"])
                s = self.closest2parent_tournament(x["situation"], archive, config)
                selected_operator = "sbx_closest2parent"
            #sample = (c, s, current_it, self.command_config["bounds"], self.situation_config["bounds"], selected_operator, self.verbose)
            #evaluation = self.eval_command(*sample)
            self.task_configs[0].task_vec = s # we always overwrite the task config
            env = self.task_env(task=self.task_configs[0])
            r = env.evaluate_solution(c)
            evaluation = {"reward": r}
            evaluation["command"] = c
            evaluation["situation"] = s
            evaluation["id"] = current_it
            evaluation["kind"] = selected_operator
            is_elite = archive.add_evaluation(evaluation)

            # Track success/failure
            if is_elite:
                success_variation += 1
            else:
                fail_variation += 1

            # Log statistics
            if current_it % self.log_frequency == 0:
                stats = self.calculate_archive_stats(archive)
                self.global_archive_stats.append(stats)
                mean_fitness_auc.update(current_it, stats['mean_fitness_elites'])
                logger.info(f"Iteration {current_it}/{self.budget}: "
                           f"Archive size={len([e for e in archive.elites if e is not None])}/{archive.n_cells}, "
                           f"Coverage={stats['coverage']:.2%}, "
                           f"Mean fitness={stats['mean_fitness_elites']:.4f}, "
                           f"Max fitness={stats['max_fitness_elites']:.4f}, "
                           f"Mean fitness AUC={mean_fitness_auc.auc:.4f}")

                if self.use_wandb:
                    wandb.log({
                        "iteration": current_it,
                        "n_evals" : current_it,
                        "archive_size": len([e for e in archive.elites if e is not None]),
                        "coverage": stats['coverage'],
                        "mean_fitness": stats['mean_fitness_elites'],
                        "median_fitness": stats['median_fitness_elites'],
                        "max_fitness": stats['max_fitness_elites'],
                        "mean_fitness_samples": stats['mean_fitness_samples'],
                        "median_fitness_samples": stats['median_fitness_samples'],
                        "qd_score": stats['qd_score'],
                        "mean_fitness_auc": mean_fitness_auc.auc,
                        "k_closest2parent": archive.k_closest2parent,
                        "success_variation": success_variation,
                        "fail_variation": fail_variation
                    })

            if current_it % 100_000 == 0:
                log_ptme_archive_to_wandb(archive, current_it, self.use_wandb)


        # Final statistics
        final_stats = self.calculate_archive_stats(archive)
        mean_fitness_auc.update(self.budget, final_stats['mean_fitness_elites'])
        logger.info("PTME computation completed.")
        logger.info(f"Final Mean Fitness AUC: {mean_fitness_auc.auc}")
        logger.info(f"Final archive size: {len([e for e in archive.elites if e is not None])}/{archive.n_cells}")
        logger.info(f"Final coverage: {final_stats['coverage']:.2%}")
        logger.info(f"Final mean fitness: {final_stats['mean_fitness_elites']:.4f}")
        logger.info(f"Final max fitness: {final_stats['max_fitness_elites']:.4f}")
        logger.info(f"Success rate: {success_variation}/{success_variation + fail_variation} ({success_variation/(success_variation + fail_variation)*100:.2f}%)")

        if self.use_wandb:
            wandb.log({
                "n_evals": self.budget,
                "final_archive_size": len([e for e in archive.elites if e is not None]),
                "final_coverage": final_stats['coverage'],
                "coverage": final_stats['coverage'],
                "final_mean_fitness": final_stats['mean_fitness_elites'],
                "mean_fitness": final_stats['mean_fitness_elites'],
                "median_fitness": final_stats['median_fitness_elites'],
                "final_max_fitness": final_stats['max_fitness_elites'],
                "final_qd_score": final_stats['qd_score'],
                "mean_fitness_auc": mean_fitness_auc.auc,
                "total_success": success_variation,
                "total_fail": fail_variation
            })

            log_ptme_archive_to_wandb(archive, self.budget, self.use_wandb)

        print(f"final_mean_fitness: {final_stats['mean_fitness_elites']}")
        print(f"final_auc: {mean_fitness_auc.auc}")

        return {"archive": archive, "config": config}