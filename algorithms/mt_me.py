#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.
#

import numpy as np
import random
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from scipy.spatial import distance
import multiprocessing
import logging
import wandb

import algorithms.mt_me_common as cm
from algorithms.common import AUCTracker

from utils.visualization import log_mtme_archive_to_wandb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MTME:
    """
    Multi-task MAP-Elites with optional distance-based niche selection.

    Responsibilities:
    - Manage archive updates
    - Evaluate candidate solutions (multiprocessing)
    - Select niches via random or distance-based tournaments
    - Adapt tournament size with a bandit heuristic
    """

    def __init__(self, task_configs: List[Any], solution_dim: int = 10, seed: int = 42):
        self.task_configs = task_configs
        self.solution_dim = solution_dim
        self.seed = seed
        self.archive: Dict[Any, cm.Species] = {}

    def add_to_archive(self, species: cm.Species) -> int:
        """
        Add or replace a species in the archive based on its centroid and fitness.
        Returns 1 if added/replaced, else 0.
        """
        centroid = cm.make_hashable(species.centroid)
        if centroid in self.archive:
            if species.fitness > self.archive[centroid].fitness:
                self.archive[centroid] = species
                return 1
            return 0
        else:
            self.archive[centroid] = species
            return 1

    def evaluate(self, task: Tuple[np.ndarray, Any, int, Any, Dict[str, Any]]) -> cm.Species:
        """
        Evaluate a single candidate solution.
        """
        z, evaluator_factory, task_id, centroid, _ = task
        task_config = self.task_configs[task_id]
        evaluator = evaluator_factory(task_config)
        fitness = evaluator.evaluate_solution(z)
        return cm.Species(z, task_config, fitness, centroid)

    def evaluate_batch(self, tasks: List[Tuple[np.ndarray, Any, int, Any, Dict[str, Any]]], params: Dict[str, Any]) -> List[cm.Species]:
        """
        Evaluate a batch of candidate solutions in parallel.

        if not tasks:
            return []
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            return cm.parallel_eval(self.evaluate, tasks, pool, params)
        """
        if not tasks:
            return []

        for task in tasks:
            yield self.evaluate(task)

    def select_niche(self, parent: cm.Species, offspring: np.ndarray, evaluator_factory: Any, centroids: np.ndarray,
                     tasks: List[Any], tournament_size: int, params: Dict[str, Any], use_distance: bool) -> List[Tuple[np.ndarray, Any, int, Any, Dict[str, Any]]]:
        """
        Select a niche for the offspring based on random or distance-based selection.
        """
        if not use_distance:
            niche_idx = np.random.randint(len(tasks))
            return [(offspring, evaluator_factory, niche_idx, centroids[niche_idx], params)]

        # Distance-based tournament
        rand_indices = np.random.randint(len(centroids), size=tournament_size)
        niche_centroids = centroids[rand_indices]
        niche_tasks = [tasks[i] for i in rand_indices]
        distances = distance.cdist(niche_centroids, [parent.centroid], 'euclidean')
        best_idx = np.argmin(distances)
        return [(offspring, evaluator_factory, rand_indices[best_idx], niche_centroids[best_idx], params)]

    def bandit(self, successes: Dict[int, List[Tuple[int, int]]], total_niches: int) -> int:
        """
        Use a bandit algorithm to adaptively select the tournament size.
        """
        import math
        total_evals = sum(len(v) for v in successes.values())
        candidate_sizes = [1, 10, 50, 100, 500]
        if len(successes) < len(candidate_sizes):
            return random.choice(candidate_sizes)
        ucb_scores = []
        for size in candidate_sizes:
            trials = successes[size]
            mean_success = sum(s[0] for s in trials) / len(trials)
            exploration = math.sqrt(2 * math.log(total_evals) / len(trials))
            ucb_scores.append(mean_success + exploration)
        return candidate_sizes[np.argmax(ucb_scores)]

    def compute(self, dim_x: int, evaluator_factory: Any, num_evals: int, centroids: np.ndarray, tasks: List[Any],
                params: Dict[str, Any], log_file=None) -> Dict[Any, cm.Species]:
        """
        Main loop for multi-task MAP-Elites.
        """
        logger.info("Starting MTME computation...")
        self.wandb = params.get('wandb', False)
        use_distance = params.get('use_distance', True)
        n_tasks = len(tasks)

        n_evals = 0
        batch_evals = 0
        tournament_size = 1
        successes = defaultdict(list)
        sucess_variation = 0
        fail_variation = 0
        mean_fitness_auc = AUCTracker()

        init_evals = params['random_init'] * n_tasks

        while n_evals < num_evals:
            to_evaluate = []

            # Random initialization
            if n_evals == 0 or n_evals < init_evals:
                for _ in range(params['random_init_batch']):
                    x = cm.random_individual(dim_x, params)
                    niche_idx = np.random.randint(n_tasks)
                    to_evaluate.append((x, evaluator_factory, niche_idx, centroids[niche_idx], params))
            else:
                # Variation and selection
                keys = list(self.archive.keys())
                parent_indices = np.random.randint(len(keys), size=params['batch_size'])
                for idx in parent_indices:
                    parent = self.archive[keys[idx]]
                    offspring = cm.variation(parent.x, self.archive, params)
                    to_evaluate.extend(self.select_niche(parent, offspring, evaluator_factory, centroids, tasks,
                                                         tournament_size, params, use_distance))

            # Evaluate batch
            evaluated_species = self.evaluate_batch(to_evaluate, params)
            n_evals += len(to_evaluate)
            batch_evals += len(to_evaluate)

            # Update archive
            improvements = [self.add_to_archive(species) for species in evaluated_species]
            fail_variation += (len(improvements) - sum(improvements))
            improvements = sum(improvements)
            sucess_variation += improvements
            if use_distance:
                successes[tournament_size].append((improvements, n_evals))

            # Log progress
            if n_evals % 10_000 == 0:
                logger.info(f"Evaluations: {n_evals}/{num_evals}, Archive size: {len(self.archive)}")

            # Adjust tournament size
            if use_distance:
                tournament_size = self.bandit(successes, n_tasks)
            # Periodic archive dump
            #if params['dump_period'] > 0 and batch_evals >= params['dump_period']:
            if False:
                cm.save_archive(self.archive, n_evals)
                batch_evals = 0
                # print archive stats
                fitness_values = [species.fitness for species in self.archive.values()]
                logger.info(f"Archive dump at {n_evals} evals: Size={len(self.archive)}, Max fitness={max(fitness_values)}, Mean fitness={np.mean(fitness_values)}")

            # Log fitness stats
            if False:
                fitness_values = [species.fitness for species in self.archive.values()]
                log_file.write(f"{n_evals} {len(self.archive)} {max(fitness_values)} {np.mean(fitness_values)}\n")
                log_file.flush()

            if params['wandb']:
                if (n_evals % params['log_frequency'] == 0) or (n_evals < init_evals):
                    fitness_values = [species.fitness for species in self.archive.values()]
                    mean_fit = float(np.mean(fitness_values)) if fitness_values else 0.0
                    mean_fitness_auc.update(n_evals, mean_fit)
                    wandb.log({"n_evals": n_evals, "archive_size": len(self.archive), "coverage": len(self.archive) / (n_tasks * 1.0),
                            "median_fitness": np.median(fitness_values), "qd_score": sum(fitness_values),
                                "max_fitness": max(fitness_values), "mean_fitness": mean_fit,
                                "mean_fitness_auc": mean_fitness_auc.auc,
                                "tournament_size": tournament_size if use_distance else 0,
                            "success_variation": sucess_variation, "fail_variation": fail_variation})

                if n_evals % params['log_interval'] == 0:
                    log_mtme_archive_to_wandb(self.archive, n_evals, self.task_configs)

        #cm.__save_archive(self.archive, n_evals)
        fitness_values = [species.fitness for species in self.archive.values()]
        mean_fit = float(np.mean(fitness_values)) if fitness_values else 0.0
        mean_fitness_auc.update(n_evals, mean_fit)
        if self.wandb:
            wandb.log({"n_evals": n_evals, "archive_size": len(self.archive), "coverage": len(self.archive) / (n_tasks * 1.0),
                    "median_fitness": np.median(fitness_values), "qd_score": sum(fitness_values),
                        "max_fitness": max(fitness_values), "mean_fitness": mean_fit,
                        "mean_fitness_auc": mean_fitness_auc.auc,
                        "tournament_size": tournament_size if use_distance else 0,
                    "success_variation": sucess_variation, "fail_variation": fail_variation})
            log_mtme_archive_to_wandb(self.archive, n_evals, self.task_configs)
        logger.info(f"Final Mean Fitness AUC: {mean_fitness_auc.auc}")
        logger.info("MTME computation completed.")
        print(f"final_mean_fitness: {mean_fit}")
        print(f"final_auc: {mean_fitness_auc.auc}")
        return self.archive


    # a small test
    if __name__ == "__main__":
        def rastrigin(xx):
            x = xx * 10.0 - 5.0
            f = 10 * x.shape[0]
            for i in range(0, x.shape[0]):
                f += x[i] * x[i] - 10 * math.cos(2 * math.pi * x[i])
            return -f, np.array([xx[0], xx[1]])
        # CVT-based version
        my_map = compute(dim_map=2, dim_x = 10, n_niches=1500, f=rastrigin)
