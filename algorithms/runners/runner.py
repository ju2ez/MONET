import numpy as np
import pandas as pd
import random
import logging
import ast
from utils.general_utils import calculate_fitness_metrics, cvt
from utils.visualization import log_node_data_to_wandb
from algorithms.monet import MONET, run_monet
from algorithms.mt_me import MTME
from algorithms.pt_me import PTME
from algorithms.mt_me_common import default_params
import omegaconf
from omegaconf import open_dict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from utils import file_logger
log = logging.getLogger(__name__)

def run_algorithm(cfg, task_config, env, TaskConfig, seed, use_wandb):
    """ Run algorithm """
    algorithm_name = cfg.algorithm.name
    sub = cfg.algorithm.get(algorithm_name) or cfg.algorithm.get(algorithm_name.lower())
    if sub is None:
        raise ValueError(f"No sub-config for cfg.algorithm.{algorithm_name}")
    with open_dict(cfg):
        for key, value in sub.items():
            cfg.algorithm[key] = value

    if cfg.task == "arm":
        task_configs = []
        # use precomputed clusters (from PT-ME run)
        df = pd.read_csv('env_inits/arm_clusters.csv')
        centroids = np.array([np.array(ast.literal_eval(s)) for s in df['situation_vec']])
        for task_id in range(task_config.num_tasks):
            task_vec = centroids[task_id]
            task = TaskConfig(task_id=task_id, task_vec=task_vec)
            task.solution_dim = task_config.solution_dim
            task_configs.append(task)
        tasks = task_configs
        centroids = centroids[:task_config.num_tasks]
    elif cfg.task == "archery":
        task_configs = []
        # use precomputed clusters (from PT-ME run)
        df = pd.read_csv('env_inits/archery_clusters.csv')
        centroids = np.array([np.array(ast.literal_eval(s)) for s in df['situation_vec']])
        for task_id in range(task_config.num_tasks):
            task_vec = centroids[task_id]
            task = TaskConfig(task_id=task_id, task_vec=task_vec)
            task.solution_dim = task_config.solution_dim
            task_configs.append(task)
        tasks = task_configs
        centroids = centroids[:task_config.num_tasks]
    elif cfg.task == "hexapod":
        from environments.hexapod_env import load_tasks
        df = pd.read_csv('env_inits/hexapod_clusters.csv')
        _, tasks = load_tasks(task_config.task_directory, task_config.num_tasks)
        # there should be 0-11 scalars for situation_vec (not the best form to save it probably)
        centroids = []
        for i in range(12):
            centroids.append(df[f'situation_vec_{i}'].values)
        centroids = np.array(centroids).T
        centroids = centroids[:task_config.num_tasks]
        task_configs = []
        for task_id in range(task_config.num_tasks):
            t = tasks[task_id]
            task_vec = centroids[task_id]
            task = TaskConfig(task_id=task_id, task_vec=task_vec, urfd_file=t[1])
            task_configs.append(task)
        tasks = task_configs
    elif cfg.task == "cartpole":
        task_configs = []
       # use precomputed clusters (from PT-ME run)
        df = pd.read_csv('env_inits/cartpole_clusters.csv')
        centroids = np.array([np.array(ast.literal_eval(s)) for s in df['situation_vec']])
        for task_id in range(task_config.num_tasks):
            task_vec = centroids[task_id]
            task = TaskConfig(task_id=task_id, task_vec=task_vec)
            task.solution_dim = task_config.solution_dim
            task_configs.append(task)
        tasks = task_configs
        centroids = centroids[:task_config.num_tasks]

    if algorithm_name == "MONET":
        num_init = int(cfg.algorithm.num_init_percent * task_config.num_tasks)
        num_neighbors = int(cfg.algorithm.neighbor_percentage * task_config.num_tasks)

        if use_wandb:
            wandb.init(
                project=task_config.project,
                config={
                    "n_tasks": task_config.num_tasks,
                    "mutation_std": cfg.algorithm.mutation_std,
                    "solution_dim": task_config.solution_dim,
                    "task_dim": task_config.task_dim,
                    "max_evals": cfg.algorithm.max_evals,
                    "init_strategy": cfg.algorithm.init_strategy,
                    "strategy": cfg.algorithm.strategy,
                    "seed": seed,
                    "algorithm": cfg.algorithm.name,
                    "domain": task_config.domain,
                    "num_init": num_init,
                    "dir": cfg.wandb.dir,
                    "num_neighbors": num_neighbors,
                    "neighbor_percentage": cfg.algorithm.neighbor_percentage,
                    "threshold": cfg.algorithm.threshold,
                    "neighborhood": cfg.algorithm.neighborhood,
                    "copy_criterion": cfg.algorithm.copy_criterion,
                    "store_similarity_matrix": cfg.algorithm.store_similarity_matrix,
                    "individual_learning": cfg.algorithm.individual_learning,
                    "social_learning": cfg.algorithm.social_learning,
                    "p_ind": cfg.algorithm.p_ind,
                    "mutation_std" : cfg.algorithm.mutation_std,
                    "iso_sigma" : cfg.algorithm.iso_sigma,
                    "line_sigma" : cfg.algorithm.line_sigma,
                },
                name=f"monet_neighborhood_{cfg.algorithm.neighborhood}_num_neighbors_{num_neighbors}",
            )
        hyperparams = cfg.algorithm[f"{cfg.algorithm.name}"]
        with open_dict(cfg.algorithm):
            for key, value in hyperparams.items():
                cfg.algorithm[key] = value

        if cfg.algorithm.neighborhood == "random":
            use_random_neighborhood = True
            distance_proportional_sampling = False
        elif cfg.algorithm.neighborhood == "closest":
            use_random_neighborhood = False
            distance_proportional_sampling = False
        elif cfg.algorithm.neighborhood == "distance_proportional":
            use_random_neighborhood = False
            distance_proportional_sampling = True

        log.info("tasks created. now creating graph")
        monet = MONET(task_configs=task_configs,
                                task_env=env,
                                solution_dim=task_config.solution_dim,
                                seed=seed,
                                threshold=cfg.algorithm.threshold,
                                use_random_neighborhood=use_random_neighborhood,
                                max_neighbors_per_node=num_neighbors,
                                store_similarity_matrix=cfg.algorithm.store_similarity_matrix,
                                distance_proportional_sampling=distance_proportional_sampling,)
        run_monet(monet=monet, cfg=cfg, task_config=task_config, use_wandb=use_wandb)

    elif algorithm_name == "MT-ME":
        if use_wandb:
            wandb.init(
                project=task_config.project,
                config={
                    "n_tasks": task_config.num_tasks,
                    "solution_dim": task_config.solution_dim,
                    "task_dim": task_config.task_dim,
                    "max_evals": cfg.algorithm.max_evals,
                    "seed": seed,
                    "algorithm": cfg.algorithm.name,
                    "domain": task_config.domain,
                    "dir": cfg.wandb.dir,
                    "variation": cfg.algorithm.variation,
                },
                name=f"mt_me_seed_{seed}",
            )

        mt_me = MTME(
            task_configs=task_configs,
            solution_dim=task_config.solution_dim
        )

        dim_x = task_config.solution_dim
        params = default_params.copy()
        # Override default params with config values
        params['use_distance'] = cfg.algorithm.use_distance
        params['p_min'] = task_config.min_solution
        params['p_max'] = task_config.max_solution
        params['batch_size'] = cfg.algorithm.batch_size
        params['parallel'] = cfg.algorithm.parallel
        params['cvt_samples'] = cfg.algorithm.cvt_samples
        params['cvt_use_cache'] = cfg.algorithm.cvt_use_cache
        params['random_init'] = cfg.algorithm.random_init
        params['random_init_batch'] = cfg.algorithm.random_init_batch
        params['dump_period'] = cfg.algorithm.dump_period
        params['max_evals'] = cfg.algorithm.max_evals
        params['variation'] = cfg.algorithm.variation  # e.g., "gaussian_mutation", "polynomial_mutation", "sbx", "iso_dd", "iso_mtme"
        params['iso_sigma'] = cfg.algorithm.iso_sigma
        params['line_sigma'] = cfg.algorithm.line_sigma
        params['mutation_std'] = cfg.algorithm.mutation_std
        params['log_interval'] = cfg.log_interval
        params['log_frequency'] = cfg.log_frequency
        params['wandb'] = cfg.wandb.enabled and WANDB_AVAILABLE

        log.info(f"Running MT-ME with parameters: {params}")
        # log config to wandb
        if use_wandb:
            wandb.config.update(params)
        mt_me.compute(dim_x=dim_x, evaluator_factory=env, centroids=centroids, tasks=tasks, num_evals=cfg.algorithm.max_evals, params=params, log_file=open('cover_max_mean.dat', 'w'))
        log.info("MT-ME computation completed.")

    elif algorithm_name == "PT-ME":
        if use_wandb:
            wandb.init(
                project=task_config.project,
                config={
                    "n_tasks": task_config.num_tasks,
                    "solution_dim": task_config.solution_dim,
                    "task_dim": task_config.task_dim,
                    "max_evals": cfg.algorithm.max_evals,
                    "seed": seed,
                    "algorithm": cfg.algorithm.name,
                    "domain": task_config.domain,
                    "dir": cfg.wandb.dir,
                    "variation": cfg.algorithm.variation,
                    "proba_regression": cfg.algorithm.proba_regression,
                    "n_cells": cfg.algorithm.n_cells,
                    "linreg_sigma": cfg.algorithm.linreg_sigma,
                    "iso_sigma": cfg.algorithm.iso_sigma,
                    "line_sigma": cfg.algorithm.line_sigma,
                },
                name=f"pt_me_seed_{seed}_ncells_{cfg.algorithm.n_cells}",
            )

        pt_me = PTME(
            task_configs=task_configs,
            task_env=env,
            solution_dim=task_config.solution_dim,
            seed=seed,
            budget=cfg.algorithm.max_evals,
            n_cells=cfg.algorithm.n_cells,
            proba_regression=cfg.algorithm.proba_regression,
            variation_operator=cfg.algorithm.variation,
            lingreg_sigma=cfg.algorithm.linreg_sigma,
            wandb=use_wandb,
        )

        pt_me.run()

    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


    # Final logging and cleanup
    if use_wandb:
        wandb.finish()