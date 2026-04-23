import numpy as np
from sklearn.cluster import KMeans
from collections import Counter


def calculate_fitness_metrics(graph_map_elites):
    """Calculate mean, median, and QD-score for all nodes with valid fitness values."""
    fits = []
    solution_ids = []
    total_nodes = len(list(graph_map_elites.nodes()))

    for node_id in graph_map_elites.nodes():
        node = graph_map_elites.get_node(node_id)
        fitness = float(node["fitness"])
        if fitness is not None and not np.isnan(fitness) and fitness != -float('inf'):
            fits.append(fitness)
            solution_ids.append(node["solution_id"])

    # count frequency of solution_ids
    solution_id_counts = Counter(solution_ids)
    # avg cluster size: total solutions / unique solutions
    avg_cluster_size = len(solution_ids) / len(solution_id_counts) if len(solution_id_counts) > 0 else 0
    # coverage: percentage of nodes with solutions
    coverage = len(fits) / total_nodes if total_nodes > 0 else 0

    if len(fits) > 0:
        return {
            "mean_fitness": np.mean(fits),
            "median_fitness": np.median(fits),
            "qd_score": np.sum(fits),
            "solution_id_count": len(solution_id_counts),
            "avg_cluster_size": avg_cluster_size,
            "coverage": coverage
        }
    else:
        return {
            "mean_fitness": 0,
            "median_fitness": 0,
            "qd_score": 0,
            "solution_id_count": 0,
            "avg_cluster_size": 0,
            "coverage": 0
        }


def cvt(k, dim, samples, cvt_use_cache=True):
    """Compute Centroidal Voronoi Tessellation."""
    print("Computing CVT (this can take a while...):")

    x = np.random.rand(samples, dim)
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init=1, verbose=1)
    k_means.fit(x)
    return k_means.cluster_centers_


def calculate_max_distance(task_configs):
    """Calculate maximum fitness for task configurations."""
    def max_fitness(task_vec, target=np.array([0.5, 0.5]), n_dofs=10):
        angular_range, total_length = task_vec
        link_length = total_length / n_dofs

        # Arm can stretch between 0 and total_length
        target_dist = np.linalg.norm(target)

        # Closest possible distance
        if target_dist > total_length:
            d_min = target_dist - total_length
        else:
            d_min = 0  # within reach

        return np.exp(-d_min)

    for task in task_configs:
        task.max_fitness = max_fitness(task.task_vec)

    # log the max fitness to wandb
    # TODO: log to wandb and extend
