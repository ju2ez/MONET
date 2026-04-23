import argparse
import numpy as np
import os
from spotoptim import SpotOptim

#  define a project name (used as the prefix for SpotOptim result files)
PROJECT = "spot_monet"


def make_monet_wrapper(task: str, num_tasks: int | None, max_evals: int | None):
    """Build a monet_wrapper closure over a specific task / budget.

    This lets the SpotOptim loop be reused both for full experiments
    (``task=hexapod``, default budget) and for smoke tests with tiny
    ``num_tasks`` / ``max_evals``.
    """

    def monet_wrapper(X):
        """
        Wrapper function for SpotOptim to optimize MONET hyperparameters.

        Args:
            X (np.ndarray): 2D array of shape (batch_size, n_vars) containing input parameters.
                            In this case, batch_size is usually 1 for sequential evaluation,
                            but SpotOptim might pass batch_size > 1 if configured.

                            The columns correspond to:
                            0: strategy (factor: 0="best_fitness", 1="random")
                            1: p_ind (float)
                            2: neighbor_percentage (float)
                            3: neighborhood (factor: 0="random", 1="closest", 2="distance_proportional")
                            4: individual_learning (factor: 0="gaussian_mutation", fixed)
                            5: social_learning (factor: 0="sbx", fixed)

        Returns:
            np.ndarray: 1D array of objective values (minimization).
        """
        import numpy as np
        import subprocess
        import re
        import os

        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        n_jobs_outer = int(os.environ.get("SPOT_N_JOBS", "1"))
        monet_workers = max(1, cpus // max(1, n_jobs_outer))

        # Mapping for categorical variables
        strategy_map = {0: "best_fitness", 1: "random"}
        neighborhood_map = {0: "random", 1: "closest", 2: "distance_proportional"}
        individual_learning_map = {0: "gaussian_mutation"}
        social_learning_map = {0: "sbx"}

        results = []

        # Iterate over each row (parameter set) in the batch X
        for row in X:
            strategy_idx = int(row[0])
            p_ind = row[1]
            neighbor_percentage = row[2]
            neighborhood_idx = int(row[3])
            individual_learning_idx = int(row[4])
            social_learning_idx = int(row[5])

            strategy = strategy_map.get(strategy_idx, "best_fitness")
            neighborhood = neighborhood_map.get(neighborhood_idx, "closest")
            individual_learning = individual_learning_map.get(individual_learning_idx, "gaussian_mutation")
            social_learning = social_learning_map.get(social_learning_idx, "sbx")

            cmd = [
                "uv", "run", "python", "run_algorithms.py",
                f"task={task}",
                "algorithm.name=MONET",
                "wandb.enabled=false",
                f"algorithm.MONET.strategy={strategy}",
                f"algorithm.MONET.p_ind={p_ind}",
                f"algorithm.MONET.neighbor_percentage={neighbor_percentage}",
                f"algorithm.MONET.neighborhood={neighborhood}",
                f"algorithm.MONET.individual_learning=[{individual_learning}]",
                f"algorithm.MONET.social_learning=[{social_learning}]",
                f"algorithm.MONET.n_workers={monet_workers}",
            ]
            if num_tasks is not None:
                cmd.append(f"{task}.num_tasks={num_tasks}")
            if max_evals is not None:
                cmd.append(f"algorithm.max_evals={max_evals}")
            print(f"Running command: {' '.join(cmd)}")

            try:
                process = subprocess.run(cmd, capture_output=True, text=True, check=True)
                output = process.stdout

                match = re.search(r"final_mean_fitness:\s*([-\d\.]+)", output)
                if match:
                    final_mean_fitness = float(match.group(1))
                    # Objective: maximize fitness => minimize (1.0 - fitness)
                    res = 1.0 - final_mean_fitness
                else:
                    print("Warning: Could not parse final_mean_fitness")
                    res = 1.0

            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
                print(f"Stderr: {e.stderr}")
                res = 1.0  # Penalty for failure

            results.append(res)

        return np.array(results)

    return monet_wrapper


def main():
    parser = argparse.ArgumentParser(description="SpotOptim MONET hyperparameter tuning")
    parser.add_argument("--task", type=str, default="hexapod",
                        choices=["arm", "archery", "cartpole", "hexapod"],
                        help="Task to tune on (default: hexapod)")
    parser.add_argument("--num-tasks", type=int, default=None,
                        help="Override <task>.num_tasks (default: config value)")
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Override algorithm.max_evals (default: config value)")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel SpotOptim workers (default: 1)")
    parser.add_argument("--n-initial", type=int, default=16,
                        help="Number of initial SpotOptim samples (default: 16)")
    parser.add_argument("--max-iter", type=float, default=float("inf"),
                        help="Maximum SpotOptim iterations (default: inf)")
    parser.add_argument("--max-time", type=float, default=2160,
                        help="Maximum wall-clock time in minutes (default: 2160 = 36h)")
    parser.add_argument("--seed", type=int, default=1,
                        help="SpotOptim seed (default: 1)")
    parser.add_argument("--result-dir", type=str, default=".",
                        help="Directory for result files (default: current dir)")
    args = parser.parse_args()

    if args.result_dir != ".":
        os.makedirs(args.result_dir, exist_ok=True)

    # Define the problem bounds and variable types
    bounds = [(0, 1), (0.0, 1.0), (0.0, 1.0), (0, 2), (0, 0), (0, 0)]
    var_type = ["factor", "float", "float", "factor", "factor", "factor"]
    var_name = ["strategy", "p_ind", "neighbor_percentage", "neighborhood",
                "individual_learning", "social_learning"]

    result_file = os.path.join(args.result_dir, f"{PROJECT}_res.pkl")
    monet_wrapper = make_monet_wrapper(args.task, args.num_tasks, args.max_evals)

    if os.path.exists(result_file):
        print(f"{result_file} already exists. Loading results...")
        try:
            opt = SpotOptim.load_result(result_file)
            print("Loaded successfully.")
        except Exception as e:
            print(f"Failed to load result: {e}")
            return
    else:
        print(f"{result_file} does not exist. Starting new optimization "
              f"(task={args.task}, n_jobs={args.n_jobs}, n_initial={args.n_initial}).")
        opt = SpotOptim(
            fun=monet_wrapper,
            bounds=bounds,
            var_type=var_type,
            var_name=var_name,
            max_iter=args.max_iter,
            max_time=args.max_time,
            n_initial=args.n_initial,
            seed=args.seed,
            verbose=False,
            tensorboard_log=False,  # disabled for n_jobs>1 (steady-state bug)
            tensorboard_clean=False,
            acquisition_optimizer='de_tricands',
            repeats_initial=1,
            repeats_surrogate=1,
            n_jobs=args.n_jobs,
        )

        opt.optimize()

        try:
            save_prefix = os.path.join(args.result_dir, PROJECT) if args.result_dir != "." else PROJECT
            opt.save_result(prefix=save_prefix)
            print(f"Results saved to {result_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

    # Display results
    print(f"Best value found: {opt.best_y_:.6f}")
    if hasattr(opt, "best_x_"):
        print(f"Best point: {opt.best_x_}")
    if hasattr(opt, "counter"):
        print(f"Total evaluations: {opt.counter}")
    if hasattr(opt, "n_iter_"):
        print(f"Number of iterations: {opt.n_iter_}")

    print("Completed")


if __name__ == "__main__":
    main()
