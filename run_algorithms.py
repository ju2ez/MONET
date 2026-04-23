import os
import time
import hydra
from omegaconf import DictConfig
import logging
import random
import numpy as np
import ntplib
from datetime import datetime

import signal
import atexit
import sys
import psutil




# --- Hydra compatibility fix for Python 3.14 ---
import argparse
_original_add_argument = argparse.ArgumentParser.add_argument

def _patched_add_argument(self, *args, **kwargs):
    help_arg = kwargs.get('help')
    if help_arg and type(help_arg).__name__ == 'LazyCompletionHelp':
        # Patch the class of this instance to have __contains__
        cls = type(help_arg)
        if not hasattr(cls, '__contains__'):
            cls.__contains__ = lambda self, item: False
    return _original_add_argument(self, *args, **kwargs)

argparse.ArgumentParser.add_argument = _patched_add_argument
# -----------------------------------------------

def kill_all_children(sig=signal.SIGTERM):
    """Recursively terminate all child processes of this process."""
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.send_signal(sig)
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(children, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

def handle_signal(signum, frame):
    print(f"Received signal {signum}, cleaning up...")
    kill_all_children()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)
atexit.register(kill_all_children)


# Try to import wandb, but don't fail if it's not available
try:
    import wandb
    WANDB_AVAILABLE = True # True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# from algorithms.runners import run_emon_algorithm, run_mtme_algorithm, run_ptme_algorithm
from algorithms.runners import runner

log = logging.getLogger(__name__)

os.environ.setdefault("PYTHONHASHSEED", "0")

def init_global_seeds(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

def get_correct_time():
    try:
        client = ntplib.NTPClient()
        response = client.request('pool.ntp.org')  # Use a reliable NTP server
        correct_time = datetime.fromtimestamp(response.tx_time)
        return correct_time
    except Exception as e:
        log.warning(f"Failed to fetch time from NTP server: {e}")
        return datetime.now()  # Fallback to system time


def shutdown(use_wandb: bool, exit_code: int = 0):
    """Stop external services, terminate descendants, and exit cleanly."""
    try:
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
            time.sleep(60)  # Give wandb some time to finalize
    except Exception:
        pass

    try:
        kill_all_children(signal.SIGTERM)
        psutil.wait_procs(psutil.Process(os.getpid()).children(recursive=True), timeout=5)
        kill_all_children(signal.SIGKILL)
    except Exception:
        pass

    # flush logs
    sys.stdout.flush()
    sys.stderr.flush()

    # Let Hydra + Submitit finalize correctly
    sys.exit(exit_code)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Configure logging to show file names and line numbers
    logging.basicConfig(
        level=logging.CRITICAL,
        format='%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
        force=True
    )

    # Fetch the correct time
    correct_time = get_correct_time()
    log.info(f"Correct time fetched: {correct_time}")

    # Set timezone
    os.environ["TZ"] = cfg.timezone
    time.tzset()

    # Import task-specific modules based on config
    if cfg.task == "archery":
        from environments.archery_env import Archery as env, TaskConfigArchery as TaskConfig
        task_config = cfg.archery
    elif cfg.task == "arm":
        from environments.robotic_arm_env import Arm as env, TaskConfigArm as TaskConfig
        task_config = cfg.arm
    elif cfg.task == "hexapod":
        from environments.hexapod_env import Hexapod as env, TaskConfigHexapod as TaskConfig
        task_config = cfg.hexapod
    elif cfg.task == "cartpole":
        from environments.cartpole_env import Cartpole as env, TaskConfigCartpole as TaskConfig
        task_config = cfg.cartpole

    seed = cfg.seed
    if not seed:
        # choose a random seed
        seed = random.randint(0, 10_000)

    # Overwrite seed with SLURM_ARRAY_TASK_ID if available
    seed = int(os.environ.get("SLURM_ARRAY_TASK_ID", seed))
    log.info(f"Using seed: {seed}")
    init_global_seeds(seed)

    log.info(f"Using algorithm: {cfg.algorithm.name}")

    # Make this process leader of a new process group
    try:
        os.setpgrp()
    except Exception:
        pass

    # Determine if wandb should be used
    use_wandb = cfg.wandb.enabled and WANDB_AVAILABLE

    runner.run_algorithm(cfg, task_config, env, TaskConfig, seed, use_wandb)

    # Robust shutdown: stop wandb, kill children, exit
    # shutdown(use_wandb, exit_code=0)

if __name__ == "__main__":
    main()
    log.info("Main function executed successfully.")
