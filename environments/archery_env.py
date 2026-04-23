from .base_env import BaseEnv
from .base_task_config import BaseTaskConfig
import numpy as np

class TaskConfigArchery(BaseTaskConfig):
    """Configuration for the archery task"""
    def __init__(self, task_id: int, task_vec):
        super().__init__(task_id, task_vec)
        self.task_id = task_id
        self.task_vec = task_vec

class Archery(BaseEnv):
    """Archery environment"""
    def __init__(self, task):
        super().__init__(task=task)
        self.task = task

        self.archery_state_bounds = {"d": {"low": 5, "high": 40}, "w": {"low": -10, "high": 10}}
        self.archery_action_bounds = {"low": np.array([-np.pi/12, -np.pi/12]), "high": np.array([np.pi/12, np.pi/12])}

        self.archery_situation = self.compute_archery_situation(n=1, archery_state_bounds=self.archery_state_bounds)
        self.archery_command = self.compute_archery_command(n=1, archery_action_config=self.archery_action_bounds, iso_sigma=0.01, line_sigma=0.2)

    def unwrap(self, x, bounds):
        # [0, 1] -> [low, high]
        return x * (bounds["high"]-bounds["low"]) + bounds["low"]

    def wrap(self, x, bounds):
        # [low, high] -> [0, 1]
        return (x - bounds["low"]) / (bounds["high"]-bounds["low"])

    def compute_archery_situation(self, n, archery_state_bounds):
        res = {}
        bounds = {"low": [], "high": []}
        for i in range(n):
            for key, bound in archery_state_bounds.items():
                bounds["low"].append(bound["low"])
                bounds["high"].append(bound["high"])
        res["bounds"] = {"low": np.array(bounds["low"]), "high": np.array(bounds["high"])}
        res["dim"] = len(bounds["high"])
        return res

    def compute_archery_command(self, n, archery_action_config, iso_sigma, line_sigma):
        action_bounds = {"low": np.concatenate([archery_action_config["low"] for _ in range(n)]), "high":  np.concatenate([archery_action_config["high"] for _ in range(n)])}
        return {"bounds": action_bounds, "iso_sigma": iso_sigma, "line_sigma": line_sigma, "dim": len(action_bounds["low"])}

    def eval_archery(self, c, s, verbose=0):
        n = len(s)//2
        assert(n==1)
        rewards = {}
        for i in range(n):
            [yaw, pitch] = c[2*i:2*i+2]
            if type(s) == dict:
                d = s[f"d_{i}"] # distance
                w = s[f"w_{i}"] # wind
            else:
                d = s[2*i] # distance
                w = s[2*i+1] # wind
            v0 = 70
            # average velocity of an arrow 70 m.s-1
            # with yaw = 0, pitch = 0.5 * arcsin(g*d/v0**2)
            v = v0 * np.array([-np.sin(yaw), np.cos(yaw)*np.cos(pitch), np.cos(yaw) * np.sin(pitch)])
            if v[1] <= 0:
                rewards[i] = 0
            else:
                t = d/v[1]
                contact = np.array([0.5*w*t**2+v[0]*t, -0.5*9.8*t**2+v[2]*t])
                distance = np.linalg.norm(contact)
                # 6.1cm rayon par tranche, 10 tranches, 122cm au total
                rewards[i] = max(0, int(10-distance//0.061))/10
        reward = rewards[0]
        return {"reward": reward, "rewards": rewards}

    def evaluate_solution(self, solution):
        # Task vec is in [0,1], unwrap it locally without modifying
        task_vec = self.task.task_vec
        situation_bounds = {"low": np.array([self.archery_state_bounds["d"]["low"], self.archery_state_bounds["w"]["low"]]),
                           "high": np.array([self.archery_state_bounds["d"]["high"], self.archery_state_bounds["w"]["high"]])}

        s_unwrapped = self.unwrap(task_vec, situation_bounds)
        c_unwrapped = self.unwrap(solution, self.archery_action_bounds)

        evaluation = self.eval_archery(c=c_unwrapped, s=s_unwrapped)
        return evaluation["reward"]
