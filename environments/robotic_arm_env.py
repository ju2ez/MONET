from .base_env import BaseEnv
from .base_task_config import BaseTaskConfig
import math
from math import cos, sin, pi, sqrt
import numpy as np
from typing import Tuple, Dict, Any

class TaskConfigArm(BaseTaskConfig):
    """Configuration for the robotic arm task"""

    def __init__(self, task_id: int, task_vec):
        super().__init__(task_id, task_vec)
        self.task_id = task_id
        self.task_vec = task_vec
        assert len(task_vec) == 2, "Task vector must have exactly two elements."

class Arm(BaseEnv):
    def __init__(self, task):
        super().__init__(task=task)
        self.task_vec = task.task_vec
        self.n_dofs = len(self.task_vec)
        self.joint_xy = []
        self.lengths = np.zeros(self.n_dofs + 1)

    def fw_kinematics(self, p):
        # assert len(p) == self.n_dofs
        p = np.append(p, 0)
        self.joint_xy = []
        mat = np.matrix(np.identity(4))
        for i in range(0, self.n_dofs + 1):
            m = [[cos(p[i]), -sin(p[i]), 0, self.lengths[i]],
                 [sin(p[i]),  cos(p[i]), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]
            mat = mat * np.matrix(m)
            v = mat * np.matrix([0, 0, 0, 1]).transpose()
            self.joint_xy += [v[0:2].A.flatten()]
        return self.joint_xy[self.n_dofs], self.joint_xy

    def evaluate_solution(self, solution):
        angular_range = self.task_vec[0] / len(solution)
        lengths = np.ones(len(solution)) * self.task_vec[1] / len(solution)
        self.n_dofs = len(lengths)
        self.lengths = np.concatenate(([0], lengths))
        target = 0.5 * np.ones(2)
        command = (solution - 0.5) * angular_range * math.pi * 2
        ef, _ = self.fw_kinematics(command)
        f = np.exp(-np.linalg.norm(ef - target))
        return f

if __name__ == "__main__":
    # Example usage
    task = TaskConfigArm(task_id=1, task_vec=[9, 5])
    arm = Arm(task=task)
    solution = np.random.random(10)
    fitness = arm.evaluate_solution(solution)
    print(f"Task: {task}, Fitness: {fitness}")