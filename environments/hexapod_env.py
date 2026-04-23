from .base_env import BaseEnv
from .base_task_config import BaseTaskConfig
import math
from math import cos, sin, pi, sqrt
import numpy as np
import pybullet
from typing import Tuple, Dict, Any
import os
import sys
from contextlib import redirect_stdout, redirect_stderr

os.environ["PYBULLET_SUPPRESS_OUTPUT"] = "1"


import environments.pyhexapod.simulator as simulator
import environments.pyhexapod.hexapod_controller as ctrl

# Single shared simulator per process — avoids PyBullet's max-connections limit
# and eliminates URDF reloading/settle steps when the same morphology is re-evaluated.
_shared_simu = None
_current_urdf = None

def load_tasks(directory, num_tasks):
    tasks = []
    centroids = []
    for i in range(0, num_tasks):
        centroid = np.loadtxt(directory + '/lengthes_' + str(i) + '.txt')
        urdf_file = directory + '/pexod_' + str(i) + '.urdf'
        centroids += [centroid]
        tasks += [(centroid, urdf_file)]
    return np.array(centroids), tasks


class TaskConfigHexapod(BaseTaskConfig):
    """Configuration for the robotic arm task"""
    def __init__(self, task_id: int, task_vec, urfd_file: str):
        super().__init__(task_id, task_vec)
        self.task_id = task_id
        self.task_vec = task_vec
        self.urdf_file = urfd_file
        # assert len(task_vec) == 36, "Task vector should be 36-dimensional."


class Hexapod(BaseEnv):
    def __init__(self, task):
        super().__init__(task=task)
        self.task_id = task.task_id
        self.task_vec = task.task_vec
        self.urdf_file = task.urdf_file

    def evaluate_solution(self, x):
        global _shared_simu, _current_urdf
        # if x has values outside [0,1], return 0 fitness
        for val in x:
            if val < 0.0 or val > 1.0:
                return 0
        if _shared_simu is None:
            _shared_simu = simulator.HexapodSimulator(gui=False, urdf=self.urdf_file, video='')
            _current_urdf = self.urdf_file
        elif self.urdf_file == _current_urdf:
            _shared_simu.fast_reset()  # no URDF reload, no settle steps
        else:
            _shared_simu.urdf = self.urdf_file
            _shared_simu.reset()  # full reload + settle for new morphology
            _current_urdf = self.urdf_file
        controller = ctrl.HexapodController(x)
        dead = False
        fit = -1e10
        steps = 3. / _shared_simu.dt
        i = 0
        while i < steps and not dead:
            _shared_simu.step(controller)
            p = _shared_simu.get_pos()[0]
            a = pybullet.getEulerFromQuaternion(_shared_simu.get_pos()[1])
            out_of_corridor = abs(p[1]) > 0.5
            out_of_angles = abs(a[0]) > math.pi/8 or abs(a[1]) > math.pi/8 or abs(a[2]) > math.pi/8
            if out_of_angles or out_of_corridor:
                dead = True
            i += 1
        fit = p[0]
        return fit