# MONET Hexapod Experiments

This document provides a reference for the hexapod controller's **36-dimensional solution vector** and **12-dimensional task vector**, with links to the source code for verification.

---

## Table of Contents

1. [Overview](#overview)
2. [Solution Vector (36D)](#solution-vector-36d)
3. [Signal Generation](#signal-generation)
4. [Joint Mapping](#joint-mapping)
5. [Task Vector (12D)](#task-vector-12d)
6. [Source Files](#source-files)

---

## Overview

The hexapod is controlled by an **open-loop periodic controller** following [Cully et al., Nature 521 (2015)](https://www.nature.com/articles/nature14422).

- **Solution vector**: 36 parameters in `[0, 1]` that define the gait
- **Task vector**: 12 values (leg segment lengths in meters) that define the morphology
- **Controller**: Generates 18 joint trajectories (3 joints × 6 legs)


---

## Solution Vector (36D)

The 36 parameters are organized as **6 groups of 6**, one per leg:

| Leg | Param Indices | Signal A (Hip) | Signal B (Knee) |
|-----|---------------|----------------|-----------------|
| 0   | 0–5           | `[0, 1, 2]`    | `[3, 4, 5]`     |
| 1   | 6–11          | `[6, 7, 8]`    | `[9, 10, 11]`   |
| 2   | 12–17         | `[12, 13, 14]` | `[15, 16, 17]`  |
| 3   | 18–23         | `[18, 19, 20]` | `[21, 22, 23]`  |
| 4   | 24–29         | `[24, 25, 26]` | `[27, 28, 29]`  |
| 5   | 30–35         | `[30, 31, 32]` | `[33, 34, 35]`  |

Each triplet contains:

| Parameter   | Range   | Effect |
|-------------|---------|--------|
| `amplitude` | [0, 1]  | Magnitude of oscillation |
| `phase`     | [0, 1]  | Timing offset within gait cycle |
| `duty_cycle`| [0, 1]  | Fraction of cycle in "up" position |

---

## Signal Generation

Each `(amplitude, phase, duty_cycle)` triplet generates a smooth periodic signal:

### Algorithm

```
1. Top-hat function:
   f(t) = +amplitude  if t < duty_cycle × T
   f(t) = −amplitude  otherwise

2. Gaussian smoothing:
   kernel_size = T / 10 = 10 steps
   σ = kernel_size / 3 ≈ 3.33
   Circular convolution with Gaussian kernel

3. Phase shift:
   output(t) = smoothed((t + phase × T) mod T)
```

Where `T = 100` timesteps (one gait cycle).

**Source**: [`open_loop_controller.py:56-106`](../environments/pyhexapod/pycontrollers/open_loop_controller.py#L56-L106)

---

## Joint Mapping

The controller maps 36 parameters to 18 joints:

```python
def _compute_trajs(self, p, array_dim):
    trajs = np.zeros((6 * 3, array_dim))  # 18 joints × 100 timesteps
    k = 0
    for i in range(0, 36, 6):             # i = 0, 6, 12, 18, 24, 30
        trajs[k,:]   = 0.5 * _control_signal(p[i], p[i+1], p[i+2])   # Hip
        trajs[k+1,:] = _control_signal(p[i+3], p[i+4], p[i+5])       # Femur
        trajs[k+2,:] = trajs[k+1,:]       # Tibia (copy of femur)
        k += 3
    return trajs * math.pi / 4.0          # Scale to ±π/4 radians
```

### Joint Details

| Joint | Name | Signal | Scaling | Range | Axis |
|-------|------|--------|---------|-------|------|
| Hip   | `body_leg_X` | A | × 0.5 × π/4 | ±π/8 (≈22.5°) | Z |
| Femur | `leg_X_1_2`  | B | × π/4 | ±π/4 (≈45°) | X |
| Tibia | `leg_X_2_3`  | B (copy) | × π/4 | ±π/4 (≈45°) | X |

> **Note**: Hip receives half amplitude (±π/8) despite having wider URDF limits (±π/2).  
> Femur and tibia always receive **identical commands**.

**Source**: [`hexapod_controller.py:51-59`](../environments/pyhexapod/pycontrollers/hexapod_controller.py#L51-L59)


```python
class HexapodController(OpenLoopController):
    ''' 
        This should be the same controller as Cully et al., Nature, 2015
        example values: ctrl = [1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0.5, 0.5, 0.25, 0.75, 0.5, 1, 0, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5, 1, 0.5, 0.5, 0.25, 0.25, 0.5, 1, 0, 0.5, 0.25, 0.75, 0.5]
    '''
    def __init__(self, params, array_dim=100):
        super(HexapodController, self).__init__(params, array_dim)
        self.trajs = self._compute_trajs(params, array_dim)

    def _compute_trajs(self, p, array_dim):
        trajs = np.zeros((6 * 3, array_dim))
        k = 0
        for i in range(0, 36, 6):
            trajs[k,:] =  0.5 * self._control_signal(p[i], p[i + 1], p[i + 2], array_dim)
            trajs[k+1,:] = self._control_signal(p[i + 3], p[i + 4], p[i + 5], array_dim)
            trajs[k+2,:] = trajs[k+1,:]
            k += 3
        return trajs * math.pi / 4.0
```

---

## Task Vector (12D)

Each hexapod morphology is defined by **12 leg segment lengths** (in meters):

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0  | L0 proximal | Leg 0 femur length |
| 1  | L0 distal   | Leg 0 tibia length |
| 2  | L1 proximal | Leg 1 femur length |
| 3  | L1 distal   | Leg 1 tibia length |
| 4  | L2 proximal | Leg 2 femur length |
| 5  | L2 distal   | Leg 2 tibia length |
| 6  | L3 proximal | Leg 3 femur length |
| 7  | L3 distal   | Leg 3 tibia length |
| 8  | L4 proximal | Leg 4 femur length |
| 9  | L4 distal   | Leg 4 tibia length |
| 10 | L5 proximal | Leg 5 femur length |
| 11 | L5 distal   | Leg 5 tibia length |

### Loading Tasks

```python
def load_tasks(directory, num_tasks):
    for i in range(num_tasks):
        centroid = np.loadtxt(directory + '/lengthes_' + str(i) + '.txt')
        urdf_file = directory + '/pexod_' + str(i) + '.urdf'
        tasks.append((centroid, urdf_file))
```

**Source**: [`hexapod_env.py:18-26`](../environments/hexapod_env.py#L18-L26)

---

## Source Files

| File | Contains |
|------|----------|
| [`open_loop_controller.py`](../environments/pyhexapod/pycontrollers/open_loop_controller.py) | `_control_signal()` — signal generation |
| [`hexapod_controller.py`](../environments/pyhexapod/pycontrollers/hexapod_controller.py) | `_compute_trajs()` — joint mapping |
| [`pexod.urdf`](../environments/pyhexapod/urdf/pexod.urdf) | Joint definitions, limits, axes |
| [`hexapod_env.py`](../environments/hexapod_env.py) | `load_tasks()`, `evaluate_solution()` |
| [`urdf/lengthes_X.txt`](../urdf/) | Leg segment lengths (12 values each) |

---

## References

- Cully, A., Clune, J., Tarapore, D., & Mouret, J. B. (2015). Robots that can adapt like animals. *Nature*, 521(7553), 503-507.
