# RL_Project


## Project Overview

This project applies **Reinforcement Learning (RL)** to train autonomous agents for simulated driving environments using the `highway-env` package. It explores both discrete and continuous action spaces in different traffic scenarios: highway driving and intersection navigation.

We use customized configurations for each scenario and employ different observation types (e.g., `OccupancyGrid`, `Kinematics`) to tailor the training environment to the needs of each task.

---

## Team Members

- Meziany Imane
- Yakhou Yousra
- Zuo Yuxian

---


## Environments and Scenarios

### 1. **Highway with Discrete Actions**

- **Config file**: `config.py`
- **Observation type**: `OccupancyGrid`
- **Action type**: `DiscreteMetaAction`
- **Vehicles**: 15 (10 for observations)
- **Duration**: 60s
- **Reward shaping**:
  - Collision: -1
  - High speed: +0.1
  - Right lane: +0.5

### 2. **Highway with Continuous Actions**

- **Config file**: `highway_continuous_config.py`
- **Observation type**: `Kinematics`
- **Action type**: `ContinuousAction`
- **Vehicles**: 15
- **Controlled vehicles**: 1
- **Duration**: 60s

### 3. **Intersection Navigation**

- **Config file**: `intersection_config.py`
- **Observation type**: `Kinematics`
- **Action type**: `DiscreteMetaAction`
- **Vehicles**: 6
- **Controlled vehicles**: 1
- **Duration**: 40s

---

## Files Description

| File                              | Description                                       |
| --------------------------------- | ------------------------------------------------- |
| `config.py`                     | Configuration for highway with discrete actions   |
| `highway_continuous_config.py`  | Configuration for highway with continuous control |
| `intersection_config.py`        | Configuration for intersection navigation         |
| `config.pkl`                    | Serialized config for discrete highway scenario   |
| `highway_continuous_config.pkl` | Serialized config for continuous highway control  |
| `intersection_config.pkl`       | Serialized config for intersection scenario       |
| `RL_project.ipynb`              | Notebook implementing training and evaluation     |
| `README.md`                     | Project documentation                             |

---
