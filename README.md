# RL_Project _ Autonomous Driving


## Project Overview

This project aims to implement, train, and evaluate reinforcement learning agents in simulated road environments using the [`highway-env`](http://highway-env.farama.org/) framework.

The goal is to explore how RL agents can be trained to navigate safely and efficiently through various road scenarios such as highways and intersections, using both **discrete** and **continuous** action spaces. We also integrate a third scenario trained with **Stable-Baselines3**.

## Team Members

- Meziany Imane
- Yakhou Yousra
- Zuo Yuxian

---


## Environments and Scenarios

### 1. Task 1: Pre-specified Environment : **Highway with Discrete Actions**

- **Config file**: `config.py`
- **Observation type**: `OccupancyGrid`
- **Action type**: `DiscreteMetaAction`
- **Vehicles**: 15
- **Duration**: 60s
- **Reward shaping**:

  - Collision: -1
  - High speed: +0.1
  - Right lane: +0.5
- **Objective**: Train a car to drive in highway conditions with discrete actions and document learning behavior.

### 2. **Highway with Continuous Actions**

- **Config file**: `highway_continuous_config.py`
- **Observation type**: `Kinematics`
- **Action type**: `ContinuousAction`
- **Objective**: Analyze how agent behavior and learning differ when using continuous actions.

### 3. **Intersection Navigation**

- **Config file**: `intersection_config.py`
- **Observation type**: `Kinematics`
- **Action type**: `DiscreteMetaAction`
- **Vehicles**: 10
- **Duration**: 60s
- **Objective**: Leverage off-the-shelf RL agents for training in a more complex urban setting.

---

## Files Description

| File                              | Description                                         |
| --------------------------------- | --------------------------------------------------- |
| `config.py`                     | Task 1 environment: highway with discrete actions   |
| `highway_continuous_config.py`  | Task 2 environment: highway with continuous actions |
| `intersection_config.py`        | Task 3 environment: intersection with SB3 agent     |
| `config.pkl`                    | Serialized config for discrete highway scenario     |
| `highway_continuous_config.pkl` | Serialized config for continuous highway control    |
| `intersection_config.pkl`       | Serialized config for intersection scenario         |
| `RL_project.ipynb`              | Main training notebook implementing all tasks       |
| `README.md`                     | Project documentation                               |

---
