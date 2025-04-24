# Reinforcement Learning Project: Highway-Env Tasks

This repository contains implementations for two tasks using reinforcement learning algorithms (DQN and PPO) on the `highway-env` simulation environment.

---

## Task1 - DQN

- `Task1_highway_dqn.ipynb`: Main notebook for Task 1, where a DQN agent is trained to navigate the highway environment.
- `gridsearch_dqn_results.pkl`: Contains grid search results for hyperparameter tuning. This file enables quick analysis.

---

## Task2_ppo - PPO

- `Task2_continuous_ppo.ipynb`: Notebook for Task 2, focusing on training a PPO agent in a continuous action space.
- `net.py`: Defines the neural network architecture used for policy and value functions.
- `PPOContinuous.py`: Core implementation of the PPO training and evaluation logic.
- `runs/`: TensorBoard logs for monitoring training progress.
- `ppo_checkpoints/`: Saved the best agents (highest evaluation reward).
- `plot_runs/`: Exported data from TensorBoard used for plotting evaluation rewards

---

## Other Files

- `env_test.ipynb`: Used for testing the environment setup.

---

## Notes

- Ensure TensorBoard is installed (`pip install tensorboard`) to visualize training logs inside the `runs/` folder.
- All notebooks are designed to run on GPU if available.

---

## Author

**Yuxian Zuo:** yuxian.zuo@student-cs.fr