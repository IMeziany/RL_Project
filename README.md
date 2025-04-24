# Reinforcement Learning Project: Highway-Env Tasks

This repository contains implementations for two tasks using reinforcement learning algorithms (DQN and PPO) on the `highway-env` simulation environment.

---

## Task1 - DQN

- `Task1_highway_dqn.ipynb`: Main notebook for Task 1, where a DQN agent is trained.
- `gridsearch_dqn_results.pkl`: Contains grid search results for hyperparameter tuning.

---

## Task2_ppo - PPO

- `Task2_continuous_ppo.ipynb`: Main notebook for training and evaluating the PPO agent.
- `PPOContinuous.py`: Defines the PPOContinuous class and training logic for continuous control.
- `net.py`: Contains the actor and critic neural network architectures used by the PPO agent.
- `runs/`: TensorBoard logs for monitoring training progress.
- `ppo_checkpoints/`: Saves the best agent from each experiment (based on highest evaluation reward).
- `plot_runs/`: Contains data exported from TensorBoard for plotting evaluation rewards during training.

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