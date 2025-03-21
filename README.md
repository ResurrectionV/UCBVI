# Puzzle Environment with UCBVI

This repository implements an Upper Confidence Bound Value Iteration (UCBVI) algorithm for a puzzle-like environment using a custom Gym environment named `PuzzleEnv`. In this environment, the agent must find the correct sequence of actions (a "combination lock") to receive a reward. The UCBVI agent learns a Q-function over a finite horizon and the training performance is visualized by plotting the total reward per episode.

## Overview

The project includes:
- **PuzzleEnv:**  
  A custom Gym environment that simulates a combination lock puzzle. The environment is parameterized by the horizon length, number of actions, and states. A predetermined sequence of "good actions" is generated based on a random seed.
  
- **UCBVI Agent:**  
  An implementation of the UCBVI algorithm that maintains counts for state-action pairs and computes optimistic Q-values using an exploration bonus.
  
- **Training Loop:**  
  A loop that trains the UCBVI agent over a specified number of episodes and records the cumulative reward per episode.

- **Visualization:**  
  After training, a plot is generated that displays the total reward obtained in each episode.

## Requirements

- Python 3.9
- [Gym](https://github.com/openai/gym)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [tqdm](https://github.com/tqdm/tqdm)

Install the required packages via pip:

```bash
pip install gym numpy matplotlib tqdm
```

## Usage
```bash
python comb_lock_ucbvi.py
```
