#!/usr/bin/env python3
import gym
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import math
from itertools import count
from tqdm import tqdm

class PuzzleEnv(gym.Env):
    """
    PuzzleEnv environment.
    
    Parameters:
        H (int): Horizon length.
        A (int): Number of actions.
        S (int): Number of states.
        R (int): Reward at the terminal state if the good action is taken.
        seed (int): Random seed for generating the sequence of good actions.
    """
    def __init__(self, H=10, A=10, S=3, R=10, seed=11):
        random.seed(a=seed)
        self.good_action_list = [random.randint(1, A) for _ in range(H + 1)]
        random.seed(None)
        self.good_action = self.good_action_list[0]
        self.state = random.choice([1, 2])
        self.reward = R
        self.h = 0
        self.A = A
        self.seed = seed
        self.H = H

    def reset(self):
        self.good_action = self.good_action_list[0]
        self.state = random.choice([1, 2])
        self.h = 0
        return self.state

    def step(self, action):
        done = False
        reward = 0

        if self.h < self.H - 1:
            if action == self.good_action and self.state in [1, 2]:
                self.state = random.choice([1, 2])
            else:
                self.state = 3
        else:
            done = True
            if action == self.good_action and self.state in [1, 2]:
                reward = self.reward

        self.h += 1  # Advance one step
        self.good_action = self.good_action_list[self.h]
        return self.state, reward, done, {}

class UCBVI(object):
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.H = 10  # Horizon length
        self.S = 3   # Number of states
        self.A = 10  # Number of actions
        # Count of state-action visits for each timestep h
        self.N_sa = np.zeros((self.H, self.S, self.A))
        # Count of state-action-next_state visits for each timestep h
        self.N_sas = np.zeros((self.H, self.S, self.A, self.S))
        # Initialize Q-values to the horizon
        self.Q = np.full((self.H, self.S, self.A), self.H).astype(float)
        self.h_current = 0

    def update(self, state, action, next_state, reward, done):
        # Convert state, action, next_state to 0-indexed values.
        state -= 1
        action -= 1
        next_state -= 1
        self.N_sa[self.h_current, state, action] += 1
        self.N_sas[self.h_current, state, action, next_state] += 1

        if not done:
            next_V = np.max(self.Q[self.h_current+1, :, :], axis=1)
            P_hat = self.N_sas[self.h_current, state, action, :] / self.N_sa[self.h_current, state, action]
            bonus_reward = self.alpha * np.sqrt(1 / self.N_sa[self.h_current, state, action])
            if self.h_current + 1 == self.H:
                self.Q[self.h_current, state, action] = min(self.H, reward + bonus_reward)
            else:
                self.Q[self.h_current, state, action] = min(self.H, reward + bonus_reward + np.sum(P_hat * next_V))
        else:
            bonus_reward = self.alpha * np.sqrt(1 / self.N_sa[self.h_current, state, action])
            self.Q[self.h_current, state, action] = min(self.H, reward + bonus_reward)

        self.h_current = 0 if done else self.h_current + 1

    def action(self, state):
        state -= 1
        # Adjust Q-values with bonus for exploration.
        adjusted_q_values = self.Q[self.h_current, state, :] + self.alpha * np.sqrt(1 / self.N_sa[self.h_current, state, :])
        action = np.argmax(adjusted_q_values) + 1
        return action

def main():
    num_episodes = 25000  # Adjust this number for convergence
    env = PuzzleEnv()
    # Tune alpha as needed
    agent = UCBVI(alpha=0.001)
    
    total_rewards = np.zeros(num_episodes)

    for i_episode in tqdm(range(num_episodes)):
        state = env.reset()
        for h in range(env.H):
            action = agent.action(state)
            next_state, reward, done, _ = env.step(action)
            total_rewards[i_episode] += reward
            agent.update(state, action, next_state, reward, done)
            state = next_state
            if done:
                break

    print('Training Complete')
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Performance')
    plt.show()

if __name__ == '__main__':
    main()
