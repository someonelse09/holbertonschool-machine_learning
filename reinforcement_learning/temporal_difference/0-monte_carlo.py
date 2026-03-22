#!/usr/bin/env python3
"""This module includes the function
that performs the Monte Carlo algorithm"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Args:
        env is environment instance
        V is a numpy.ndarray of shape (s,)
         containing the value estimate
        policy is a function that takes in a state
         and returns the next action to take
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
    Returns:
        V, the updated value estimate
    """
    #  For each episode, the environment is reset and
    #  the policy is followed step-by-step, collecting
    #  (state, reward) tuples until the episode ends
    for e in range(episodes):
        state, _ = env.reset()
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = \
                env.step(action)
            episode.append((state, reward))
            state = new_state
            if terminated or truncated:
                break
        # After an episode, we traverse it in reverse,
        # accumulating the discounted return G = γ·G + r at each step.
        G = 0
        episode = np.array(episode, dtype=int)

        for t in range(len(episode) - 1, -1, -1):
            state_t, reward_t = episode[t]
            G = gamma * G + reward_t
            if state_t not in episode[:e, 0]:
                V[state_t] = V[state_t] + alpha * (G - V[state_t])

    return V


"""
Episode Generation — For each episode, the environment is reset and
 the policy is followed step-by-step, collecting (state, reward) tuples
  until the episode ends (terminal state or max steps reached).

Backward Return Calculation — After an episode, we traverse it in reverse,
 accumulating the discounted return G = γ·G + r at each step.
  This is more efficient than recalculating from scratch for every timestep.

We track a visited set and only update V[s] the first time a state
appears in the episode (scanning backward = last occurrence forward).
The incremental update rule used is:
V[s] ← V[s] + α · (G - V[s])
"""
