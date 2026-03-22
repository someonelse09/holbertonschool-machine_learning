#!/usr/bin/env python3
""" This module includes the function
that performs SARSA(λ) """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Determine action using epsilon-greedy policy.
    """
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state, :])
    else:
        return np.random.randint(0, Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000,
                  max_steps=100, alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """
    Args:
        env is the environment instance
        Q is a numpy.ndarray of shape (s,a)
         containing the Q table
        lambtha is the eligibility trace factor
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
        epsilon is the initial threshold for epsilon greedy
        min_epsilon is the minimum value that
         epsilon should decay to
        epsilon_decay is the decay rate for
         updating epsilon between episodes
    Returns:
        Q: the updated Q table
    """
    initial_epsilon = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        eligibility = np.zeros_like(Q)

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            delta = (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            eligibility *= gamma * lambtha
            eligibility[state, action] += 1

            Q += (alpha * delta * eligibility)

            state, action = next_state, next_action

            if terminated or truncated:
                break

        epsilon = (
            min_epsilon + (initial_epsilon - min_epsilon)
            * np.exp(-epsilon_decay * ep)
        )

    return Q


"""
Q-table instead of V — Eligibility traces E are now shape
 (s, a) since we're tracking state-action pairs, not just states.

SARSA is on-policy — The next action next_action is chosen before
 the update using the same epsilon-greedy policy (not greedily).
  This is what makes it SARSA vs Q-learning.

The TD error uses Q[next_state, next_action] —
 the action you actually plan to take, not the best possible one.

Epsilon decay — After each episode, epsilon is
 reduced by epsilon_decay but clamped at min_epsilon:
ε = max(min_ε, ε - epsilon_decay)
This shifts behavior from exploration-heavy
 early on to more exploitation as training progresses.

Everything else — accumulating traces,
 updating all (s,a) pairs by α·δ·E, then decaying E *= γ·λ —
  follows the exact same pattern as TD(λ).
"""
