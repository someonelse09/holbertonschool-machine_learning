#!/usr/bin/env python3
"""This module includes the function
that performs SARSA(λ)"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000,
                  max_steps=100, alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Args:
        env is the environment instance
        Q is a numpy.ndarray of shape (s,a) containing the Q table
        lambtha is the eligibility trace factor
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
        epsilon is the initial threshold for epsilon greedy
        min_epsilon is the minimum value that epsilon should decay to
        epsilon_decay is the decay rate for updating epsilon between episodes
    Returns:
        Q:   the updated Q table
    """
    n_states, n_actions = Q.shape

    def epsilon_greedy(state, eps):
        if np.random.uniform() < eps:
            return env.action_space.sample()
        return np.argmax(Q[state])

    for _ in range(episodes):
        state, _ = env.reset()
        E = np.zeros((n_states, n_actions))
        action = epsilon_greedy(state, epsilon)
        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(next_state, epsilon)

            delta = (reward
                     + gamma * Q[next_state, next_action]
                     - Q[state, action])

            E[state, action] += 1
            Q += alpha * delta * E
            E *= (gamma * lambtha)

            state = next_state
            action = next_action

            if terminated or truncated:
                break

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

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
