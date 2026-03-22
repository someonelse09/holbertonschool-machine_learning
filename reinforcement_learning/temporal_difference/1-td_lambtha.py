#!/usr/bin/env python3
"""This module includes the function
that performs the TD(λ) algorithm"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Args:
        env is the environment instance
        V is a numpy.ndarray of shape (s,)
         containing the value estimate
        policy is a function that takes in a state
         and returns the next action to take
        lambtha is the eligibility trace factor
        episodes is the total number of episodes to train over
        max_steps is the maximum number of steps per episode
        alpha is the learning rate
        gamma is the discount rate
    Returns:
        V, the updated value estimate
    """
    n_states = V.shape[0]
    for _ in range(episodes):
        state, _ = env.reset()
        E = np.zeros(n_states)
        for _ in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[new_state] * (not terminated) - V[state]

            E[state] += 1
            V += alpha * delta * E
            E *= (gamma * lambtha)

            state = new_state
            if terminated or truncated:
                break

    return V


"""
Eligibility Traces E — Reset to zeros at the start of every episode.
They act as a memory of which states were recently visited and
how much credit they should receive for the current TD error.

TD Error δ — Computed at each step as:
δ = r + γ·V(s') - V(s)

When the episode terminates, the bootstrap target is just r
 (no next state value), handled by * (not terminated).
Online Updates — Unlike Monte Carlo which waits until episode end,
 TD(λ) updates every step:
E[s] += 1          # accumulate trace for current state
V   += α·δ·E       # update ALL states by their trace weight
E   *= γ·λ         # decay all traces
λ Controls the Spectrum — With λ=0 it reduces to pure TD(0)
(only current state updated), and with λ=1 it approximates
Monte Carlo (credit flows back through the whole trajectory).
"""
