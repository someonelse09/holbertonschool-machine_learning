#!/usr/bin/env python3
"""Module for training a policy gradient agent using REINFORCE."""

import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Train a policy gradient agent using the REINFORCE algorithm.

    Args:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor
        show_result: if True, render the environment every 1000 episodes

    Returns:
        scores: list of total rewards (score) for each episode
    """
    weight = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        done = False

        while not done:
            if show_result and episode % 1000 == 0:
                env.render()

            action, grad = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state

        score = sum(episode_rewards)
        scores.append(score)

        for t in range(len(episode_rewards)):
            G = sum(
                gamma ** (k - t) * episode_rewards[k]
                for k in range(t, len(episode_rewards))
            )
            _, grad = policy_gradient(episode_states[t], weight)
            weight += alpha * G * grad

        print("Episode: {} Score: {}".format(episode, score))

    return scores
