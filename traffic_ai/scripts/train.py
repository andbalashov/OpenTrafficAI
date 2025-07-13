#!/usr/bin/env python
"""Train a DQN agent on the TrafficEnv."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from traffic_ai.environment import TrafficEnv
from traffic_ai.agents import DQNAgent


def index_to_action(idx: int, n: int):
    """Convert scalar index to binary phase vector of length *n*."""
    return [(idx >> i) & 1 for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--n", type=int, default=4, help="Number of intersections")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = TrafficEnv(n_intersections=args.n)
    action_size = 2 ** args.n  # binary phases per intersection
    agent = DQNAgent(obs_shape=env.observation_space.shape, action_size=action_size)

    eps_start, eps_end, eps_decay = 1.0, 0.05, 0.995
    epsilon = eps_start

    for ep in range(args.episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action_idx = agent.select_action(state, epsilon)
            action_vec = index_to_action(action_idx, args.n)
            next_state, reward, terminated, truncated, _ = env.step(action_vec)
            done = terminated or truncated
            agent.store(state, action_idx, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
            if args.render:
                env.render()
        epsilon = max(eps_end, epsilon * eps_decay)
        if ep % 10 == 0:
            print(f"Episode {ep:4d} | Return {total_reward:8.1f} | Epsilon {epsilon:.3f}")

    # save weights
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(agent.policy_net.state_dict(), ckpt_dir / f"dqn_{ts}.pth")


if __name__ == "__main__":
    main()