from __future__ import annotations
import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    """Deep Q‑Network agent with target‑network & experience replay."""

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        memory_size: int = 100_000,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_size = int(np.prod(obs_shape))
        self.action_size = action_size

        self.policy_net = QNetwork(self.obs_size, self.action_size).to(self.device)
        self.target_net = QNetwork(self.obs_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory: deque = deque(maxlen=memory_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.step_count = 0
        self.update_target_every = 500

    # ϵ‑greedy
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).flatten().unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax().item())

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state.copy(), action, reward, next_state.copy(), done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device).view(self.batch_size, -1)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device).view(self.batch_size, -1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())