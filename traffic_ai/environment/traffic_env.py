import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TrafficEnv(gym.Env):
    """Simplified multi‑intersection traffic‑signal environment."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, n_intersections: int = 4, lanes_per_intersection: int = 4, max_queue: int = 20):
        super().__init__()
        self.n = n_intersections
        self.lanes = lanes_per_intersection
        self.max_queue = max_queue

        self.observation_space = spaces.Box(0, max_queue, (self.n, self.lanes), dtype=np.float32)
        # binary phase per intersection (0 = NS green, 1 = EW green)
        self.action_space = spaces.MultiDiscrete([2] * self.n)

        self.state: np.ndarray | None = None
        self.time_step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.integers(0, self.max_queue // 2, size=(self.n, self.lanes)).astype(np.float32)
        self.time_step = 0
        return self.state, {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        # vehicles leave on green phases (lanes 0 & 2 = NS, 1 & 3 = EW)
        departures = np.zeros_like(self.state)
        for i, phase in enumerate(action):
            if phase == 0:
                departures[i, [0, 2]] = np.minimum(self.state[i, [0, 2]], 3)
            else:
                departures[i, [1, 3]] = np.minimum(self.state[i, [1, 3]], 3)
        self.state -= departures

        # random arrivals
        arrivals = self.np_random.poisson(1, size=self.state.shape)
        self.state = np.clip(self.state + arrivals, 0, self.max_queue)

        reward = -float(self.state.sum())  # minimise queues
        self.time_step += 1
        terminated = self.time_step >= 300  # episode length
        truncated = False
        return self.state, reward, terminated, truncated, {}

    def render(self, mode: str = "human"):
        if mode != "human":
            raise NotImplementedError
        print(f"t={self.time_step}\ttotal queue={int(self.state.sum())}")