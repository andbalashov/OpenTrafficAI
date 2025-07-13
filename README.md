# OpenTrafficAI

OpenTrafficAI is an **open‑source Python toolkit for adaptive traffic‑signal control with Reinforcement Learning**.

* **Multi‑intersection Gymnasium environment** simulating traffic queues.
* **Reference agents** implemented in PyTorch (DQN today, PPO stub for extension).
* **Training / evaluation scripts**, simple visualisation, and unit‑tests.
* **Docker & GitHub CI** for easy reproducibility.

## Quick start
```bash
git clone https://github.com/andbalashov/OpenTrafficAI.git
cd OpenTrafficAI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m traffic_ai.scripts.train --episodes 1000 --n 4 --render
