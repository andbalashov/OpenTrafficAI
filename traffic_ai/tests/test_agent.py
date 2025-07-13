from traffic_ai.agents import DQNAgent
import numpy as np

def test_agent_act():
    agent = DQNAgent(obs_shape=(4, 4), action_size=16)
    action = agent.select_action(np.zeros((4, 4)), epsilon=0.0)
    assert 0 <= action < 16