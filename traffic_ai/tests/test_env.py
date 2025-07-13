from traffic_ai.environment import TrafficEnv

def test_env_basic():
    env = TrafficEnv()
    obs, _ = env.reset()
    assert obs.shape == (4, 4)
    action = [0, 0, 0, 0]
    next_obs, reward, terminated, truncated, _ = env.step(action)
    assert next_obs.shape == (4, 4)
    assert isinstance(reward, float)