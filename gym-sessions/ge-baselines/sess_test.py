from run import GymSession
import numpy as np

env_stub = lambda: None
algo_stub = lambda: None
sess = GymSession(env_stub, algo_stub)


def test_reward_to_value():
    vals = sess.reward_to_value([1.0] * 100)
    correct_val = list(np.arange(100.0, 0.0, -1.0))
    assert vals == correct_val or (vals == correct_val).all(), "vals should be a decreasing set of future accumulated values"