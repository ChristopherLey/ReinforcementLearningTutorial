import gymnasium as gym
import numpy as np
from gymnasium.utils.play import play, PlayPlot


def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    return [rew, ]


plotter = PlayPlot(
    callback, 150, ["reward"])

play(gym.make("CarRacing-v2", render_mode="rgb_array"),
     keys_to_action={
        "w": np.array([0, 0.7, 0], dtype=np.float32),
        "a": np.array([-1, 0, 0], dtype=np.float32),
        "s": np.array([0, 0, 1], dtype=np.float32),
        "d": np.array([1, 0, 0], dtype=np.float32),
        "wa": np.array([-1, 0.7, 0], dtype=np.float32),
        "dw": np.array([1, 0.7, 0], dtype=np.float32),
        "ds": np.array([1, 0, 1], dtype=np.float32),
        "as": np.array([-1, 0, 1], dtype=np.float32),
    },
    noop=np.array([0, 0, 0], dtype=np.float32),
    callback=plotter.callback
)
