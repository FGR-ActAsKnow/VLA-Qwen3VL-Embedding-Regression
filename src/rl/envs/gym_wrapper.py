import gymnasium as gym
import numpy as np

from .reach import ReachEnv
from .push import PushEnv
from .pick_place import PickPlaceEnv


TASK_MAP = {"reach": ReachEnv, "push": PushEnv, "pick_place": PickPlaceEnv}


class FrankaGymEnv(gym.Env):
    def __init__(self, task="reach", render_mode=None, max_episode_steps=200):
        super().__init__()
        env_cls = TASK_MAP[task]
        self._env = env_cls({"max_episode_steps": max_episode_steps})
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._env.obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-0.1, high=0.1, shape=(self._env.act_dim,), dtype=np.float32
        )
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, float(reward), done, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._env.render(224, 224)

    def close(self):
        pass
