import numpy as np
from .franka_env import FrankaEnv


class ReachEnv(FrankaEnv):
    def __init__(self, task_config=None):
        super().__init__(task_config)
        self.obs_dim = 25

    def _get_obs(self):
        base_obs = super()._get_obs()
        target_pos = self.get_target_pos()
        return np.concatenate([base_obs, target_pos]).astype(np.float32)

    def reset(self, seed=None):
        obs = super().reset(seed)
        self._prev_dist = np.linalg.norm(self.get_ee_pos() - self.get_target_pos())
        self._best_dist = self._prev_dist
        self._rewarded_10 = self._prev_dist < 0.10
        self._rewarded_05 = self._prev_dist < 0.05
        return obs

    def compute_reward(self):
        ee = self.get_ee_pos()
        target = self.get_target_pos()
        dist = np.linalg.norm(ee - target)

        delta = self._prev_dist - dist
        self._prev_dist = dist
        self._best_dist = min(self._best_dist, dist)

        reach_reward = -2.0 * dist + 2.0 * delta

        milestone_bonus = 0.0
        if dist < 0.10 and not self._rewarded_10:
            milestone_bonus += 10.0
            self._rewarded_10 = True
        if dist < 0.05 and not self._rewarded_05:
            milestone_bonus += 20.0
            self._rewarded_05 = True

        success_bonus = 100.0 if dist < 0.03 else 0.0
        return float(reach_reward - 0.001 * np.sum(np.square(self.last_action)) + milestone_bonus + success_bonus)

    def is_success(self):
        return np.linalg.norm(self.get_ee_pos() - self.get_target_pos()) < 0.03

    def step(self, action):
        self.last_action = np.clip(action, -0.1, 0.1)
        obs, reward, done, info = super().step(action)
        info["best_dist"] = self._best_dist
        info["dist"] = np.linalg.norm(self.get_ee_pos() - self.get_target_pos())
        return obs, reward, done, info
