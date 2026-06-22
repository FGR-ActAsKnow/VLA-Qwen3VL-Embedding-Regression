import numpy as np
from .franka_env import FrankaEnv


class PushEnv(FrankaEnv):
    def __init__(self, task_config=None):
        super().__init__(task_config)
        self.obs_dim = 28
        self._last_obj_to_target = None

    def _get_obs(self):
        base_obs = super()._get_obs()
        obj_pos = self.get_object_pos()
        target_pos = self.get_target_pos()
        return np.concatenate([base_obs, obj_pos, target_pos]).astype(np.float32)

    def reset(self, seed=None):
        obs = super().reset(seed)
        self._last_obj_to_target = np.linalg.norm(self.get_object_pos() - self.get_target_pos())
        self._best_dist = self._last_obj_to_target
        self._rewarded_near = self._last_obj_to_target < 0.10
        return obs

    def compute_reward(self):
        obj = self.get_object_pos()
        target = self.get_target_pos()
        dist = np.linalg.norm(obj - target)

        if self._last_obj_to_target is None:
            self._last_obj_to_target = dist

        delta = self._last_obj_to_target - dist
        self._last_obj_to_target = dist
        self._best_dist = min(self._best_dist, dist)

        push_reward = -2.0 * dist + 2.0 * delta

        milestone_bonus = 0.0
        if dist < 0.10 and not self._rewarded_near:
            milestone_bonus += 20.0
            self._rewarded_near = True

        success_bonus = 100.0 if dist < 0.05 else 0.0
        return float(push_reward - 0.001 * np.sum(np.square(self.last_action)) + milestone_bonus + success_bonus)

    def is_success(self):
        return np.linalg.norm(self.get_object_pos() - self.get_target_pos()) < 0.05

    def step(self, action):
        self.last_action = np.clip(action, -0.1, 0.1)
        obs, reward, done, info = super().step(action)
        info["best_dist"] = self._best_dist
        info["dist"] = np.linalg.norm(self.get_object_pos() - self.get_target_pos())
        obj = self.get_object_pos()
        target = self.get_target_pos()
        info["obj_to_target"] = float(np.linalg.norm(obj - target))
        return obs, reward, done, info
