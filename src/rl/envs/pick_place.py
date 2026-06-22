import numpy as np
from .franka_env import FrankaEnv


class PickPlaceEnv(FrankaEnv):
    def __init__(self, task_config=None):
        super().__init__(task_config)
        self.obs_dim = 32
        self._stage = 0
        self._grasped = False

    def _get_obs(self):
        base_obs = super()._get_obs()
        obj_pos = self.get_object_pos()
        obj_quat = np.zeros(4)
        obj_quat[0] = 1.0
        target_pos = self.get_target_pos()
        return np.concatenate([base_obs, obj_pos, obj_quat, target_pos]).astype(np.float32)

    def reset(self, seed=None):
        obs = super().reset(seed)
        self._stage = 0
        self._grasped = False
        return obs

    def _check_grasp(self):
        ee = self.get_ee_pos()
        obj = self.get_object_pos()
        grip = (self.data.qpos[7] + self.data.qpos[8]) / 2.0
        dist_to_obj = np.linalg.norm(ee - obj)
        return dist_to_obj < 0.06 and (0.04 - grip) > 0.01

    def _check_lift(self):
        table_z = 0.37
        obj_z = self.get_object_pos()[2]
        return obj_z > table_z + 0.08 and self._grasped

    def compute_reward(self):
        ee = self.get_ee_pos()
        obj = self.get_object_pos()
        target = self.get_target_pos()
        table_z = 0.37

        dist_ee_obj = np.linalg.norm(ee - obj)
        dist_obj_target = np.linalg.norm(obj - target)

        was_grasped = self._grasped
        self._grasped = self._check_grasp()

        reward = 0.0

        if not self._grasped:
            reward += -0.5 * dist_ee_obj
            if self._grasped and not was_grasped:
                reward += 5.0
        elif not self._check_lift():
            reward += -0.3 * abs(obj[2] - (table_z + 0.12))
            if self._check_lift():
                reward += 5.0
        else:
            reward += -0.3 * dist_obj_target + 0.5 * min(0, dist_obj_target - 0.05)
            if dist_obj_target < 0.04:
                grip = (self.data.qpos[7] + self.data.qpos[8]) / 2.0
                if grip < 0.005:
                    reward += 15.0

        reward -= 0.005 * np.sum(np.square(self.last_action))
        return float(reward)

    def is_success(self):
        dist = np.linalg.norm(self.get_object_pos() - self.get_target_pos())
        grip = (self.data.qpos[7] + self.data.qpos[8]) / 2.0
        return dist < 0.04 and grip < 0.005

    def step(self, action):
        self.last_action = np.clip(action, -0.1, 0.1)
        return super().step(action)
