import numpy as np

FRANKA_ARM_XML = """
<mujoco model="franka_scene">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <default>
    <geom solimp="0.998 0.998 0.001" solref="0.01 1"/>
    <joint limited="true" damping="1.0" armature="0.01"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.1 0.2 0.3" width="512" height="512"/>
  </asset>

  <worldbody>
    <light name="light0" diffuse="0.8 0.8 0.8" pos="1 0 2" dir="-1 0 -2"/>
    <camera name="head" pos="0.8 -0.5 0.8" fovy="60" mode="fixed"/>
    <geom name="floor" type="plane" pos="0 0 0" size="2 2 0.1" rgba="0.4 0.4 0.4 1"/>

    <body name="table" pos="0.40 0 0">
      <geom name="table_top" type="box" size="0.3 0.3 0.02" pos="0 0 0.35" rgba="0.6 0.4 0.2 1"/>
      <geom name="table_leg1" type="cylinder" size="0.02 0.35" pos="-0.25 -0.25 0.175" rgba="0.5 0.3 0.1 1"/>
      <geom name="table_leg2" type="cylinder" size="0.02 0.35" pos="-0.25 0.25 0.175" rgba="0.5 0.3 0.1 1"/>
      <geom name="table_leg3" type="cylinder" size="0.02 0.35" pos="0.25 -0.25 0.175" rgba="0.5 0.3 0.1 1"/>
      <geom name="table_leg4" type="cylinder" size="0.02 0.35" pos="0.25 0.25 0.175" rgba="0.5 0.3 0.1 1"/>
    </body>

    <body name="base" pos="0 0 0">
      <geom type="box" size="0.15 0.1 0.05" pos="0 0 0.05" rgba="0.5 0.5 0.5 1"/>
      <body name="link0" pos="0 0 0.05">
        <joint name="joint1" type="hinge" pos="0 0 0" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.333" size="0.045" rgba="0.8 0.2 0.2 1"/>
        <body name="link1" pos="0 0 0.333">
          <joint name="joint2" type="hinge" pos="0 0 0" axis="0 -1 0" range="-1.7628 1.7628"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.333" size="0.045" rgba="0.2 0.8 0.2 1"/>
          <body name="link2" pos="0 0 0.333">
            <joint name="joint3" type="hinge" pos="0 0 0" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.316" size="0.045" rgba="0.2 0.2 0.8 1"/>
            <body name="link3" pos="0 0 0.316">
              <joint name="joint4" type="hinge" pos="0 0 0" axis="0 -1 0" range="-3.0718 0.0698"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.384" size="0.045" rgba="0.8 0.8 0.2 1"/>
              <body name="link4" pos="0 0 0.384">
                <joint name="joint5" type="hinge" pos="0 0 0" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.384" size="0.045" rgba="0.8 0.2 0.8 1"/>
                <body name="link5" pos="0 0 0.384">
                  <joint name="joint6" type="hinge" pos="0 0 0" axis="0 -1 0" range="-0.0873 0.0873"/>
                  <geom type="capsule" fromto="0 0 0 0 0 0.210" size="0.045" rgba="0.2 0.8 0.8 1"/>
                  <body name="link6" pos="0 0 0.210">
                    <joint name="joint7" type="hinge" pos="0 0 0" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom type="capsule" fromto="0 0 0 0 0 0.180" size="0.04" rgba="0.5 0.5 0.5 1"/>
                    <body name="end_effector" pos="0 0 0.180">
                      <geom type="box" size="0.03 0.03 0.03" pos="0 0 0.02" rgba="0.9 0.6 0.1 1"/>
                      <body name="left_finger" pos="0 -0.025 0.02">
                        <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
                        <geom name="left_finger_geom" type="box" size="0.015 0.005 0.03" rgba="0.9 0.6 0.1 1"/>
                      </body>
                      <body name="right_finger" pos="0 0.025 0.02">
                        <joint name="finger_joint2" type="slide" axis="0 -1 0" range="0 0.04"/>
                        <geom name="right_finger_geom" type="box" size="0.015 0.005 0.03" rgba="0.9 0.6 0.1 1"/>
                      </body>
                      <site name="grasp" pos="0 0 0.05" size="0.01"/>
                      <site name="ee" pos="0 0 0.05" size="0.005"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <body name="target_marker" pos="0.45 0 0.40">
      <geom name="target_vis" type="sphere" size="0.025" rgba="0 1 0 0.5"/>
    </body>
  </worldbody>

  <actuator>
    <position name="arm1" joint="joint1" kp="200" kv="10"/>
    <position name="arm2" joint="joint2" kp="200" kv="10"/>
    <position name="arm3" joint="joint3" kp="200" kv="10"/>
    <position name="arm4" joint="joint4" kp="200" kv="10"/>
    <position name="arm5" joint="joint5" kp="200" kv="10"/>
    <position name="arm6" joint="joint6" kp="200" kv="10"/>
    <position name="arm7" joint="joint7" kp="200" kv="10"/>
    <position name="finger1" joint="finger_joint1" kp="100"/>
    <position name="finger2" joint="finger_joint2" kp="100"/>
  </actuator>
</mujoco>
"""

OBJECT_BLOCK_XML = """
<body name="object" pos="{x} {y} {z}">
  <joint name="obj_free" type="free" limited="false"/>
  <geom name="obj_geom" type="box" size="0.025 0.025 0.025" rgba="1 0.3 0.3 1" mass="0.05" solimp="0.995 0.995 0.001"/>
</body>
"""

TARGET_SITE_XML = """
<site name="target_site" pos="{x} {y} {z}" size="0.02" type="sphere" rgba="0 1 0 0.5"/>
"""


def _ik_solve(model, data, target_pos, max_iter=100, tol=1e-3):
    import mujoco
    body_id = model.body("end_effector").id
    site_id = model.site("grasp").id
    jacp = np.zeros((3, model.nv))
    lam = 0.5
    for _ in range(max_iter):
        mujoco.mj_fwdPosition(model, data)
        err = target_pos - data.site(site_id).xpos.copy()
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacBody(model, data, jacp, None, body_id)
        J = jacp[:, :7].copy()
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), err)
        data.qpos[:7] += dq
        lam *= 0.95
    mujoco.mj_fwdPosition(model, data)


class FrankaEnv:
    def __init__(self, task_config=None):
        import mujoco

        self.task_config = task_config or {}

        xml = self._build_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.control_freq = self.task_config.get("control_freq", 25)
        sim_timestep = self.model.opt.timestep
        self.n_substeps = max(1, int(1.0 / self.control_freq / sim_timestep))

        self.max_episode_steps = self.task_config.get("max_episode_steps", 200)

        self.obs_dim = self._get_obs_dim()
        self.act_dim = 7

        self._joint_qpos0 = np.array([0.0, 0.0, 0.0, -2.0, 0.0, -1.0, 0.0])

    def _build_xml(self):
        xml = FRANKA_ARM_XML

        table_surface_z = 0.37
        obj_x = self.task_config.get("object_init_x", 0.45)
        obj_y = self.task_config.get("object_init_y", 0.0)
        obj_z = table_surface_z + 0.03
        xml = xml.replace("</worldbody>", OBJECT_BLOCK_XML.format(x=obj_x, y=obj_y, z=obj_z) + "\n</worldbody>")

        target_x = self.task_config.get("target_x", 0.55)
        target_y = self.task_config.get("target_y", 0.10)
        target_z = table_surface_z + 0.005
        xml = xml.replace("</worldbody>", TARGET_SITE_XML.format(x=target_x, y=target_y, z=target_z) + "\n</worldbody>")

        return xml

    def _get_obs_dim(self):
        return 22

    def _get_obs(self):
        qpos = self.data.qpos[:7].copy()
        qvel = self.data.qvel[:7].copy()
        ee_pos = self.data.site("ee").xpos.copy()
        ee_quat = self.data.site("ee").xmat.reshape(9)[[0, 4, 8, 1, 2, 3, 5, 6, 7]]
        r = ee_quat.copy()
        tr = r[0] + r[4] + r[8]
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (r[7] - r[5]) / s
            y = (r[2] - r[6]) / s
            z = (r[3] - r[1]) / s
        elif r[0] > r[4] and r[0] > r[8]:
            s = np.sqrt(1.0 + r[0] - r[4] - r[8]) * 2
            w = (r[7] - r[5]) / s
            x = 0.25 * s
            y = (r[1] + r[3]) / s
            z = (r[2] + r[6]) / s
        elif r[4] > r[8]:
            s = np.sqrt(1.0 + r[4] - r[0] - r[8]) * 2
            w = (r[2] - r[6]) / s
            x = (r[1] + r[3]) / s
            y = 0.25 * s
            z = (r[5] + r[7]) / s
        else:
            s = np.sqrt(1.0 + r[8] - r[0] - r[4]) * 2
            w = (r[3] - r[1]) / s
            x = (r[2] + r[6]) / s
            y = (r[5] + r[7]) / s
            z = 0.25 * s
        ee_quat_xyzw = np.array([x, y, z, w])

        grip = (self.data.qpos[7] + self.data.qpos[8]) / 2.0
        obs = np.concatenate([qpos, qvel, ee_pos, ee_quat_xyzw, [grip]])
        return obs.astype(np.float32)

    def _default_obs(self):
        return np.zeros(self.obs_dim, dtype=np.float32)

    def reset(self, seed=None):
        import mujoco

        mujoco.mj_resetData(self.model, self.data)
        qpos0 = self._joint_qpos0.copy()
        qpos0[:7] += np.random.uniform(-0.05, 0.05, 7)
        self.data.qpos[:7] = qpos0
        self.data.qpos[7:9] = 0.0

        if self.task_config.get("use_domain_randomization", False):
            self.model.body_mass[self.model.body("object").id] *= np.random.uniform(0.5, 1.5)

        self._reset_object()
        self._reset_target()

        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:7] = self.data.qpos[:7].copy()
        self.data.ctrl[7:9] = 0.02
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        return self._get_obs()

    def _reset_object(self):
        table_z = 0.37
        body_id = self.model.body("object").id
        jnt_id = self.model.body_jntadr[body_id]
        self.data.qpos[jnt_id : jnt_id + 3] = [
            np.random.uniform(0.35, 0.55),
            np.random.uniform(-0.15, 0.15),
            table_z + 0.03,
        ]
        self.data.qpos[jnt_id + 3 : jnt_id + 7] = [1, 0, 0, 0]

    def _reset_target(self):
        target_x = np.random.uniform(0.60, 0.90)
        target_y = np.random.uniform(-0.15, 0.15)
        target_z = np.random.uniform(0.15, 0.40)
        site_id = self.model.site("target_site").id
        self.model.site_pos[site_id] = [target_x, target_y, target_z]

        marker_id = self.model.body("target_marker").id
        self.model.body_pos[marker_id] = [target_x, target_y, target_z]

    def step(self, action):
        import mujoco

        action = np.clip(action, -0.1, 0.1)
        target = self.data.qpos[:7].copy() + action
        target = np.clip(target, self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])

        self.data.ctrl[:7] = target
        self.data.ctrl[7:9] = 0.02

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward = self.compute_reward()
        done = self._step_count >= self.max_episode_steps
        info = {"success": self.is_success(), "step": self._step_count}

        return obs, reward, done, info

    def compute_reward(self):
        return 0.0

    def is_success(self):
        return False

    def get_ee_pos(self):
        return self.data.site("ee").xpos.copy()

    def get_object_pos(self):
        return self.data.body("object").xpos.copy()

    def get_target_pos(self):
        return self.model.site_pos[self.model.site("target_site").id].copy()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def render(self, width=224, height=224):
        import mujoco

        renderer = mujoco.Renderer(self.model, width, height)
        try:
            renderer.update_scene(self.data, camera="head")
            return renderer.render()
        finally:
            renderer.close()
