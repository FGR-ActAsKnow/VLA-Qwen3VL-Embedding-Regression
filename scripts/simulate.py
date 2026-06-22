#!/usr/bin/env python3
"""MuJoCo closed-loop simulation for Franka Panda + VLA model."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.qwen3vl_robot import Qwen3VLRobotPolicy
from src.data.dataset import collate_fn
from src.data.processing import ActionNormalizer


FRANKA_XML = """
<mujoco model="franka_panda">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light name="light0" diffuse="0.8 0.8 0.8" pos="0 0 2"/>
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
                      <geom type="box" size="0.03 0.03 0.03" rgba="0.9 0.6 0.1 1"/>
                      <body name="left_finger" pos="0 -0.025 0">
                        <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
                        <geom type="box" size="0.015 0.005 0.03" rgba="0.9 0.6 0.1 1"/>
                      </body>
                      <body name="right_finger" pos="0 0.025 0">
                        <joint name="finger_joint2" type="slide" axis="0 -1 0" range="0 0.04"/>
                        <geom type="box" size="0.015 0.005 0.03" rgba="0.9 0.6 0.1 1"/>
                      </body>
                      <site name="grasp" pos="0 0 0.03" size="0.01 0.01 0.01"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="arm1" joint="joint1" kp="200"/>
    <position name="arm2" joint="joint2" kp="200"/>
    <position name="arm3" joint="joint3" kp="200"/>
    <position name="arm4" joint="joint4" kp="200"/>
    <position name="arm5" joint="joint5" kp="200"/>
    <position name="arm6" joint="joint6" kp="200"/>
    <position name="arm7" joint="joint7" kp="200"/>
    <position name="finger1" joint="finger_joint1" kp="100"/>
    <position name="finger2" joint="finger_joint2" kp="100"/>
  </actuator>
</mujoco>
"""


def ik_solve(m, d, target_pos, max_iter=100, tol=1e-3):
    """Damped Least Squares IK for Franka arm."""
    import mujoco
    body_id = m.body("end_effector").id
    site_id = m.site("grasp").id
    jacp = np.zeros((3, m.nv))
    lam = 0.5
    for _ in range(max_iter):
        mujoco.mj_fwdPosition(m, d)
        err = target_pos - d.site(site_id).xpos.copy()
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacBody(m, d, jacp, None, body_id)
        J = jacp[:, :7].copy()
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), err)
        d.qpos[:7] += dq
        lam *= 0.95
    mujoco.mj_fwdPosition(m, d)


def render_rgb(ctx, m, d, w=224, h=224):
    """Render with available backend."""
    import mujoco
    try:
        ctx.update_scene(d, camera="head")
        return Image.fromarray(ctx.render())
    except Exception:
        return Image.new("RGB", (w, h), (128, 128, 128))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--normalizer", type=str, default="checkpoints/normalizer.json")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--record", type=str, default=None)
    args = parser.parse_args()

    import yaml
    import mujoco

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if Path("model-embedding").exists():
        config["model"]["name"] = str(Path("model-embedding").resolve())

    print("Loading VLA model...")
    model = Qwen3VLRobotPolicy.from_pretrained(args.checkpoint, config)
    model.action_head = model.action_head.to("cuda")
    model.eval()

    normalizer = ActionNormalizer()
    normalizer.load(args.normalizer)

    print("Initializing MuJoCo...")
    m = mujoco.MjModel.from_xml_string(FRANKA_XML)
    d = mujoco.MjData(m)
    mujoco.mj_fwdPosition(m, d)
    d.qpos[:7] = [0.0, 0.3, 0.0, -1.0, 0.0, 0.03, 0.0]
    mujoco.mj_fwdPosition(m, d)

    # Try to create renderer
    renderer = None
    for backend in ["egl", "osmesa", "glfw"]:
        try:
            import os
            os.environ["MUJOCO_GL"] = backend
            mujoco.Renderer(m, 224, 224).close()
            break
        except Exception:
            continue

    try:
        renderer = mujoco.Renderer(m, 224, 224)
        print(f"Renderer initialized (MUJOCO_GL={os.environ.get('MUJOCO_GL','?')})")
    except Exception as e:
        print(f"No renderer available: {e}")
        print("Simulation will use dummy images (kinematics-only mode)")

    instruction = "Execute the robot task."
    history = []
    trajectory = []

    print(f"Running simulation for {args.steps} steps...")
    for step in range(args.steps):
        if renderer:
            img = render_rgb(renderer, m, d)
        else:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        history.append(img)
        while len(history) < 4:
            history.append(img)
        if len(history) > 4:
            history.pop(0)

        sample = {"images": list(history), "action": torch.zeros(70)}
        batch = collate_fn([sample], model.processor, instruction)
        batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch.pop("action_labels", None)

        with torch.no_grad():
            outputs = model(**batch)

        delta_action = outputs["action_pred"][0, 0].cpu().numpy()
        delta_action = normalizer.denormalize(delta_action)

        target_pos = d.site("grasp").xpos.copy() + delta_action[:3] * 0.05
        ik_solve(m, d, target_pos)

        grip = float(np.clip(delta_action[6], 0, 1))
        d.qpos[7:9] = [grip * 0.02, grip * 0.02]
        mujoco.mj_forward(m, d)

        trajectory.append({
            "step": step,
            "joints": d.qpos[:7].tolist(),
            "ee_pos": d.site("grasp").xpos.copy().tolist(),
        })

        if (step + 1) % 20 == 0:
            ee = d.site("grasp").xpos
            print(f"  Step {step+1}/{args.steps} | EE: {ee[0]:.3f} {ee[1]:.3f} {ee[2]:.3f}")

    if renderer:
        renderer.close()

    if args.record:
        with open(args.record, "w") as f:
            json.dump(trajectory, f, indent=2)
        print(f"Saved: {args.record}")

    print(f"\n=== Simulation complete ===")
    print(f"Final EE: {d.site('grasp').xpos}")


if __name__ == "__main__":
    main()
