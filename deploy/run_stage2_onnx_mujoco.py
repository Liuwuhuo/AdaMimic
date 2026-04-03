#!/usr/bin/env python3
"""
Run AdaMimic stage2 ONNX policy in MuJoCo (adam_inspire scene).

Usage example:
python deploy/run_stage2_onnx_mujoco.py \
  --onnx /path/to/policy.onnx \
  --info /path/to/info.json \
  --xml deploy/robot/adam_inspire/scene.xml
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import deque

import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS_DIR = REPO_ROOT / ".python_deps"
if LOCAL_DEPS_DIR.exists():
    sys.path.insert(0, str(LOCAL_DEPS_DIR))

try:
    import mujoco
    import mujoco.viewer
except ImportError as e:
    raise ImportError(
        "Missing dependency 'mujoco'. Install in current env with:\n"
        "  python3 -m pip install mujoco -i https://pypi.tuna.tsinghua.edu.cn/simple"
    ) from e

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "Missing dependency 'onnxruntime'. Install in current env with:\n"
        "  python3 -m pip install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple"
    ) from e

try:
    import pygame
except Exception:
    pygame = None


EXPORT_DIR = REPO_ROOT / "exports"
DEFAULT_INFO_PATH = EXPORT_DIR / "info.json"
DEFAULT_XML_PATH = REPO_ROOT / "deploy/robot/adam_inspire/scene.xml"
DEFAULT_CONFIG_PATH = REPO_ROOT / "deploy/run_stage2_onnx_mujoco.config.json"
BUILTIN_ARMATURE_ADAM_SP = {
    "hipPitch": 0.13426,
    "hipRoll": 0.281573,
    "hipYaw": 0.23409,
    "kneePitch": 0.13426,
    "anklePitch": 0.0549,
    "ankleRoll": 0.0549,
    "waistRoll": 0.23409,
    "waistPitch": 0.23409,
    "waistYaw": 0.23409,
    "shoulderPitch": 0.01,
    "shoulderRoll": 0.01,
    "shoulderYaw": 0.01,
    "elbow": 0.01,
    "wristYaw": 0.01,
    "wristPitch": 0.01,
    "wristRoll": 0.01,
}


def resolve_default_onnx():
    candidates = sorted(EXPORT_DIR.glob("*stage2*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    return EXPORT_DIR / "stage2.onnx"


def quat_conjugate(q):
    # q: [w, x, y, z]
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul(a, b):
    # a,b: [w,x,y,z]
    return np.array([
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ], dtype=np.float32)


def quat_rotate_inverse(q, v):
    # inverse-rotate vector v by quat q (wxyz)
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return quat_mul(quat_mul(quat_conjugate(q), vq), q)[1:]


def build_kp_kd(dof_names, stiffness_cfg, damping_cfg):
    def match_gain(name, cfg, default=0.0):
        for key, val in cfg.items():
            if key in name:
                return float(val)
        return default

    kp = np.array([match_gain(n, stiffness_cfg) for n in dof_names], dtype=np.float32)
    kd = np.array([match_gain(n, damping_cfg) for n in dof_names], dtype=np.float32)
    return kp, kd


def build_by_name_map(dof_names, value_map, default=0.0):
    def match_val(name, cfg, d):
        for key, val in cfg.items():
            if key in name:
                return float(val)
        return float(d)
    return np.array([match_val(n, value_map, default) for n in dof_names], dtype=np.float32)


def load_runtime_config(config_path):
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Runtime config not found: {path}")
    with open(path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must be a JSON object: {path}")
    return cfg


def resolve_model_paths(export_dir, onnx_value, info_value):
    export_path = None
    if export_dir:
        export_path = Path(export_dir).expanduser().resolve()
        if not export_path.exists():
            raise FileNotFoundError(f"export_dir not found: {export_path}")
        if not export_path.is_dir():
            raise NotADirectoryError(f"export_dir is not a directory: {export_path}")

    def resolve_one(value, default_name=None, suffix=None):
        if value:
            p = Path(value).expanduser()
            if p.is_absolute():
                return p.resolve()
            if export_path is not None:
                p2 = (export_path / p).resolve()
                if p2.exists():
                    return p2
            return p.resolve()
        if export_path is not None:
            if default_name is not None:
                candidate = (export_path / default_name).resolve()
                if candidate.exists():
                    return candidate
            if suffix is not None:
                candidates = sorted(export_path.glob(f"*{suffix}"), key=lambda x: x.stat().st_mtime, reverse=True)
                if candidates:
                    return candidates[0].resolve()
        return None

    onnx_path = resolve_one(onnx_value, default_name=None, suffix=".onnx")
    info_path = resolve_one(info_value, default_name="info.json", suffix=None)
    return onnx_path, info_path


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to runtime JSON config.")
    bootstrap.add_argument("--verbose", action="store_true", help="Enable console logs (overrides config).")
    bootstrap_args, _ = bootstrap.parse_known_args()
    runtime_cfg = load_runtime_config(bootstrap_args.config)

    required_keys = [
        "onnx", "info", "xml", "export_dir",
        "control_every", "history_len", "one_step_obs",
        "motion_period", "motion_fps", "max_steps",
        "start_enabled", "joystick_index",
        "button_a", "button_b", "button_x",
        "print_buttons",
        "obs_ang_vel_scale", "obs_dof_pos_scale", "obs_dof_vel_scale", "obs_norm_time_scale",
        "lidar_update_interval", "infer_curriculum", "terrain_difficulty",
        "joint_action_clip", "fixed_action_time",
        "debug_every", "viewer_sync_every",
        "apply_armature", "clip_default_to_limits", "clip_target_to_limits",
        "headless", "verbose",
    ]
    missing = [k for k in required_keys if k not in runtime_cfg]
    if missing:
        raise KeyError(
            "Missing required config keys in {}: {}".format(
                Path(bootstrap_args.config).expanduser().resolve(),
                ", ".join(missing),
            )
        )

    merged_cfg = dict(runtime_cfg)
    if bootstrap_args.verbose:
        merged_cfg["verbose"] = True
    args = argparse.Namespace(config=bootstrap_args.config, **merged_cfg)

    def log(*msg):
        if args.verbose:
            print(*msg)

    onnx_path, info_path = resolve_model_paths(args.export_dir, args.onnx, args.info)
    if onnx_path is None:
        raise FileNotFoundError(
            "Cannot resolve onnx path. Set config['onnx'] or provide config['export_dir'] with at least one .onnx file."
        )
    if info_path is None:
        raise FileNotFoundError(
            "Cannot resolve info path. Set config['info'] or provide config['export_dir'] containing info.json."
        )
    xml_path = Path(args.xml).expanduser().resolve()

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found: {info_path}")
    if not xml_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

    log(f"[config] onnx: {onnx_path}")
    log(f"[config] info: {info_path}")
    log(f"[config] xml : {xml_path}")
    log("[control] keyboard: A=reset(default pose), X=enter model-stand, Y=single jump, B=disable model")
    log(f"[control] joystick target: js{args.joystick_index}, buttons A/B/X/Y={args.button_a}/{args.button_b}/{args.button_x}/{args.button_y}")
    log(f"[config] control_every={args.control_every}, joint_action_clip={args.joint_action_clip}, fixed_action_time={args.fixed_action_time}")

    with open(info_path, "r") as f:
        info = json.load(f)

    dof_names = info["DOF NAMES"]
    action_scale = np.asarray(info["ACTION SCALE"], dtype=np.float32).reshape(-1)
    default_dof_pos = np.asarray(info["DEFAULT DOF POS"], dtype=np.float32).reshape(-1)
    torque_limits = np.asarray(info["TORQUE LIMITS"], dtype=np.float32).reshape(-1)
    stiffness_cfg = info["STIFFNESS"]
    damping_cfg = info["DAMPING"]
    kp, kd = build_kp_kd(dof_names, stiffness_cfg, damping_cfg)

    motion_length = float(info.get("MOTION LENGTH", 0.0))
    if args.motion_period > 0:
        motion_period = float(args.motion_period)
    elif motion_length > 0 and args.motion_fps > 0:
        motion_period = motion_length / float(args.motion_fps)
    else:
        motion_period = 6.0
    log(f"[config] motion_period={motion_period:.6f}s (motion_length={motion_length}, fps={args.motion_fps})")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Map policy dof names -> MuJoCo joint addresses
    qpos_adr = []
    qvel_adr = []
    act_adr = []
    joint_ids = []
    for name in dof_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in XML.")
        joint_ids.append(jid)
        qpos_adr.append(model.jnt_qposadr[jid])
        qvel_adr.append(model.jnt_dofadr[jid])

        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise ValueError(f"Actuator '{name}' not found in XML.")
        act_adr.append(aid)

    qpos_adr = np.asarray(qpos_adr, dtype=np.int32)
    qvel_adr = np.asarray(qvel_adr, dtype=np.int32)
    act_adr = np.asarray(act_adr, dtype=np.int32)
    joint_ids = np.asarray(joint_ids, dtype=np.int32)

    # Align MuJoCo effective rotor inertia with training-side armature settings.
    if args.apply_armature:
        armature_vec = build_by_name_map(dof_names, BUILTIN_ARMATURE_ADAM_SP, default=0.0)
        model.dof_armature[qvel_adr] = armature_vec
        log("[config] applied builtin armature map for adam_sp")

    joint_lower = model.jnt_range[joint_ids, 0].astype(np.float32)
    joint_upper = model.jnt_range[joint_ids, 1].astype(np.float32)
    if args.clip_default_to_limits:
        default_dof_pos = np.clip(default_dof_pos, joint_lower, joint_upper)

    upper_body_name = "pelvis"
    upper_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, upper_body_name)
    if upper_bid < 0:
        raise ValueError("Body 'pelvis' not found in XML.")

    odom_body_name = "torso"
    odom_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, odom_body_name)
    if odom_bid < 0:
        odom_bid = upper_bid

    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    num_dof = len(dof_names)
    num_actions = num_dof + 1  # infer_keyframe_time=true -> last dim is action_time
    obs_dim = args.one_step_obs
    hist_len = args.history_len
    actor_obs_dim = obs_dim * hist_len

    obs_hist = deque([np.zeros(obs_dim, dtype=np.float32) for _ in range(hist_len)], maxlen=hist_len)
    prev_actions = np.zeros(num_actions, dtype=np.float32)
    last_joint_action = np.zeros(num_dof, dtype=np.float32)

    # For lidar_pos-like odometry in one-step actor obs
    init_odom_pos = None
    init_odom_quat = None
    lidar_pos_cached = np.zeros(3, dtype=np.float32)
    elapsed_t = 0.0
    dt_scale = 1.0
    action_time_last = 0.02  # fixed_dt default
    infer_curriculum = float(args.infer_curriculum)
    terrain_difficulty = float(args.terrain_difficulty)

    clip_actions = 100.0

    def print_default_pose(tag="default"):
        log(f"[{tag}] dof count={num_dof}")
        for i, name in enumerate(dof_names):
            log(f"[{tag}] {i:02d} {name:>24s} = {float(np.asarray(default_dof_pos[i]).reshape(-1)[0]): .6f}")

    # Runtime control states
    state = {
        "policy_enabled": bool(args.start_enabled),
        "reset_to_default": False,
        "quit": False,
        "hold_default": False,
        # single-jump state machine
        "start_single_jump": False,
        "single_jump_active": False,
    }
    jump_start_t = 0.0

    def reset_robot_to_default():
        # Reset full simulation state first, then overwrite policy-controlled joints.
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_adr] = default_dof_pos
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        log(f"[reset] base_z={float(data.xpos[upper_bid][2]):.4f}")
        # Reset odometry state to match training episode reset behavior.
        nonlocal init_odom_pos, init_odom_quat, lidar_pos_cached, elapsed_t, action_time_last, last_joint_action
        init_odom_pos = None
        init_odom_quat = None
        lidar_pos_cached[:] = 0.0
        elapsed_t = 0.0
        action_time_last = 0.02
        obs_hist.clear()
        for _ in range(hist_len):
            obs_hist.append(np.zeros(obs_dim, dtype=np.float32))
        prev_actions[:] = 0.0
        last_joint_action[:] = 0.0

    # Keyboard callback for MuJoCo viewer: a/x/y/b
    def key_callback(keycode):
        # GLFW key codes: A/a=65/97, B/b=66/98, X/x=88/120, Y/y=89/121
        if keycode in (65, 97):
            state["reset_to_default"] = True
            state["policy_enabled"] = False
            state["hold_default"] = True
            log("[control] A pressed: reset to default_dof_pos + HOLD_DEFAULT")
        elif keycode in (88, 120):
            # X: 进入“模型接管但保持站立”的模式（不主动推进时间，相当于固定在当前相位）
            state["policy_enabled"] = True
            state["hold_default"] = False
            state["single_jump_active"] = False
            log("[control] X pressed: MODEL STAND (policy enabled, time frozen)")
        elif keycode in (89, 121):
            # Y: 触发单次跳跃；若模型未启用，则先启用模型
            state["start_single_jump"] = True
            log("[control] Y pressed: request SINGLE JUMP")
        elif keycode in (66, 98):
            state["policy_enabled"] = False
            state["hold_default"] = False
            state["single_jump_active"] = False
            log("[control] B pressed: MODEL DISABLED (zero torque)")

    # Optional gamepad support (Xbox mapping: A=0, B=1, X=2, Y=3)
    joystick = None
    last_buttons = {"a": 0, "b": 0, "x": 0, "y": 0}
    if pygame is not None:
        try:
            pygame.init()
            pygame.joystick.init()
            joystick_count = pygame.joystick.get_count()
            log(f"[control] detected joystick count: {joystick_count}")
            for j in range(joystick_count):
                log(f"[control] js{j}: {pygame.joystick.Joystick(j).get_name()}")

            if joystick_count > args.joystick_index:
                joystick = pygame.joystick.Joystick(args.joystick_index)
                joystick.init()
                log(f"[control] Gamepad bound to js{args.joystick_index}: {joystick.get_name()} (A=reset, X=stand, Y=single, B=disable)")
            else:
                log("[control] Target joystick not found, use keyboard: a(reset) x(enable) b(disable)")
        except Exception:
            joystick = None
            log("[control] Gamepad init failed, use keyboard: a(reset) x(stand) y(single) b(disable)")
    else:
        log("[control] pygame not installed, use keyboard: a(reset) x(stand) y(single) b(disable)")

    log(f"[control] Policy starts {'ENABLED' if state['policy_enabled'] else 'DISABLED'} (X=stand, Y=single)")
    print_default_pose(tag="default_dof_pos")
    log("[control] Startup: policy DISABLED does not send command. Press A to reset default pose, X to run policy.")

    def run_sim_loop(viewer=None):
        # Use outer-scope state for odometry and last actions, matching training behavior.
        nonlocal init_odom_pos, elapsed_t, action_time_last, prev_actions, last_joint_action, lidar_pos_cached, jump_start_t
        wall_t0 = time.perf_counter()
        sim_dt = float(model.opt.timestep)
        # env_step: corresponds to IsaacGym's "decision step" frequency (policy update),
        # while `step` here is MuJoCo sim steps.
        env_step = 0
        for step in range(args.max_steps):
            if state["quit"]:
                break

            if state["reset_to_default"]:
                reset_robot_to_default()
                state["reset_to_default"] = False

            # Poll gamepad buttons
            if joystick is not None:
                pygame.event.pump()
                if args.print_buttons:
                    num_buttons = joystick.get_numbuttons()
                    pressed = [str(i) for i in range(num_buttons) if joystick.get_button(i)]
                    if pressed:
                        log(f"[control] pressed buttons: {', '.join(pressed)}")
                a_now = joystick.get_button(args.button_a)
                b_now = joystick.get_button(args.button_b)
                x_now = joystick.get_button(args.button_x)
                y_now = joystick.get_button(args.button_y)
                if a_now and not last_buttons["a"]:
                    state["reset_to_default"] = True
                    state["policy_enabled"] = False
                    state["hold_default"] = True
                    log("[control] Gamepad A: reset to default_dof_pos + HOLD_DEFAULT")
                if x_now and not last_buttons["x"]:
                    state["policy_enabled"] = True
                    state["hold_default"] = False
                    state["single_jump_active"] = False
                    log("[control] Gamepad X: policy ENABLED (continuous)")
                if y_now and not last_buttons["y"]:
                    state["start_single_jump"] = True
                    log("[control] Gamepad Y: request SINGLE JUMP")
                if b_now and not last_buttons["b"]:
                    state["policy_enabled"] = False
                    state["hold_default"] = False
                    state["single_jump_active"] = False
                    log("[control] Gamepad B: policy DISABLED")
                last_buttons["a"], last_buttons["b"], last_buttons["x"], last_buttons["y"] = a_now, b_now, x_now, y_now

            # Handle single-jump trigger at sim loop level (both keyboard and gamepad).
            # 仅在当前不在跳跃中时响应 Y，以避免打断正在进行的跳跃。
            if state.get("start_single_jump") and not state.get("single_jump_active"):
                state["start_single_jump"] = False
                state["policy_enabled"] = True
                state["hold_default"] = False
                state["single_jump_active"] = True
                # 为了让每次单次跳跃从同一相位开始，这里重置 elapsed_t。
                elapsed_t = 0.0
                jump_start_t = elapsed_t

            mujoco.mj_forward(model, data)

            # Robot states
            dof_pos = data.qpos[qpos_adr].astype(np.float32)
            dof_vel = data.qvel[qvel_adr].astype(np.float32)
            is_control_step = (step % args.control_every == 0)

            # body quaternion from xquat: [w, x, y, z]
            upper_quat = data.xquat[upper_bid].astype(np.float32)
            # MuJoCo cvel layout is [angular, linear]; training uses base angular velocity.
            upper_wvel_world = data.cvel[upper_bid, 0:3].astype(np.float32)
            base_ang_vel_local = quat_rotate_inverse(upper_quat, upper_wvel_world)
            projected_gravity = quat_rotate_inverse(upper_quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))

            # Odometry-like lidar_pos (sample/hold every N steps to match training behavior).
            # Training computes this in initial lidar frame, not world frame.
            odom_pos = data.xpos[odom_bid].astype(np.float32)
            odom_quat = data.xquat[odom_bid].astype(np.float32)
            if init_odom_pos is None:
                init_odom_pos = odom_pos.copy()
                init_odom_quat = odom_quat.copy()
            interval = max(args.lidar_update_interval, 1)
            if is_control_step and ((env_step + 1) % interval == 0):
                # Match IsaacGym's lidar update cadence: update only on decision steps.
                lidar_pos_cached = quat_rotate_inverse(init_odom_quat, odom_pos - init_odom_pos)
            lidar_pos = lidar_pos_cached

            if is_control_step:
                # norm_time in [0,1] (based on IsaacGym motion_time, advanced once per control step)
                norm_time = np.array(
                    [(elapsed_t % motion_period) / motion_period * args.obs_norm_time_scale],
                    dtype=np.float32,
                )
                infer_curr = np.array([infer_curriculum], dtype=np.float32)
                terrain = np.array([terrain_difficulty], dtype=np.float32)

                # One-step actor obs (matches motion_tracking.py current_actor_obs[:num_one_step_obs]).
                one_step = np.concatenate(
                    [
                        base_ang_vel_local * args.obs_ang_vel_scale,  # 3
                        projected_gravity,  # 3
                        dof_pos * args.obs_dof_pos_scale,  # 29
                        dof_vel * args.obs_dof_vel_scale,  # 29
                        prev_actions,  # 30 (includes previous action_time)
                        norm_time,  # 1
                        infer_curr,  # 1
                        terrain,  # 1
                        lidar_pos,  # 3
                    ],
                    axis=0,
                ).astype(np.float32)

                if one_step.shape[0] != obs_dim:
                    raise RuntimeError(f"One-step obs dim mismatch: got {one_step.shape[0]}, expected {obs_dim}")

                obs_hist.append(one_step)

                if state["policy_enabled"]:
                    actor_in = np.concatenate(list(obs_hist), axis=0).reshape(1, actor_obs_dim).astype(np.float32)
                    actor_in = np.clip(actor_in, -100.0, 100.0)
                    action = sess.run([out_name], {in_name: actor_in})[0][0].astype(np.float32)
                    action = np.clip(action, -clip_actions, clip_actions)

                    prev_actions = action.copy()
                    joint_action = action[:-1]
                    if args.joint_action_clip is not None and args.joint_action_clip > 0:
                        joint_action = np.clip(joint_action, -args.joint_action_clip, args.joint_action_clip)
                    last_joint_action = joint_action.copy()

                    if args.fixed_action_time is None:
                        action_time_last = float(action[-1])
                    else:
                        action_time_last = float(args.fixed_action_time)

                    # 只有在执行单次跳跃时才推进时间；模型站立时保持相位不变。
                    if state.get("single_jump_active"):
                        elapsed_t += action_time_last * dt_scale

                # 如果正在执行单次跳跃，且已走完一个 motion_period，则结束本次跳跃，
                # 但保持 policy_enabled=True，让策略继续在当前相位附近维持站立。
                if state.get("single_jump_active") and (elapsed_t - jump_start_t) >= motion_period:
                    state["single_jump_active"] = False
                    log("[control] single jump completed -> MODEL STAND")

                env_step += 1

            # Always recompute PD each sim step from latest state.
            # This matches training behavior where torques are state-feedback, not held constant.
            if state["policy_enabled"]:
                q_target = default_dof_pos + last_joint_action * action_scale
                if args.clip_target_to_limits:
                    q_target = np.clip(q_target, joint_lower, joint_upper)
                tau = kp * (q_target - dof_pos) - kd * dof_vel
                tau = np.clip(tau, -torque_limits, torque_limits)
            elif state["hold_default"]:
                # A mode: move to and hold default pose.
                q_target = default_dof_pos
                if args.clip_target_to_limits:
                    q_target = np.clip(q_target, joint_lower, joint_upper)
                tau = kp * (q_target - dof_pos) - kd * dof_vel
                tau = np.clip(tau, -torque_limits, torque_limits)
            else:
                # Disabled mode: send no command.
                q_target = dof_pos
                tau = np.zeros_like(torque_limits, dtype=np.float32)
            data.ctrl[act_adr] = tau

            if args.debug_every > 0 and step % args.debug_every == 0:
                log(
                    "[debug] step={} policy={} q_err_max={:.3f} tau_abs_max={:.3f} dof_abs_max={:.3f} act_abs_max={:.3f} dt_pred={:.4f}".format(
                        step,
                        int(state["policy_enabled"]),
                        float(np.max(np.abs(q_target - dof_pos))),
                        float(np.max(np.abs(tau))),
                        float(np.max(np.abs(dof_pos))),
                        float(np.max(np.abs(last_joint_action))),
                        float(action_time_last),
                    )
                )

            mujoco.mj_step(model, data)
            if viewer is not None and step % max(args.viewer_sync_every, 1) == 0:
                viewer.sync()
            # Keep normal playback speed (1x real-time).
            target_elapsed = (step + 1) * sim_dt
            now_elapsed = time.perf_counter() - wall_t0
            sleep_s = target_elapsed - now_elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    if args.headless:
        log("[run] headless mode enabled (no viewer).")
        run_sim_loop(viewer=None)
    else:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            run_sim_loop(viewer=viewer)

    if joystick is not None:
        try:
            joystick.quit()
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()

