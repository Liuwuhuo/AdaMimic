from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import os
from pathlib import Path
import json

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, AttrDict, export_jit_to_onnx
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


def _load_snapshot_cfgs(export_policy_path, resume_path):
    """Load snapshot yaml configs with priority:
    1) export dir's config_snapshot
    2) checkpoint run dir's config_snapshot
    3) fallback to runtime env cfg
    """
    candidate_dirs = []
    if export_policy_path:
        candidate_dirs.append(Path(export_policy_path).expanduser().resolve() / "config_snapshot")
    if resume_path:
        ckpt_path = Path(resume_path).expanduser().resolve()
        candidate_dirs.append(ckpt_path.parent / "config_snapshot")

    for snapshot_dir in candidate_dirs:
        if not snapshot_dir.exists():
            continue
        loaded = {}
        for name in ("robot", "algorithm", "dataset"):
            yaml_path = snapshot_dir / f"{name}.yaml"
            if yaml_path.exists():
                loaded[name] = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
        if loaded:
            loaded["_snapshot_dir"] = str(snapshot_dir)
            return loaded
    return {}


@hydra.main(config_path="../configs", config_name="eval", version_base="1.1")
def main(cfg):
    cfg.env.terrain.curriculum = False
    cfg.env.termination_curriculum.terminate_when_motion_far_curriculum = False
    cfg.env.termination_curriculum.terminate_when_motion_far_initial_threshold = 1000
    cfg.env.termination.height_termination = False
    cfg.env.termination.rot_termination = False
    cfg.env.termination.dof_termination = False
    cfg.env.algorithm.rsi = False
    cfg.env.noise.add_noise = False
    cfg.env.domain_rand.use_random = False

    cfg = AttrDict(OmegaConf.to_container(cfg, resolve=True))
    cfg.run_dir = HydraConfig.get().runtime.output_dir
    
    cfg.env.terrain.num_rows = 4
    cfg.env.terrain.num_cols = 2
    cfg.env.env.test = True
    cfg.algo.policy.checkpoint_path = None

    env, env_cfg = task_registry.make_env_hydra(cfgs=cfg)

    obs = env.get_observations()

    cfg.algo.runner.resume = True
    cfg.algo.policy.resume = False

    ppo_runner, train_cfg = task_registry.make_alg_runner_hydra(env=env, env_cfg=env_cfg, cfgs=cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if cfg.export_policy:
        snapshot_cfgs = _load_snapshot_cfgs(
            getattr(cfg, "export_policy_path", None),
            getattr(cfg, "resume_path", None),
        )
        robot_snapshot = snapshot_cfgs.get("robot", {})
        algorithm_snapshot = snapshot_cfgs.get("algorithm", {})
        dataset_snapshot = snapshot_cfgs.get("dataset", {})
        snapshot_control = robot_snapshot.get("control", {})
        snapshot_init_state = robot_snapshot.get("init_state", {})

        stiffness = snapshot_control.get("stiffness", env.cfg.control.stiffness)
        damping = snapshot_control.get("damping", env.cfg.control.damping)
        action_scale = snapshot_control.get("action_scale", env.cfg.control.action_scale)
        default_joint_angles = snapshot_init_state.get(
            "actutaed_default_joint_angles",
            env.cfg.init_state.actutaed_default_joint_angles,
        )

        ppo_runner.export_policy_as_jit(ppo_runner.get_actor_critic(), cfg.export_policy_path, cfg.export_policy_name)
        jit_model = torch.jit.load(os.path.join(cfg.export_policy_path, cfg.export_policy_name + '.pt')).to('cuda:0')
        dummy_obs = env.get_observations()
        export_onnx_path = os.path.join(cfg.export_policy_path, cfg.export_policy_name + '.onnx')
        export_jit_to_onnx(jit_model, export_onnx_path, dummy_obs)
        print('successful export to onnx')

        info = {
            "STIFFNESS": stiffness,
            "DAMPING": damping,
            "ACTION SCALE": action_scale,
            "DEFAULT JOINT ANGLES": default_joint_angles,
            "DOF NAMES": env.dof_names,
            "KEYFRAME NAMES": env.keyframe_names,
            "DEFAULT DOF POS": env.default_dof_pos.cpu().numpy().tolist(),
            "TORQUE LIMITS": env.torque_limits.cpu().numpy().tolist(),
            "MOTION LENGTH": int(env.motions.length[0]),
            "Difficulty MIN": float(env.terrain_difficulty.min().cpu().item()),
            "Difficulty MAX": float(env.terrain_difficulty.max().cpu().item()),
            "INFO_CFG_SOURCE": "config_snapshot" if robot_snapshot else "runtime_env_cfg",
            "INFO_SNAPSHOT_DIR": snapshot_cfgs.get("_snapshot_dir", ""),
            "ROBOT YAML SNAPSHOT": robot_snapshot,
            "ALGORITHM YAML SNAPSHOT": algorithm_snapshot,
            "DATASET YAML SNAPSHOT": dataset_snapshot,
        }
        with open(f'{cfg.export_policy_path}/info' + ".json", "w") as f:
            json.dump(info, f, indent=2)

    _, _ = env.reset()

    # 力矩截断调试：每 N 步打印 env 0 的力矩与限幅，>0 时开启（0=不打印）
    print_torque_debug_interval = 120  # 约 2 秒一次 @60Hz
    step_count = 0

    for i in range(100000*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, critic_obs, obs_high, rews, _ , dones, infos, _ = env.step(actions.detach())

        # 力矩是否被截断：每 N 步都打印一次；有截断则列关节，无则打一行摘要
        if print_torque_debug_interval > 0 and (step_count % print_torque_debug_interval == 0):
            limits = env.torque_limits.cpu().numpy()
            comp = env.computed_torques[0].cpu().numpy()
            actual = env.torques[0].cpu().numpy()
            names = env.dof_names
            clipped = []
            max_ratio, max_j = 0.0, -1
            for j in range(len(names)):
                if limits[j] <= 0:
                    continue
                ratio = abs(actual[j]) / limits[j]
                if ratio > max_ratio:
                    max_ratio, max_j = ratio, j
                is_clip = abs(comp[j] - actual[j]) > 1e-4
                if is_clip or ratio > 0.85:
                    clipped.append((names[j], float(limits[j]), float(comp[j]), float(actual[j]), ratio, "CLIPPED" if is_clip else ""))
            if clipped:
                print(f"\n[torque @ step {step_count}] env 0 力矩接近/触及限幅:")
                for name, lim, c, a, r, tag in clipped:
                    print(f"  {name}: limit={lim:.1f}  computed={c:.1f}  actual={a:.1f}  |actual|/limit={r:.2%}  {tag}")
            else:
                jname = names[max_j] if max_j >= 0 else ""
                print(f"[torque @ step {step_count}] env 0 无截断  max |actual|/limit={max_ratio:.2%} ({jname})")
        step_count += 1
        

if __name__ == '__main__':
    main()
