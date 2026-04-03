# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, AttrDict
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
import shutil
import json


def _copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def snapshot_training_configs(cfg, choices):
    run_dir = Path(cfg.run_dir).resolve()
    snapshot_dir = run_dir / "config_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Hydra may change cwd to run_dir; always resolve configs from repo root.
    config_root = Path(LEGGED_GYM_ROOT_DIR) / "legged_gym" / "configs"
    robot_name = str(choices.get("robot", "")).strip()
    algorithm_name = str(choices.get("algorithm", "")).strip()
    dataset_name = str(choices.get("dataset", "")).strip()

    if not robot_name or not algorithm_name or not dataset_name:
        raise KeyError(
            f"Cannot resolve config choices from Hydra runtime: robot={robot_name}, "
            f"algorithm={algorithm_name}, dataset={dataset_name}"
        )

    robot_src = (config_root / "robot" / f"{robot_name}.yaml").resolve()
    algorithm_src = (config_root / "algorithm" / f"{algorithm_name}.yaml").resolve()
    dataset_src = (config_root / "dataset" / f"{dataset_name}.yaml").resolve()

    copied = {
        "robot.yaml": _copy_if_exists(robot_src, snapshot_dir / "robot.yaml"),
        "algorithm.yaml": _copy_if_exists(algorithm_src, snapshot_dir / "algorithm.yaml"),
        "dataset.yaml": _copy_if_exists(dataset_src, snapshot_dir / "dataset.yaml"),
    }
    missing = [name for name, ok in copied.items() if not ok]
    if missing:
        raise FileNotFoundError(
            "Failed to snapshot config yaml(s): "
            + ", ".join(missing)
            + ". Resolved paths => "
            + f"robot: {robot_src}, algorithm: {algorithm_src}, dataset: {dataset_src}"
        )

    # Keep a fully resolved run config for exact reproducibility.
    (snapshot_dir / "train_resolved.json").write_text(
        json.dumps(dict(cfg), indent=2, ensure_ascii=True)
    )

    # Record original source paths for traceability.
    source_map = {
        "hydra_choices": {
            "robot": robot_name,
            "algorithm": algorithm_name,
            "dataset": dataset_name,
        },
        "robot_config": str(robot_src),
        "algorithm_config": str(algorithm_src),
        "dataset_config": str(dataset_src),
    }
    (snapshot_dir / "source_paths.json").write_text(
        json.dumps(source_map, indent=2, ensure_ascii=True)
    )


@hydra.main(config_path="../configs", config_name="train", version_base="1.1")
def main(cfg):
    hydra_runtime = HydraConfig.get().runtime
    cfg = AttrDict(OmegaConf.to_container(cfg, resolve=True))
    cfg.run_dir = hydra_runtime.output_dir
    snapshot_training_configs(cfg, hydra_runtime.choices)
    env, env_cfg = task_registry.make_env_hydra(cfgs=cfg)

    ppo_runner, train_cfg = task_registry.make_alg_runner_hydra(env=env, env_cfg=env_cfg, cfgs=cfg)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    main()
