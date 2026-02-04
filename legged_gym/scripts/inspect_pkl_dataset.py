#!/usr/bin/env python3
"""
解析 AdaMimic 源动作数据 .pkl 文件（retarget 前的人体/SMPL 格式）。
字段通常包含: poses (SMPL 轴角), betas, gender, trans, mocap_framerate 等。
用法:
  python legged_gym/scripts/inspect_pkl_dataset.py [path/to/data.pkl]
  python legged_gym/scripts/inspect_pkl_dataset.py legged_gym/resources/dataset/g1_dof27_data/badminton_hit/data.pkl
若不传路径，默认使用 badminton_hit/data.pkl。
"""

import os
import sys
import argparse
import pickle
import numpy as np


def _describe(obj, indent=0, max_sample=5):
    """递归描述对象：类型、形状、前几个值（若为数组）。"""
    prefix = "  " * indent
    if isinstance(obj, np.ndarray):
        shape = obj.shape
        dtype = str(obj.dtype)
        print(f"{prefix}ndarray shape={shape}, dtype={dtype}")
        if obj.size > 0 and obj.size <= 12:
            print(f"{prefix}  value = {obj.tolist()}")
        elif obj.size > 12:
            flat = obj.flatten()
            sample = flat[:max_sample].tolist()
            print(f"{prefix}  sample (first {max_sample}) = {sample}")
        return
    if isinstance(obj, dict):
        print(f"{prefix}dict with {len(obj)} keys: {list(obj.keys())}")
        for k, v in list(obj.items())[:20]:
            print(f"{prefix}  [{k!r}]")
            _describe(v, indent + 2, max_sample)
        if len(obj) > 20:
            print(f"{prefix}  ... and {len(obj) - 20} more keys")
        return
    if isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} len={len(obj)}")
        if len(obj) > 0 and len(obj) <= 5:
            for i, x in enumerate(obj):
                print(f"{prefix}  [{i}]")
                _describe(x, indent + 2, max_sample)
        elif len(obj) > 5:
            print(f"{prefix}  [0]")
            _describe(obj[0], indent + 2, max_sample)
            print(f"{prefix}  ... and {len(obj) - 1} more items")
        return
    print(f"{prefix}{type(obj).__name__} = {obj!r}")


def inspect_pkl_file(pkl_path):
    """加载单个 .pkl 文件并打印所有字段描述。"""
    print(f"\n{'='*60}")
    print(f"File: {pkl_path}")
    print("=" * 60)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        print(f"Top-level type: {type(data).__name__}")
        _describe(data, indent=0)
        return

    print(f"Top-level keys: {list(data.keys())}")
    for clip_name, clip in data.items():
        print(f"\n--- clip: {clip_name!r} ---")
        if not isinstance(clip, dict):
            _describe(clip, indent=0)
            continue
        # 常见 SMPL/人体格式字段
        for key in ["poses", "betas", "gender", "trans", "mocap_framerate", "fps"]:
            if key in clip:
                v = clip[key]
                if isinstance(v, np.ndarray):
                    print(f"  {key}: shape={v.shape}, dtype={v.dtype}")
                    if v.size > 0 and v.size <= 8:
                        print(f"    value = {v.tolist()}")
                    elif v.ndim >= 1 and v.shape[0] > 0:
                        print(f"    sample [0] = {v.flat[:min(6, v.size)].tolist()}")
                else:
                    print(f"  {key}: {type(v).__name__} = {v!r}")
        # 若有其他键也一并列出
        other = [k for k in clip.keys() if k not in ["poses", "betas", "gender", "trans", "mocap_framerate", "fps"]]
        if other:
            print(f"  other keys: {other}")
            for k in other[:10]:
                _describe(clip[k], indent=2)
        # 时长摘要
        if "poses" in clip and "mocap_framerate" in clip:
            poses = clip["poses"]
            fps = clip["mocap_framerate"]
            if hasattr(poses, "shape") and poses.shape[0] > 0:
                n_frames = poses.shape[0]
                duration = (n_frames - 1) / fps if fps > 0 else 0
                print(f"  summary: num_frames={n_frames}, framerate={fps}, duration≈{duration:.2f}s")
        elif "poses" in clip and "fps" in clip:
            poses = clip["poses"]
            fps = clip["fps"]
            if hasattr(poses, "shape") and poses.shape[0] > 0:
                n_frames = poses.shape[0]
                duration = (n_frames - 1) / fps if fps > 0 else 0
                print(f"  summary: num_frames={n_frames}, fps={fps}, duration≈{duration:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Inspect AdaMimic source motion .pkl (SMPL/human format).")
    parser.add_argument(
        "path",
        nargs="?",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "resources", "dataset", "g1_dof27_data", "badminton_hit", "data.pkl",
        ),
        help="Path to .pkl file (default: legged_gym/resources/dataset/g1_dof27_data/badminton_hit/data.pkl)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If path is a directory, inspect all .pkl files in it.",
    )
    args = parser.parse_args()

    path = args.path
    if not os.path.isabs(path):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.normpath(os.path.join(repo_root, path))

    if os.path.isfile(path):
        if not path.endswith(".pkl"):
            print("Warning: file is not .pkl", file=sys.stderr)
        inspect_pkl_file(path)
        return
    if os.path.isdir(path):
        pkl_files = [f for f in os.listdir(path) if f.endswith(".pkl")]
        if not pkl_files:
            print(f"No .pkl files in {path}", file=sys.stderr)
            sys.exit(1)
        if args.all:
            for f in sorted(pkl_files):
                inspect_pkl_file(os.path.join(path, f))
        else:
            inspect_pkl_file(os.path.join(path, pkl_files[0]))
            if len(pkl_files) > 1:
                print(f"\n(Use --all to inspect all {len(pkl_files)} .pkl files in this folder.)")
        return
    print(f"Error: not found: {path}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
