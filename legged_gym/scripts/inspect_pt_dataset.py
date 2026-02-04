#!/usr/bin/env python3
"""
解析 AdaMimic 动作数据 .pt 文件中的字段，打印结构、形状和样例值。
用法:
  python legged_gym/scripts/inspect_pt_dataset.py [path/to/output]
  python legged_gym/scripts/inspect_pt_dataset.py legged_gym/resources/dataset/g1_dof27_data/badminton_hit/output
若不传路径，默认使用 badminton_hit/output。
"""

import os
import sys
import argparse
import torch
import numpy as np


def _describe(obj, indent=0, max_sample=3):
    """递归描述对象：类型、形状、前几个值（若为数组）。"""
    prefix = "  " * indent
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        arr = obj.numpy() if isinstance(obj, torch.Tensor) else obj
        shape = arr.shape
        dtype = str(arr.dtype)
        print(f"{prefix}shape={shape}, dtype={dtype}")
        if arr.size > 0 and arr.size <= 12:
            print(f"{prefix}  value = {arr.tolist()}")
        elif arr.size > 12:
            flat = arr.flatten()
            sample = flat[:max_sample].tolist()
            print(f"{prefix}  sample (first {max_sample}) = {sample}")
        return
    if isinstance(obj, dict):
        print(f"{prefix}dict with {len(obj)} keys: {list(obj.keys())[:10]}{'...' if len(obj) > 10 else ''}")
        for k, v in list(obj.items())[:15]:
            print(f"{prefix}  [{k!r}]")
            _describe(v, indent + 2, max_sample)
        if len(obj) > 15:
            print(f"{prefix}  ... and {len(obj) - 15} more keys")
        return
    if isinstance(obj, (list, tuple)):
        print(f"{prefix}len={len(obj)}")
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


def inspect_pt_file(pt_path):
    """加载单个 .pt 文件并打印所有字段描述。"""
    print(f"\n{'='*60}")
    print(f"File: {pt_path}")
    print("=" * 60)
    data = torch.load(pt_path, map_location="cpu")
    if not isinstance(data, dict):
        print(f"Top-level type: {type(data)}")
        _describe(data, indent=0)
        return
    print(f"Top-level keys: {list(data.keys())}")
    for key in sorted(data.keys()):
        print(f"\n--- {key} ---")
        _describe(data[key], indent=0)
    # 若有 framerate，顺带算一下时长
    if "framerate" in data and "base_position" in data:
        base = data["base_position"]
        n_frames = base.shape[0] if hasattr(base, "shape") else len(base)
        fps = data["framerate"]
        if isinstance(fps, (torch.Tensor, np.ndarray)):
            fps = float(fps.flat[0])
        duration = (n_frames - 1) / fps if fps > 0 else 0
        print(f"\n--- summary ---")
        print(f"  num_frames = {n_frames}, framerate = {fps}, duration ≈ {duration:.2f} s")


def main():
    parser = argparse.ArgumentParser(description="Inspect AdaMimic motion .pt dataset files.")
    parser.add_argument(
        "folder",
        nargs="?",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "resources", "dataset", "g1_dof27_data", "badminton_hit", "output",
        ),
        help="Folder containing .pt files (default: legged_gym/resources/dataset/g1_dof27_data/badminton_hit/output)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Inspect all .pt files in folder; default is only the first one.",
    )
    args = parser.parse_args()

    folder = args.folder
    if not os.path.isabs(folder):
        # 以项目根为基准
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        folder = os.path.normpath(os.path.join(repo_root, folder))
    if not os.path.isdir(folder):
        print(f"Error: folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    pt_files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    if not pt_files:
        print(f"No .pt files in {folder}", file=sys.stderr)
        sys.exit(1)

    if args.all:
        for f in sorted(pt_files):
            inspect_pt_file(os.path.join(folder, f))
    else:
        inspect_pt_file(os.path.join(folder, pt_files[0]))
        if len(pt_files) > 1:
            print(f"\n(Use --all to inspect all {len(pt_files)} .pt files in this folder.)")


if __name__ == "__main__":
    main()
