#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return None

def describe_diff(name, a, b, max_frames=1000):
    a_np = to_numpy(a)
    b_np = to_numpy(b)
    if a_np is None or b_np is None:
        print(f"  [{name}] 非数组类型，跳过对比: {type(a)}, {type(b)}")
        return

    print(f"\n=== [{name}] ===")
    print(f"  old shape: {a_np.shape}")
    print(f"  new shape: {b_np.shape}")

    if a_np.ndim == 0 or b_np.ndim == 0:
        print("  标量，跳过差值统计")
        return

    # 对齐帧数（第 0 维）
    T = min(a_np.shape[0], b_np.shape[0])
    if T == 0:
        print("  空序列，跳过")
        return
    if T > max_frames:
        T = max_frames
        print(f"  只对比前 {T} 帧")

    a_slice = a_np[:T]
    b_slice = b_np[:T]

    # 除了时间外其他维度若不一致，取公共最小维度粗略比较
    if a_slice.shape != b_slice.shape:
        min_shape = tuple(min(sa, sb) for sa, sb in zip(a_slice.shape, b_slice.shape))
        a_slice = a_slice[tuple(slice(0, m) for m in min_shape)]
        b_slice = b_slice[tuple(slice(0, m) for m in min_shape)]
        print(f"  维度不完全一致，按公共 shape 对齐: {min_shape}")

    diff = np.abs(a_slice - b_slice)
    print(f"  mean abs diff: {float(diff.mean()):.6f}")
    print(f"  max  abs diff: {float(diff.max()):.6f}")

    # 对几个关键字段做更细一点的打印
    if name == "joint_position" and a_slice.ndim == 2:
        T_, J = a_slice.shape
        print(f"  关节数 J = {J}")
        # 每个关节一列的平均绝对差
        per_joint = diff.mean(axis=0)
        # 打印前若干个关节的差值
        max_show = min(J, 10)
        for j in range(max_show):
            print(f"    joint {j}: mean abs diff = {float(per_joint[j]):.6f}")
        if J > max_show:
            print("    ...(更多关节省略)")
    elif name == "base_position" and a_slice.ndim == 2 and a_slice.shape[1] == 3:
        print(f"  base_position 每个轴 mean abs diff: x={diff[:,0].mean():.6f}, "
              f"y={diff[:,1].mean():.6f}, z={diff[:,2].mean():.6f}")
    elif name == "base_pose" and a_slice.ndim == 2 and a_slice.shape[1] == 3:
        print(f"  base_pose 每个轴 mean abs diff: r={diff[:,0].mean():.6f}, "
              f"p={diff[:,1].mean():.6f}, y={diff[:,2].mean():.6f}")
    elif name == "link_position" and a_slice.ndim == 3:
        # [T, N_body, 3]
        print(f"  link_position: mean abs diff (xyz 平均) = {float(diff.mean()):.6f}")
        # 可以大致看躯干/头部等 body 的差距，先只给总体的


def main():
    parser = argparse.ArgumentParser(
        description="Compare two AdaMimic motion .pt dataset files."
    )
    parser.add_argument("old_pt", help="原始 .pt 文件路径")
    parser.add_argument("new_pt", help="新处理 .pt 文件路径")
    parser.add_argument(
        "--max-frames", type=int, default=1000,
        help="最多对比多少帧（默认 1000）"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.old_pt):
        print(f"旧文件不存在: {args.old_pt}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.new_pt):
        print(f"新文件不存在: {args.new_pt}", file=sys.stderr)
        sys.exit(1)

    print(f"加载 old: {args.old_pt}")
    old = torch.load(args.old_pt, map_location="cpu")
    print(f"加载 new: {args.new_pt}")
    new = torch.load(args.new_pt, map_location="cpu")

    if not isinstance(old, dict) or not isinstance(new, dict):
        print("至少有一个 .pt 顶层不是 dict，当前脚本只支持 dict 结构", file=sys.stderr)
        print(f"old type: {type(old)}, new type: {type(new)}")
        sys.exit(1)

    old_keys = set(old.keys())
    new_keys = set(new.keys())
    common_keys = sorted(old_keys & new_keys)

    print("\n====================================")
    print("公共字段:", common_keys)
    print("只在 old 中的字段:", sorted(old_keys - new_keys))
    print("只在 new 中的字段:", sorted(new_keys - old_keys))
    print("====================================\n")

    # 先对关键字段排序，方便阅读
    priority = ["base_position", "base_pose", "joint_position",
                "link_position", "link_orientation",
                "link_velocity", "link_angular_velocity"]
    ordered_keys = [k for k in priority if k in common_keys] + \
                   [k for k in common_keys if k not in priority]

    for k in ordered_keys:
        describe_diff(k, old[k], new[k], max_frames=args.max_frames)


if __name__ == "__main__":
    main()