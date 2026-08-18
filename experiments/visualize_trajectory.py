#!/usr/bin/env python
"""在 meshcat 中回放 traj_data/csv_data 下的激励轨迹。

加载 biped_s49_left_arm 的 URDF（含 mesh），把 CSV 里的 q1~q7 逐帧驱动
zarm_l1~l7 关节，在浏览器里 3D 回放激励轨迹。

用法:
    # 回放最新生成的轨迹
    python experiments/visualize_trajectory.py

    # 指定 CSV 文件
    python experiments/visualize_trajectory.py --csv traj_data/csv_data/xxx.csv

    # 指定机器人
    python experiments/visualize_trajectory.py --robot biped_s49_left_arm

运行后终端会打印 meshcat 的 URL，浏览器打开即可看到回放。
按 Ctrl+C 退出。
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import pandas as pd
from pinocchio.visualize import MeshcatVisualizer

from system_identification.utils import find_path


def load_trajectory(csv_path):
    """加载激励轨迹 CSV。

    Returns:
        t, q (shape=(N,7)), v, a
    """
    df = pd.read_csv(csv_path)
    t = df["t"].values
    q = df[["q1", "q2", "q3", "q4", "q5", "q6", "q7"]].values
    v = df[["v1", "v2", "v3", "v4", "v5", "v6", "v7"]].values
    a = df[["a1", "a2", "a3", "a4", "a5", "a6", "a7"]].values
    return t, q, v, a


def build_visualizer(robot_name):
    """加载 URDF + mesh，构建 meshcat 可视化器。

    URDF 里 mesh 路径是 package://kuavo_assets/models/biped_s49/meshes/xxx.STL，
    实际在 robot_description/biped_s49_description/meshes/ 下。
    通过创建符号链接让 pinocchio 按 package:// 协议找到 mesh。
    """
    import os
    import tempfile

    urdf_path = find_path(f"{robot_name}.urdf", "./robot_description")
    package_root = Path("./robot_description").resolve()

    # 创建临时目录，建立 kuavo_assets/models/biped_s49 -> biped_s49_description 的符号链接
    # 这样 pinocchio 解析 package://kuavo_assets/models/biped_s49/meshes/xxx.STL 时
    # 会找到 tmpdir/kuavo_assets/models/biped_s49/meshes/xxx.STL
    # -> 实际指向 robot_description/biped_s49_description/meshes/xxx.STL
    tmpdir = Path(tempfile.mkdtemp(prefix="pinocchio_pkg_"))
    kuavo_dir = tmpdir / "kuavo_assets" / "models" / "biped_s49"
    kuavo_dir.parent.mkdir(parents=True, exist_ok=True)
    kuavo_dir.symlink_to(package_root / "biped_s49_description")

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_path, package_dirs=[str(tmpdir)]
    )

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(loadModel=True)
    return viz, model


def find_latest_csv(csv_dir):
    """找 csv_dir 下最新的 condFriction_*.csv 文件。"""
    csv_dir = Path(csv_dir)
    csvs = sorted(csv_dir.glob("condFriction_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"{csv_dir} 下没有 condFriction_*.csv 文件")
    return csvs[-1]


def main():
    parser = argparse.ArgumentParser(description="3D 回放激励轨迹")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="CSV 文件路径，不指定则用 traj_data/csv_data 下最新的",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="./traj_data/csv_data",
        help="CSV 目录（--csv 未指定时从此目录找最新文件）",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="biped_s49_left_arm",
        help="机器人名（对应 robot_description 下的 URDF）",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="回放速度倍率（1.0=实时，2.0=2倍速）",
    )
    parser.add_argument(
        "--render_hz",
        type=float,
        default=50.0,
        help="渲染帧率（Hz），默认 50。CSV 原始 500Hz 太高，meshcat display 跟不上",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="循环回放",
    )
    args = parser.parse_args()

    # 确定 CSV 文件
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = find_latest_csv(args.csv_dir)
    print(f"[info] 回放: {csv_path}")

    # 加载轨迹
    t, q, v, a = load_trajectory(csv_path)
    print(f"[info] 轨迹: {len(t)} 帧, 时长 {t[-1]:.2f}s, 7 关节")

    # 构建可视化器（initViewer 会自动启动 meshcat 服务并打印 URL）
    viz, model = build_visualizer(args.robot)

    # 等待用户打开浏览器
    print("[info] meshcat 已启动，请在浏览器打开上面的 URL")
    print(f"[info] 回放 {len(t)} 帧，速度 {args.speed}x，渲染 {args.render_hz}Hz")
    if args.loop:
        print("[info] 循环模式，按 Ctrl+C 退出")
    time.sleep(2)

    # 创建 data
    data = model.createData()

    # 按 render_hz 降采样选帧索引
    # CSV 原始时间轴 t，渲染间隔 dt_render = 1/render_hz（按回放速度调整）
    dt_render = 1.0 / args.render_hz / args.speed
    # 选出每 dt_render 对应的帧索引（在原始时间轴上最近邻）
    render_indices = np.searchsorted(t, t[0] + np.arange(0, t[-1] - t[0], dt_render))
    render_indices = np.clip(render_indices, 0, len(t) - 1)
    n_render = len(render_indices)
    print(f"[info] 渲染 {n_render} 帧（原始 {len(t)} 帧降采样）")

    # 回放：睡眠时间 = 目标间隔 - display 耗时（不小于 0）
    # 期望总时间 = 最后一帧时间戳 - 第一帧时间戳
    expected_total = t[render_indices[-1]] - t[render_indices[0]]
    try:
        while True:
            t_round_start = time.perf_counter()
            for k, i in enumerate(render_indices):
                t_disp_start = time.perf_counter()
                viz.display(q[i])
                t_disp_end = time.perf_counter()
                disp_cost = t_disp_end - t_disp_start
                # 睡眠补偿：目标间隔减去 display 耗时，下限 0
                sleep_time = max(0.0, dt_render - disp_cost)
                if k < n_render - 1:
                    time.sleep(sleep_time)
            t_round_end = time.perf_counter()
            actual_total = t_round_end - t_round_start
            drift = actual_total - expected_total
            print(
                f"[info] 本轮回放完成: 实际 {actual_total:.3f}s / 期望 {expected_total:.3f}s"
                f" / 偏差 {drift*1000:+.1f}ms ({drift/expected_total*100:+.2f}%)"
            )
            if not args.loop:
                break
            print("[info] 回放结束，5 秒后重新开始...")
            time.sleep(5.0)
    except KeyboardInterrupt:
        print("\n[info] 已退出")


if __name__ == "__main__":
    main()
