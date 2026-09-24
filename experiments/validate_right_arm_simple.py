#!/usr/bin/env python
"""Validate identified params from NPZ (simple or friction) by predicting torque.

Loads an NPZ produced by identify_right_arm_simple.py or
identify_right_arm_friction_simple.py, rebuilds the regressor on the specified
data, and compares predicted vs measured torque.

Usage:
    python experiments/validate_right_arm_simple.py --params NPZ [--data_dir DIR]
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

import pinocchio as pin
from system_identification.utils import find_path, savgol_filter_acceleration


def load_data(data_dir, trim_head=1, trim_tail=1, torque_prefer=None):
    """Load and align data. Tries preferred torque file first, then falls back."""
    data_dir = Path(data_dir)
    df_q = pd.read_csv(data_dir / "sensors_joint_q.csv")
    df_v = pd.read_csv(data_dir / "sensors_joint_v.csv")

    # Determine torque file priority
    if torque_prefer:
        candidates = [torque_prefer, "sensor_actual_torque.csv", "sensors_joint_torque.csv"]
    else:
        candidates = ["sensor_actual_torque.csv", "sensors_joint_torque.csv"]
    tau_file = None
    for name in candidates:
        if (data_dir / name).exists():
            tau_file = name
            break
    if tau_file is None:
        raise FileNotFoundError("No torque CSV found")
    df_tau = pd.read_csv(data_dir / tau_file)
    df_cmd = pd.read_csv(data_dir / "csv_trajectory_a.csv")

    t_ref = df_q["time"].values
    q = df_q.iloc[:, 1:1 + 7].values
    nj = q.shape[1]

    def align(t_src, data):
        if len(t_src) == len(t_ref) and np.allclose(t_src, t_ref):
            return data
        return np.column_stack([np.interp(t_ref, t_src, data[:, j]) for j in range(nj)])

    v = align(df_v["time"].values, df_v.iloc[:, 1:1 + nj].values)
    tau_meas = align(df_tau["time"].values, df_tau.iloc[:, 1:1 + nj].values)
    a_cmd = align(df_cmd["time"].values, df_cmd.iloc[:, 1:1 + nj].values)

    a_meas = None
    a_file = data_dir / "sensors_joint_a.csv"
    if a_file.exists():
        df_a = pd.read_csv(a_file)
        a_meas = align(df_a["time"].values, df_a.iloc[:, 1:1 + nj].values)

    logger.info(f"Loaded {len(t_ref)} samples, {nj} joints, torque={tau_file}")

    th, tt = trim_head, trim_tail
    if th > 0 or tt > 0:
        sl = slice(th, -tt if tt > 0 else None)
        t_ref = t_ref[sl]; q = q[sl]; v = v[sl]; tau_meas = tau_meas[sl]; a_cmd = a_cmd[sl]
        if a_meas is not None:
            a_meas = a_meas[sl]

    return t_ref, q, v, tau_meas, a_cmd, a_meas


def feat_block(feat, N, nj):
    """Reshape (N,nj) feature into (N*nj, nj) block-diagonal."""
    Y = np.zeros((N * nj, nj))
    for i in range(N):
        Y[i * nj:(i + 1) * nj] = np.diag(feat[i])
    return Y


def main():
    parser = argparse.ArgumentParser(description="Validate identified params from NPZ")
    parser.add_argument("--params", type=str, required=True, help="Path to NPZ file")
    parser.add_argument("--data_dir", type=str, default="./replay_right_arm_101_log_data")
    parser.add_argument("--acc_source", type=str, default="csv", choices=["csv", "sensors"],
                        help="Acceleration source (default: use NPZ stored value)")
    parser.add_argument("--trim_head", type=int, default=1)
    parser.add_argument("--trim_tail", type=int, default=1)
    parser.add_argument("--savgol_window", type=int, default=21)
    parser.add_argument("--savgol_poly", type=int, default=5)
    args = parser.parse_args()

    # Load NPZ
    data = np.load(args.params, allow_pickle=True)
    robot = str(data["robot"])
    nj = int(data["njoints"])
    vcoul = float(data["vcoul"]) if "vcoul" in data else 0.002
    npz_acc = str(data["acc_source"]) if "acc_source" in data else "csv"
    acc_source = args.acc_source if args.acc_source is not None else npz_acc
    is_friction_only = bool(data["inertia_frozen"]) if "inertia_frozen" in data else False

    logger.info(f"NPZ: {args.params}")
    logger.info(f"  robot={robot}, njoints={nj}, vcoul={vcoul:.6f}, acc_source={acc_source}")
    logger.info(f"  model type: {'friction-only (URDF frozen)'}" if is_friction_only else "  model type: joint (inertia+friction)")

    # Load data — match torque file to identification script
    torque_prefer = "sensor_actual_torque.csv" if is_friction_only else "sensors_joint_torque.csv"
    t, q, v, tau_meas, a_cmd, a_meas = load_data(
        args.data_dir, trim_head=args.trim_head, trim_tail=args.trim_tail,
        torque_prefer=torque_prefer)
    N = q.shape[0]

    # Select acceleration
    if acc_source == "csv":
        a = a_cmd
        a_label = "CSV command"
    else:
        if a_meas is None:
            logger.warning("No sensor acceleration, falling back to central diff + Savgol")
            dt = np.diff(t, prepend=t[0])
            a_fd = np.zeros_like(v)
            a_fd[1:-1] = (v[2:] - v[:-2]) / (dt[1:] + dt[:-1])[:, None]
            a = savgol_filter_acceleration(a_fd, window_length=args.savgol_window, polyorder=args.savgol_poly)
        else:
            a = a_meas
        a_label = "Sensor measured"

    # Build pinocchio model
    urdf_file = find_path(f"{robot}.urdf", "./robot_description")
    model = pin.buildModelFromUrdf(urdf_file)
    pin_data = model.createData()

    # Build friction/armature/offset regressors (shared by both model types)
    tanh_v = np.tanh(v / vcoul)
    Y_friction = np.hstack([feat_block(tanh_v, N, nj), feat_block(v, N, nj)])
    Y_armature = feat_block(a, N, nj)
    Y_offset = feat_block(np.ones_like(v), N, nj)
    Y_fric_block = np.hstack([Y_friction, Y_armature, Y_offset])

    if is_friction_only:
        # Friction-only: RNEA (frozen URDF) + friction model
        model.friction[:] = 0.0
        model.damping[:] = 0.0
        tau_rnea = np.zeros((N, nj))
        for i in range(N):
            tau_rnea[i] = pin.rnea(model, pin_data, q[i], v[i], a[i])
        phi = data["phi_friction"]
        tau_pred = tau_rnea + (Y_fric_block @ phi).reshape(N, nj)
        tau_nominal = tau_rnea
        nominal_label = "RNEA (URDF)"
    else:
        # Joint: inertia regressor + friction model
        Y_inertia = np.zeros((N * nj, 10 * nj))
        for i in range(N):
            Y_inertia[i * nj:(i + 1) * nj] = pin.computeJointTorqueRegressor(model, pin_data, q[i], v[i], a[i])
        Y_all = np.hstack([Y_inertia, Y_fric_block])
        phi = data["phi"]
        tau_pred = (Y_all @ phi).reshape(N, nj)
        # Nominal: URDF RNEA with default friction
        nom_model = pin.buildModelFromUrdf(urdf_file)
        nom_data = nom_model.createData()
        tau_nominal = np.zeros((N, nj))
        for i in range(N):
            tau_nominal[i] = pin.rnea(nom_model, nom_data, q[i], v[i], a[i])
        nominal_label = "Nominal URDF"

    # Print comparison
    logger.info("=" * 75)
    logger.info(f"{'Joint':>6} | {'RMS meas':>10} | {'Nominal err':>12} {'ratio':>7} | {'Pred err':>12} {'ratio':>7}")
    logger.info("-" * 75)
    for j in range(nj):
        rms_meas = np.sqrt(np.mean(tau_meas[:, j] ** 2))
        rms_nom = np.sqrt(np.mean((tau_meas[:, j] - tau_nominal[:, j]) ** 2))
        rms_pred = np.sqrt(np.mean((tau_meas[:, j] - tau_pred[:, j]) ** 2))
        logger.info(f"J{j+1:>4}  | {rms_meas:10.4f} | {rms_nom:12.6f} {rms_nom/rms_meas:7.4f} | "
                    f"{rms_pred:12.6f} {rms_pred/rms_meas:7.4f}")
    logger.info("=" * 75)

    # Plot: predicted vs measured torque
    fig, axes = plt.subplots(nj, 1, sharex=True, figsize=(12, 2 * nj))
    if nj == 1:
        axes = [axes]
    for j in range(nj):
        ax = axes[j]
        ax.plot(t, tau_meas[:, j], 'b-', linewidth=1, label='Measured')
        ax.plot(t, tau_nominal[:, j], 'g-', linewidth=1, alpha=0.5, label=nominal_label)
        ax.plot(t, tau_pred[:, j], 'r--', linewidth=1, label='Identified')
        ax.set_ylabel(f"J{j+1}\n(N·m)")
        if j == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"VALIDATION: {Path(args.params).name} on {args.data_dir}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
