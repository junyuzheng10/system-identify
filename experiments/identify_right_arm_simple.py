#!/usr/bin/env python
"""Minimal system identification for marvinM6 right arm.

Uses pinocchio's analytical joint-torque regressor (minimal parameter set,
10 params/joint: mass, com_x, com_y, com_z, Ixx, Iyy, Izz, Ixy, Ixz, Iyz),
no friction, no armature, no regularization. Plain unweighted least squares.

Usage:
    python experiments/identify_right_arm_simple.py [--data_dir DIR] [--acc_source sensors|csv]

Outputs:
    fig1: acceleration used vs csv acceleration
    fig2: predicted vs measured torque
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

import pinocchio as pin
from system_identification.utils import find_path, savgol_filter_acceleration


def load_data(data_dir):
    data_dir = Path(data_dir)
    df_q = pd.read_csv(data_dir / "sensors_joint_q.csv")
    df_v = pd.read_csv(data_dir / "sensors_joint_v.csv")
    df_tau = pd.read_csv(data_dir / "sensors_joint_torque.csv")
    df_cmd = pd.read_csv(data_dir / "csv_trajectory_a.csv")

    t = df_q["time"].values
    q = df_q.iloc[:, 1:8].values
    v = df_v.iloc[:, 1:8].values
    tau = df_tau.iloc[:, 1:8].values

    # Central difference acceleration + light Savgol
    dt = np.diff(t)
    a_fd = np.zeros_like(v)
    a_fd[1:-1] = (v[2:] - v[:-2]) / (dt[1:] + dt[:-1])[:, None]
    a_fd = savgol_filter_acceleration(a_fd, window_length=21, polyorder=3)

    # CSV command acceleration
    t_cmd = df_cmd["time"].values
    a_cmd_raw = df_cmd.iloc[:, 1:8].values
    a_cmd = np.column_stack(
        [np.interp(t, t_cmd, a_cmd_raw[:, j]) for j in range(a_cmd_raw.shape[1])]
    )

    # Measured sensor acceleration (optional)
    a_meas = None
    a_meas_file = data_dir / "sensors_joint_a.csv"
    if a_meas_file.exists():
        df_a = pd.read_csv(a_meas_file)
        t_a = df_a["time"].values
        a_raw = df_a.iloc[:, 1:8].values
        if len(t_a) == len(t) and np.allclose(t_a, t):
            a_meas = a_raw
        else:
            a_meas = np.column_stack(
                [np.interp(t, t_a, a_raw[:, j]) for j in range(a_raw.shape[1])]
            )

    return t, q, v, a_fd, tau, a_cmd, a_meas


def main():
    parser = argparse.ArgumentParser(description="Minimal right-arm inertia identification")
    parser.add_argument("--data_dir", type=str, default="./replay_right_arm_101_log_data")
    parser.add_argument("--robot", type=str, default="marvinM6_right")
    parser.add_argument("--acc_source", type=str, default="sensors", choices=["sensors", "csv"])
    parser.add_argument("--trim", type=int, default=1)
    parser.add_argument("--savgol_window", type=int, default=21, help="Savgol window for sensor acceleration filter")
    parser.add_argument("--savgol_poly", type=int, default=5)
    args = parser.parse_args()

    # Load
    t, q, v, a_fd, tau_meas, a_cmd, a_meas = load_data(args.data_dir)
    n = args.trim
    t = t[n:-n]; q = q[n:-n]; v = v[n:-n]; a_fd = a_fd[n:-n]; tau_meas = tau_meas[n:-n]; a_cmd = a_cmd[n:-n]
    if a_meas is not None:
        a_meas = a_meas[n:-n]
    N, nj = q.shape
    logger.info(f"Loaded {N} samples, {nj} joints from {args.data_dir}")

    # Select acceleration source
    if args.acc_source == "csv":
        a = a_cmd
        a_label = "CSV command"
    else:
        if a_meas is None:
            logger.warning("No sensor acceleration, falling back to central diff + Savgol")
            a = a_fd
            a_label = "Central diff + Savgol (fallback)"
        else:
            w = args.savgol_window
            if w % 2 == 0: w += 1
            w = min(w, N - 1 if (N - 1) % 2 == 1 else N - 2)
            p = min(args.savgol_poly, w - 1)
            a = savgol_filter_acceleration(a_meas, window_length=w, polyorder=p)
            a_label = f"Sensor measured (Savgol w={w} p={p})"
    logger.info(f"Acceleration source: {a_label}")

    # Build pinocchio model
    urdf_file = find_path(f"{args.robot}.urdf", "./robot_description")
    model = pin.buildModelFromUrdf(urdf_file)
    data = model.createData()

    # Build regressor: Y(N*nj, 10*nj) using analytical minimal parameter set
    logger.info("Building regressor...")
    Y = pin.computeJointTorqueRegressor(model, data, q[0], v[0], a[0])
    n_params = Y.shape[1]
    Y_all = np.zeros((N * nj, n_params))
    for i in range(N):
        Y_all[i * nj:(i + 1) * nj] = pin.computeJointTorqueRegressor(model, data, q[i], v[i], a[i])
    tau_flat = tau_meas.reshape(-1)
    logger.info(f"Regressor shape: {Y_all.shape}, condition: {np.linalg.cond(Y_all):.2f}")

    # Plain least squares
    phi, residuals, rank, sv = np.linalg.lstsq(Y_all, tau_flat, rcond=None)
    logger.info(f"Rank: {rank}, singular values: [{sv.min():.4e}, {sv.max():.4e}]")
    logger.info(f"phi shape: {phi.shape}")

    # Predict
    tau_pred = (Y_all @ phi).reshape(N, nj)

    # Print per-joint RMS
    logger.info("=" * 70)
    logger.info(f"{'Joint':>6} | {'RMS meas':>10} | {'Ident err':>12} {'ratio':>7}")
    logger.info("-" * 70)
    for j in range(nj):
        rms_meas = np.sqrt(np.mean(tau_meas[:, j] ** 2))
        rms_err = np.sqrt(np.mean((tau_meas[:, j] - tau_pred[:, j]) ** 2))
        logger.info(f"J{j+1:>4}  | {rms_meas:10.4f} | {rms_err:12.6f} {rms_err/rms_meas:7.4f}")
    logger.info("=" * 70)

    # Print mass/CoM per joint
    for j in range(nj):
        pi = phi[j * 10:(j + 1) * 10]
        mass = pi[0]
        com = pi[1:4] / mass if mass > 1e-10 else np.zeros(3)
        logger.info(f"J{j+1}: Mass={mass:.6f}, CoM=[{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}]")

    # fig1: acceleration used vs csv
    fig1, axes1 = plt.subplots(nj, 1, sharex=True, figsize=(12, 2 * nj))
    if nj == 1: axes1 = [axes1]
    for j in range(nj):
        ax = axes1[j]
        ax.plot(t, a_cmd[:, j], color='C0', linewidth=1.5, label="CSV command" if j == 0 else None)
        ax.plot(t, a[:, j], color='C3', linewidth=1.0, alpha=0.7, label=a_label if j == 0 else None)
        ax.set_ylabel(f"J{j+1}\n(rad/s²)")
    axes1[0].legend(loc='upper right', fontsize=8)
    axes1[-1].set_xlabel("time (s)")
    fig1.suptitle(f"Acceleration: CSV command (blue) vs {a_label} (red)")
    plt.tight_layout()

    # fig2: predicted vs measured torque
    fig2, axes2 = plt.subplots(nj, 1, sharex=True, figsize=(12, 2 * nj))
    if nj == 1: axes2 = [axes2]
    for j in range(nj):
        ax = axes2[j]
        ax.plot(t, tau_meas[:, j], 'b-', linewidth=1, label='Measured' if j == 0 else None)
        ax.plot(t, tau_pred[:, j], 'r--', linewidth=1, label='Identified' if j == 0 else None)
        ax.set_ylabel(f"J{j+1}\n(N·m)")
    axes2[0].legend(loc='upper right', fontsize=8)
    axes2[-1].set_xlabel("time (s)")
    fig2.suptitle("Predicted vs Measured Torque (inertia-only, plain LS)")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
