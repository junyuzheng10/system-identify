#!/usr/bin/env python
"""Compare inverse dynamics torque from pinocchio model vs measured torque.

Loads sensor data (q, v), computes acceleration via central difference + Savitzky-Golay,
runs pinocchio inverse dynamics (RNEA), and compares with measured torques.
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
    """Load sensor data from log_data directory.

    Returns:
        t, q, v, tau_meas
    """
    data_dir = Path(data_dir)

    df_q = pd.read_csv(data_dir / "sensors_joint_q.csv")
    df_v = pd.read_csv(data_dir / "sensors_joint_v.csv")
    df_tau = pd.read_csv(data_dir / "sensors_joint_torque.csv")

    t = df_q["time"].values
    q = df_q.iloc[:, 1:8].values
    v = df_v.iloc[:, 1:8].values
    tau = df_tau.iloc[:, 1:8].values

    return t, q, v, tau


def main():
    parser = argparse.ArgumentParser(description="Inverse dynamics torque comparison")
    parser.add_argument("--robot", type=str, default="biped_s49_left_arm")
    parser.add_argument("--data_dir", type=str, default="./log_data")
    parser.add_argument("--trim", type=int, default=10, help="Frames to trim from start and end")
    parser.add_argument("--window_length", type=int, default=21, help="Savitzky-Golay window length (odd)")
    parser.add_argument("--polyorder", type=int, default=3, help="Savitzky-Golay polynomial order")
    args = parser.parse_args()

    # Load data
    t, q, v, tau_meas = load_data(args.data_dir)
    n_samples_full, njoints = q.shape
    logger.info(f"Loaded {n_samples_full} samples, {njoints} joints")

    # Compute acceleration via central difference of velocity
    dt = np.diff(t)
    a_fd = np.zeros_like(v)
    a_fd[1:-1] = (v[2:] - v[:-2]) / (dt[1:] + dt[:-1])[:, None]
    # Apply Savitzky-Golay filter
    a_fd = savgol_filter_acceleration(a_fd, window_length=args.window_length, polyorder=args.polyorder)

    # Trim boundary frames
    n = args.trim
    t = t[n:-n]
    q = q[n:-n]
    v = v[n:-n]
    a_fd = a_fd[n:-n]
    tau_meas = tau_meas[n:-n]
    logger.info(f"Trimmed {n} frames from each end, remaining: {t.shape[0]} samples")

    # Load pinocchio model
    urdf_file = find_path(f"{args.robot}.urdf", "./robot_description")
    model = pin.buildModelFromUrdf(urdf_file)
    data = model.createData()
    logger.info(f"Loaded pinocchio model: {args.robot}, njoints={model.njoints - 1}")

    # Friction parameters from URDF model
    coulomb_gain = np.array(model.friction)
    viscous_gain = np.array(model.damping)
    armature = 0.05
    eps = 0.0001

    # Run inverse dynamics + friction compensation for each time step
    # Use pinocchio's dedicated functions (internally consistent with RNEA):
    #   M(q)  via crba
    #   C(q,v) via computeCoriolisMatrix
    #   g(q)  via computeGeneralizedGravity
    #   τ = M·a + C·v + g + friction + armature
    tau_pred = np.zeros_like(tau_meas)
    for i in range(t.shape[0]):
        q_i, v_i, a_i = q[i], v[i], a_fd[i]

        # 1. Inertia matrix M(q) — already symmetric, no manual symmetrization needed
        pin.crba(model, data, q_i)
        M = np.array(data.M)

        # 2. Coriolis matrix C(q, qdot)
        pin.computeCoriolisMatrix(model, data, q_i, v_i)
        C = np.array(data.C)

        # 3. Gravity g(q)
        pin.computeGeneralizedGravity(model, data, q_i)
        g = np.array(data.g)

        # 4. τ = M*qddot + C*qdot + g
        tau_pred[i] = M @ a_i + C @ v_i + g

        # 5. Friction compensation: coulomb*tanh(v/eps) + viscous*v + armature*qddot
        smooth_sign = np.tanh(v_i / eps)
        tau_pred[i] += coulomb_gain * smooth_sign + viscous_gain * v_i + armature * a_i

    # Plot comparison per joint
    fig, axes = plt.subplots(njoints, 1, sharex=True, figsize=(12, 2 * njoints))
    if njoints == 1:
        axes = [axes]
    for j in range(njoints):
        ax = axes[j]
        ax.plot(t, tau_meas[:, j], color='C0', linewidth=1.5, label="Measured" if j == 0 else None)
        ax.plot(t, tau_pred[:, j], color='C3', linewidth=1.0, alpha=0.7, label="RNEA Predicted" if j == 0 else None)
        ax.set_ylabel(f"J{j+1}\n(N·m)")
        rms = np.sqrt(np.mean((tau_meas[:, j] - tau_pred[:, j]) ** 2))
        ax.text(0.02, 0.95, f"RMS: {rms:.4f}", transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Measured Torque (blue) vs RNEA Predicted (red)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
