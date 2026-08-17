#!/usr/bin/env python
"""Compare identified parameters (phi + vbrk) on validation data.

Loads a saved NPZ parameter file, builds the regressor from validation data,
and compares identified torque prediction vs measured vs nominal URDF RNEA.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

from system_identification.inertia_model import InertiaModel
from system_identification.excitation_optimization import (
    generateSymFrictionReg,
    generateAsymFrictionReg,
)
from system_identification.utils import (
    savgol_filter_acceleration,
    compute_causal_acceleration,
    feature2regressor,
    find_path,
)


def load_data(data_dir):
    """Load sensor and command data from a data directory."""
    data_dir = Path(data_dir)
    df_q = pd.read_csv(data_dir / "sensors_joint_q.csv")
    df_v = pd.read_csv(data_dir / "sensors_joint_v.csv")
    df_tau = pd.read_csv(data_dir / "sensors_joint_torque.csv")
    df_cmd = pd.read_csv(data_dir / "csv_trajectory_a.csv")

    t = df_q["time"].values
    q = df_q.iloc[:, 1:8].values
    v = df_v.iloc[:, 1:8].values
    tau = df_tau.iloc[:, 1:8].values

    dt = np.diff(t)
    a_fd = np.zeros_like(v)
    a_fd[1:-1] = (v[2:] - v[:-2]) / (dt[1:] + dt[:-1])[:, None]
    a_fd = savgol_filter_acceleration(a_fd, window_length=801, polyorder=5)

    t_cmd = df_cmd["time"].values
    a_cmd = df_cmd.iloc[:, 1:8].values
    return t, q, v, a_fd, tau, a_cmd


def main():
    parser = argparse.ArgumentParser(description="Compare identified params on validation data")
    parser.add_argument("--params", type=str, default="experiments/identified_params.npz",
                        help="Path to NPZ file with identified phi and vbrk (default: experiments/identified_params.npz)")
    parser.add_argument("--data_dir", type=str, default="./test_data",
                        help="Validation data directory")
    parser.add_argument("--trim", type=int, default=1,
                        help="Number of frames to trim from start and end")
    parser.add_argument("--val_acc_method", type=str, default="central",
                        choices=["central", "causal", "zero"],
                        help="Acceleration computation method: 'central', 'causal', or 'zero' (no acceleration input)")
    args = parser.parse_args()

    # Load identified parameters
    data = np.load(args.params, allow_pickle=True)
    phi = data["phi"]
    vbrk = float(data["vbrk"])
    robot = str(data["robot"])
    friction_model = str(data["friction_model"])
    njoints = int(data["njoints"])
    n_params = int(data["n_params"])
    logger.info(f"Loaded params from {args.params}")
    logger.info(f"  robot={robot}, friction_model={friction_model}, njoints={njoints}")
    logger.info(f"  vbrk={vbrk:.8f}, phi shape={phi.shape}")

    # Load validation data
    t, q, v, a_fd, tau_meas, a_cmd = load_data(args.data_dir)
    n_samples_full, nj = q.shape
    logger.info(f"Loaded {n_samples_full} samples, {njoints} joints from {args.data_dir}")

    # Trim boundary frames
    n = args.trim
    t = t[n:-n]; q = q[n:-n]; v = v[n:-n]
    a_fd = a_fd[n:-n]; tau_meas = tau_meas[n:-n]; a_cmd = a_cmd[n:-n]
    n_samples = t.shape[0]
    logger.info(f"Trimmed {n} frames from each end, remaining: {n_samples} samples")

    # Compute acceleration
    if args.val_acc_method == "central":
        a = a_fd
        acc_label = "Central Diff + Savgol"
    elif args.val_acc_method == "causal":
        a = compute_causal_acceleration(t, v)
        acc_label = "Causal Savgol"
    else:  # zero
        a = np.zeros_like(v)
        acc_label = "Zero Acceleration"

    # Build regressor: inertia + friction + armature
    inertia_model = InertiaModel(robot)
    reg_inertia = inertia_model.regressor(q, v, a)
    reg_armature = feature2regressor([a], n_samples, njoints)
    if friction_model == "symmetric":
        reg_friction = generateSymFrictionReg(v, vbrk=vbrk)
    else:
        reg_friction = generateAsymFrictionReg(v, vbrk=vbrk)
    regressor = np.hstack([reg_inertia, reg_friction, reg_armature])

    # Predict torque using identified phi
    tau_pred = (regressor @ phi).reshape(n_samples, njoints)

    # Nominal URDF RNEA prediction
    import pinocchio as pin
    urdf_file = find_path(f"{robot}.urdf", "./robot_description")
    nom_model = pin.buildModelFromUrdf(urdf_file)
    nom_data = nom_model.createData()
    nom_coulomb = np.array(nom_model.friction)
    nom_viscous = np.array(nom_model.damping)
    nom_armature = 0.05
    eps_nom = 0.0001
    tau_nominal = np.zeros((n_samples, njoints))
    for i in range(n_samples):
        tau_nominal[i] = pin.rnea(nom_model, nom_data, q[i], v[i], a[i])
        tau_nominal[i] += nom_coulomb * np.tanh(v[i] / eps_nom) + nom_viscous * v[i] + nom_armature * a[i]
    logger.info("Nominal URDF RNEA prediction computed")

    # Print 3-way comparison
    logger.info("=" * 80)
    logger.info("3-way RMS error comparison: measured vs nominal (RNEA) vs identified")
    logger.info(f"{'Joint':>6} | {'RMS meas':>10} | {'Nominal err':>12} {'ratio':>7} | {'Ident err':>12} {'ratio':>7}")
    logger.info("-" * 80)
    for j in range(njoints):
        rms_meas = np.sqrt(np.mean(tau_meas[:, j] ** 2))
        rms_nom_err = np.sqrt(np.mean((tau_meas[:, j] - tau_nominal[:, j]) ** 2))
        rms_id_err = np.sqrt(np.mean((tau_meas[:, j] - tau_pred[:, j]) ** 2))
        logger.info(f"J{j+1:>4}  | {rms_meas:10.4f} | {rms_nom_err:12.6f} {rms_nom_err/rms_meas:7.4f} | "
                    f"{rms_id_err:12.6f} {rms_id_err/rms_meas:7.4f}")
    logger.info("=" * 80)

    # Plot torque comparison
    fig, axes = plt.subplots(njoints, 1, sharex=True, figsize=(12, 2 * njoints))
    if njoints == 1:
        axes = [axes]
    for j in range(njoints):
        ax = axes[j]
        ax.plot(t, tau_meas[:, j], 'b-', linewidth=1, label='Measured')
        ax.plot(t, tau_nominal[:, j], 'g-', linewidth=1, alpha=0.5, label='Nominal (RNEA)')
        ax.plot(t, tau_pred[:, j], 'r--', linewidth=1, label='Identified')
        ax.set_ylabel(f"Joint {j+1}\n(N·m)")
        if j == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"VALIDATION ({args.data_dir}): Measured vs Nominal vs Identified (phi from {Path(args.params).name})")
    plt.tight_layout()

    # Plot acceleration
    fig_acc, axes_acc = plt.subplots(njoints, 1, sharex=True, figsize=(12, 2 * njoints))
    if njoints == 1:
        axes_acc = [axes_acc]
    for j in range(njoints):
        ax = axes_acc[j]
        ax.plot(t, a_cmd[:, j], color='C0', linewidth=1.5, label="Expected" if j == 0 else None)
        ax.plot(t, a[:, j], color='C3', linewidth=1.0, alpha=0.7, label=acc_label if j == 0 else None)
        ax.set_ylabel(f"J{j+1}\n(rad/s²)")
    axes_acc[0].legend(loc='upper right', fontsize=8)
    axes_acc[-1].set_xlabel("time (s)")
    fig_acc.suptitle(f"VALIDATION: Expected Acceleration (blue) vs {acc_label} (red)")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
