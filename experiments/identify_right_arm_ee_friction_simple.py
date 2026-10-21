#!/usr/bin/env python
"""EE-link inertia + friction identification for marvinM6 right arm.

Based on identify_right_arm_friction_simple.py (URDF inertia frozen), adding
identification of the last link (end-effector link) inertia (10 params).
Inertial parameters of links 1..nj-1 stay frozen at URDF values: their torque
contribution is computed via the URDF parameters (regressor identity
Y·phi_urdf = rnea holds exactly), the EE link's nominal contribution is
removed from the residual and replaced by the identified parameters.

Parameter layout (identified):
  [EE inertia(10) | Fc(nj) | Fv(nj) | armature(nj) | offset(nj)]

Optional physical constraint (--nonneg / --no-nonneg, default ON): friction
(Fc, Fv) and armature are constrained >= 0 via bounded lsq_linear (EE inertia
and offset unbounded). Unconstrained lstsq can assign negative
friction/armature when they are collinear with URDF inertia errors.

Output npz layout (identical to identify_right_arm.py / identify_right_arm_joint.py,
directly consumable by the C++ TorquePredictor used by replay/dyn_comp/collision
observer):
  phi = [inertia(10*nj) | Fc(nj) | Fv(nj) | armature(nj) | offset(nj)]
  where the inertia block is URDF params for links 1..nj-1 and the identified
  EE-link params for the last link. inertia_frozen=True is set.
  vbrk = fixed scalar (C++ broadcasts to all joints)

Usage:
    python experiments/identify_right_arm_ee_friction_simple.py [--data_dir DIR] [--acc_source sensors|csv] [--tau_source joint|current] [--no-nonneg]

Outputs:
    fig1: acceleration used vs csv acceleration
    fig2: predicted vs measured torque (URDF nominal + identified EE inertia + friction)
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from scipy.optimize import lsq_linear

import pinocchio as pin
from system_identification.utils import find_path, savgol_filter_acceleration


def load_data(data_dir, tau_source="joint", trim_head=1, trim_tail=1):
    """Load and align all sensor/command data onto a common time axis."""
    data_dir = Path(data_dir)
    df_q = pd.read_csv(data_dir / "sensors_joint_q.csv")
    df_v = pd.read_csv(data_dir / "sensors_joint_v.csv")
    tau_file = {"joint": "sensors_joint_torque.csv", "current": "sensor_actual_torque.csv"}[tau_source]
    logger.info(f"Torque source: {tau_source} ({tau_file})")
    df_tau = pd.read_csv(data_dir / tau_file)
    df_cmd = pd.read_csv(data_dir / "csv_trajectory_a.csv")

    t_ref = df_q["time"].values
    q = df_q.iloc[:, 1:8].values
    nj = q.shape[1]

    t_v = df_v["time"].values
    v_raw = df_v.iloc[:, 1:1+nj].values
    if len(t_v) == len(t_ref) and np.allclose(t_v, t_ref):
        v = v_raw
    else:
        v = np.column_stack([np.interp(t_ref, t_v, v_raw[:, j]) for j in range(nj)])

    t_tau = df_tau["time"].values
    tau_raw = df_tau.iloc[:, 1:1+nj].values
    if len(t_tau) == len(t_ref) and np.allclose(t_tau, t_ref):
        tau_meas = tau_raw
    else:
        tau_meas = np.column_stack([np.interp(t_ref, t_tau, tau_raw[:, j]) for j in range(nj)])

    t_cmd = df_cmd["time"].values
    a_cmd_raw = df_cmd.iloc[:, 1:1+nj].values
    a_cmd = np.column_stack([np.interp(t_ref, t_cmd, a_cmd_raw[:, j]) for j in range(nj)])

    a_meas = None
    a_meas_file = data_dir / "sensors_joint_a.csv"
    if a_meas_file.exists():
        df_a = pd.read_csv(a_meas_file)
        t_a = df_a["time"].values
        a_raw = df_a.iloc[:, 1:1+nj].values
        if len(t_a) == len(t_ref) and np.allclose(t_a, t_ref):
            a_meas = a_raw
        else:
            a_meas = np.column_stack([np.interp(t_ref, t_a, a_raw[:, j]) for j in range(nj)])

    logger.info(f"Aligned {nj} signals onto q-sensor clock ({len(t_ref)} samples)")
    for name, t_src in [("v", t_v), ("tau", t_tau), ("cmd", t_cmd)]:
        if len(t_src) != len(t_ref) or not np.allclose(t_src, t_ref):
            logger.info(f"  {name}: {len(t_src)} samples -> interpolated to {len(t_ref)}")
        else:
            logger.info(f"  {name}: already aligned ({len(t_src)} samples)")

    th, tt = trim_head, trim_tail
    if th > 0 or tt > 0:
        t_ref = t_ref[th:len(t_ref)-tt] if tt > 0 else t_ref[th:]
        q = q[th:len(q)-tt] if tt > 0 else q[th:]
        v = v[th:len(v)-tt] if tt > 0 else v[th:]
        tau_meas = tau_meas[th:len(tau_meas)-tt] if tt > 0 else tau_meas[th:]
        a_cmd = a_cmd[th:len(a_cmd)-tt] if tt > 0 else a_cmd[th:]
        if a_meas is not None:
            a_meas = a_meas[th:len(a_meas)-tt] if tt > 0 else a_meas[th:]
        logger.info(f"Trimmed head={th}, tail={tt}, remaining: {len(t_ref)} samples")

    dt = np.diff(t_ref)
    a_fd = np.zeros_like(v)
    a_fd[1:-1] = (v[2:] - v[:-2]) / (dt[1:] + dt[:-1])[:, None]
    a_fd = savgol_filter_acceleration(a_fd, window_length=21, polyorder=3)

    return t_ref, q, v, a_fd, tau_meas, a_cmd, a_meas


def feat_block(feat, N, nj):
    """Build regressor block from feature array (N, nj) -> (N*nj, nj)."""
    repeated = np.repeat(feat, nj, axis=0)
    ident = np.tile(np.eye(nj), (N, 1))
    return repeated * ident


def main():
    parser = argparse.ArgumentParser(description="EE-link inertia + friction identification (links 1..nj-1 URDF frozen)")
    parser.add_argument("--data_dir", type=str, default="./replay_right_arm_200_log_data")
    parser.add_argument("--robot", type=str, default="marvinM6_right")
    parser.add_argument("--acc_source", type=str, default="csv", choices=["sensors", "csv"])
    parser.add_argument("--tau_source", type=str, default="current", choices=["joint", "current"],
                        help="Torque input: joint=sensors_joint_torque.csv, current=sensor_actual_torque.csv")
    parser.add_argument("--trim_head", type=int, default=1, help="Frames to drop from start")
    parser.add_argument("--trim_tail", type=int, default=1, help="Frames to drop from end")
    parser.add_argument("--savgol_window", type=int, default=301)
    parser.add_argument("--savgol_poly", type=int, default=5)
    parser.add_argument("--vbrk", type=float, default=0.005, help="Coulomb smoothing parameter (vcoul = 2*vbrk)")
    parser.add_argument("--nonneg", action=argparse.BooleanOptionalAction, default=True,
                        help="Constrain Fc/Fv/armature >= 0 (EE inertia and offset unbounded) via bounded lsq_linear. Disable with --no-nonneg")
    parser.add_argument("--save_params", type=str, default="experiments/identified_params_right_arm_ee_friction.npz",
                        help="Path to save identified params NPZ")
    args = parser.parse_args()

    # Load + align + trim
    t, q, v, a_fd, tau_meas, a_cmd, a_meas = load_data(
        args.data_dir, tau_source=args.tau_source,
        trim_head=args.trim_head, trim_tail=args.trim_tail)
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

    # Build pinocchio model — zero out URDF friction/damping so RNEA gives pure
    # rigid-body torque, kept as the nominal comparison curve only.
    urdf_file = find_path(f"{args.robot}.urdf", "./robot_description")
    model = pin.buildModelFromUrdf(urdf_file)
    data = model.createData()
    model.friction[:] = 0.0
    model.damping[:] = 0.0
    logger.info(f"Computing URDF nominal RNEA torque (comparison only)...")

    # Full-URDF nominal rigid-body torque (comparison curve) + URDF minimal
    # parameter set extracted EXACTLY from the model inertias (pinocchio's
    # regressor is parameterized by these same 10 spatial-inertia params per
    # joint, so Y·phi_urdf = rnea holds identically — verified by spot-check
    # below). Links 1..nj-1 are frozen at these values.
    tau_rnea = np.zeros((N, nj))
    Y_inertia = np.zeros((N * nj, 10 * nj))
    for i in range(N):
        Y_inertia[i * nj:(i + 1) * nj] = pin.computeJointTorqueRegressor(model, data, q[i], v[i], a[i])
        tau_rnea[i] = pin.rnea(model, data, q[i], v[i], a[i])
    phi_inertia = np.concatenate([
        model.inertias[j].toDynamicParameters() for j in range(1, nj + 1)
    ])
    # Spot-check the regressor identity Y·phi_urdf = rnea on a few states
    check_err = 0.0
    for i in np.linspace(0, N - 1, min(20, N)).astype(int):
        Y_chk = pin.computeJointTorqueRegressor(model, data, q[i], v[i], a[i])
        check_err = max(check_err, float(np.abs(Y_chk @ phi_inertia - tau_rnea[i]).max()))
    logger.info(f"Nominal RNEA torque computed (mean abs: {np.mean(np.abs(tau_rnea)):.4f} N·m)")
    logger.info(f"URDF minimal inertia params extracted (frozen): "
                f"J1~J{nj} mass=[{', '.join(f'{model.inertias[j].mass:.4f}' for j in range(1, nj + 1))}], "
                f"max spot-check err={check_err:.2e} N·m")

    # EE-link (last link) regressor block and its URDF nominal contribution.
    # Residual nominal excludes the EE link: it is replaced by the identified
    # params, so links 1..nj-1 stay frozen at URDF values.
    n_frozen = 10 * (nj - 1)
    n_ee = 10
    Y_ee = Y_inertia[:, n_frozen:]
    phi_ee_urdf = phi_inertia[n_frozen:]
    tau_ee_urdf = (Y_ee @ phi_ee_urdf).reshape(N, nj)
    logger.info(f"EE link (J{nj}) nominal torque mean abs: {np.mean(np.abs(tau_ee_urdf)):.4f} N·m")

    # Build friction regressor: [Fc(nj) | Fv(nj)]
    logger.info("Building friction regressor (Coulomb + viscous)...")
    vcoul = args.vbrk * 2
    tanh_v = np.tanh(v / vcoul)
    Y_friction = np.hstack([
        feat_block(tanh_v, N, nj),
        feat_block(v, N, nj),
    ])

    # Build armature regressor: [armature(nj)]
    logger.info("Building armature regressor...")
    Y_armature = feat_block(a, N, nj)

    # Build torque offset regressor: [offset(nj)]
    logger.info("Building torque offset regressor...")
    Y_offset = feat_block(np.ones_like(v), N, nj)

    # Residual = measured - nominal rigid-body torque (links 1..nj-1 URDF frozen,
    # EE link removed) — the identified EE inertia explains it together with
    # friction/armature/offset.
    tau_resid = tau_meas - (tau_rnea - tau_ee_urdf)
    tau_resid_flat = tau_resid.reshape(-1)
    logger.info(f"Residual mean abs: {np.mean(np.abs(tau_resid)):.4f} N·m")

    # Solve regressor: [EE inertia(10) | Fc(nj) | Fv(nj) | armature(nj) | offset(nj)]
    Y_all = np.hstack([Y_ee, Y_friction, Y_armature, Y_offset])
    n_friction = 2 * nj
    n_armature = nj
    n_offset = nj
    n_params = n_ee + n_friction + n_armature + n_offset
    logger.info(f"Regressor shape: {Y_all.shape} ({n_ee} EE inertia + {n_friction} friction + "
                f"{n_armature} armature + {n_offset} offset = {n_params} params)")
    logger.info(f"Condition: {np.linalg.cond(Y_all):.2f}")

    # Solve: EE inertia/friction/armature/offset against the residual
    if args.nonneg:
        # 物理约束解：Fc/Fv/armature >= 0（EE 惯性与 offset 不限）。armature 列与冻结
        # 惯性对角项共线，无约束最小二乘会把 URDF 惯性误差记成负 armature；边界约束
        # 下该误差回到残差，换取参数物理合规。
        lower = np.full(n_params, -np.inf)
        lower[n_ee:n_ee + n_friction + n_armature] = 0.0
        upper = np.full(n_params, np.inf)
        res = lsq_linear(Y_all, tau_resid_flat, bounds=(lower, upper), method='trf', tol=1e-12)
        phi = res.x
        logger.info(f"Nonneg bounded solve: status={res.status}, optimality={res.optimality:.2e}")
    else:
        phi, _, rank, sv = np.linalg.lstsq(Y_all, tau_resid_flat, rcond=None)
        logger.info(f"Rank: {rank}/{n_params}, singular values: [{sv.min():.4e}, {sv.max():.4e}]")
    resid_cost = float(np.linalg.norm(Y_all @ phi - tau_resid_flat))
    logger.info(f"||resid||={resid_cost:.4f}")

    # Predict: total = (RNEA - EE nominal) + identified EE inertia + friction model
    tau_ee_pred = (Y_ee @ phi[:n_ee]).reshape(N, nj)
    tau_friction_pred = (Y_all[:, n_ee:] @ phi[n_ee:]).reshape(N, nj)
    tau_pred = (tau_rnea - tau_ee_urdf) + tau_ee_pred + tau_friction_pred

    # Print per-joint RMS
    logger.info("=" * 70)
    logger.info(f"{'Joint':>6} | {'RMS meas':>10} | {'RNEA err':>12} {'ratio':>7} | {'Ident err':>12} {'ratio':>7}")
    logger.info("-" * 70)
    for j in range(nj):
        rms_meas = np.sqrt(np.mean(tau_meas[:, j] ** 2))
        rms_rnea_err = np.sqrt(np.mean((tau_meas[:, j] - tau_rnea[:, j]) ** 2))
        rms_id_err = np.sqrt(np.mean((tau_meas[:, j] - tau_pred[:, j]) ** 2))
        logger.info(f"J{j+1:>4}  | {rms_meas:10.4f} | {rms_rnea_err:12.6f} {rms_rnea_err/rms_meas:7.4f} | "
                    f"{rms_id_err:12.6f} {rms_id_err/rms_meas:7.4f}")
    logger.info("=" * 70)

    # Print inertia params: links 1..nj-1 frozen at URDF, EE link identified
    logger.info(f"--- Inertia (links 1..{nj - 1} frozen at URDF, J{nj} identified) ---")
    phi_ee = phi[:n_ee]
    for j in range(nj):
        pi = phi_ee if j == nj - 1 else phi_inertia[j * 10:(j + 1) * 10]
        src = "ident" if j == nj - 1 else "URDF"
        mass = pi[0]
        com = pi[1:4] / mass if abs(mass) > 1e-10 else np.zeros(3)
        logger.info(f"J{j+1} [{src}]: Mass={mass:.6f}, CoM=[{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}]")
    m_ee_urdf = phi_ee_urdf[0]
    logger.info(f"EE link mass: URDF nominal {m_ee_urdf:.6f} -> identified {phi_ee[0]:.6f} (delta {phi_ee[0] - m_ee_urdf:+.6f})")

    # Print friction params
    logger.info("--- Friction (Coulomb Fc / viscous Fv) ---")
    phi_fric = phi[n_ee:n_ee + n_friction].reshape(2, nj)
    for j in range(nj):
        logger.info(f"J{j+1}: Fc={phi_fric[0,j]:.6f}, Fv={phi_fric[1,j]:.6f}")

    # Print armature params
    logger.info("--- Armature ---")
    phi_arm = phi[n_ee + n_friction:n_ee + n_friction + n_armature]
    for j in range(nj):
        logger.info(f"J{j+1}: Armature={phi_arm[j]:.6f}")

    # Print torque offset params
    logger.info("--- Torque Offset ---")
    phi_off = phi[n_ee + n_friction + n_armature:]
    for j in range(nj):
        logger.info(f"J{j+1}: Offset={phi_off[j]:.6f}")

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
        ax.plot(t, tau_rnea[:, j], 'g-', linewidth=1, alpha=0.5, label='URDF nominal (RNEA)' if j == 0 else None)
        ax.plot(t, tau_pred[:, j], 'r--', linewidth=1, label='Identified (EE inertia + friction)' if j == 0 else None)
        ax.set_ylabel(f"J{j+1}\n(N·m)")
    axes2[0].legend(loc='upper right', fontsize=8)
    axes2[-1].set_xlabel("time (s)")
    fig2.suptitle(f"Predicted vs Measured Torque (links 1..{nj-1} URDF frozen + EE inertia + friction, nonneg={args.nonneg})")

    # Save identified parameters — same format as identify_right_arm.py so the
    # C++ TorquePredictor (replay/dyn_comp/collision observer) consumes the
    # npz through the exact same code path as the full npz.
    # phi = [inertia(10*nj) | Fc(nj) | Fv(nj) | armature(nj) | offset(nj)];
    # inertia block = URDF params for links 1..nj-1, identified EE params for
    # the last link; vbrk = fixed scalar (C++ TorquePredictor broadcasts it to
    # all joints); inertia_frozen=True marks that links 1..nj-1 are the exact
    # URDF parameter set (not identified from data).
    n_inertia = 10 * nj
    n_params_full = n_inertia + n_friction + n_armature + n_offset
    phi_inertia_full = phi_inertia.copy()
    phi_inertia_full[n_frozen:] = phi_ee
    phi_full = np.concatenate([phi_inertia_full, phi[n_ee:]])
    np.savez(
        args.save_params,
        phi=phi_full,
        robot=args.robot,
        njoints=nj,
        n_params=n_params_full,
        n_inertia=n_inertia,
        n_friction=n_friction,
        n_armature=n_armature,
        n_offset=n_offset,
        vbrk=args.vbrk,
        acc_source=args.acc_source,
        vcoul=2 * args.vbrk,
        inertia_frozen=True,
        nonneg=bool(args.nonneg),
    )
    logger.info(f"Saved {n_params_full} params (inertia = URDF links 1..{nj-1} + identified EE link, "
                f"vbrk = fixed {args.vbrk:.6f}, nonneg={args.nonneg}) to {args.save_params}")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
