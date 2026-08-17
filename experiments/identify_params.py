#!/usr/bin/env python
"""System identification from logged data.

Loads filtered sensor data (q, v, torque), computes acceleration via finite
differencing of velocity, builds the inertia+friction regressor, reduces via
QR, solves least squares, and plots measured vs predicted torque per joint.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
import xml.etree.ElementTree as ET
import yaml

from system_identification.inertia_model import InertiaModel
from system_identification.excitation_optimization import (
    generateSymFrictionReg,
    generateAsymFrictionReg,
)
from system_identification.utils import QR_dim_reduction, savgol_filter_acceleration, compute_causal_acceleration, feature2regressor, find_path, inertiaVecToIcQs


def write_to_urdf(robot_name, phi, njoints, friction_model, output_path=None):
    """Write identified parameters back to a URDF file.

    Writes back:
      - Inertia (mass, CoM, inertia tensor) to each link's <inertial> block
      - Friction (coulomb, viscous) to each joint's <dynamics> block
      - Armature as a custom <dynamics armature="..."> attribute (Pinocchio extension)

    Args:
        robot_name: robot name for finding the source URDF
        phi: identified parameter vector [inertia(10*nj) | friction | armature]
        njoints: number of joints
        friction_model: "symmetric" or "asymmetric"
        output_path: output URDF path, default is source_urdf_identicated.urdf
    """
    urdf_file = find_path(f"{robot_name}.urdf", "./robot_description")
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Parse revolute joints (in order, matching Pinocchio joint indices 1..nj)
    joints = [j for j in root.findall("joint") if j.get("type") == "revolute"]
    # Map joint index -> child link element (the link whose inertia belongs to this joint)
    child_link_names = [j.find("child").get("link") for j in joints[:njoints]]
    link_map = {l.get("name"): l for l in root.findall("link")}

    n_inertia = 10 * njoints
    if friction_model == "symmetric":
        n_friction = 2 * njoints
    else:
        n_friction = 4 * njoints
    n_armature = njoints

    # ---- Update inertia parameters (mass, CoM, inertia tensor) per link ----
    for i in range(njoints):
        link = link_map.get(child_link_names[i])
        if link is None:
            logger.warning(f"Link '{child_link_names[i]}' not found, skipping inertia write")
            continue
        inertial = link.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(link, "inertial")

        phi_i = phi[i * 10: (i + 1) * 10]
        mass = phi_i[0]
        # First moments h = m * CoM  =>  CoM = h / m
        if mass > 1e-10:
            com = phi_i[1:4] / mass
        else:
            com = np.zeros(3)

        # Get inertia tensor at CoM
        Ic, _ = inertiaVecToIcQs(phi_i)

        # Update <origin xyz=...> inside <inertial>
        origin = inertial.find("origin")
        if origin is None:
            origin = ET.SubElement(inertial, "origin")
        origin.set("xyz", f"{com[0]:.8f} {com[1]:.8f} {com[2]:.8f}")
        origin.set("rpy", "0 0 0")

        # Update <mass>
        mass_el = inertial.find("mass")
        if mass_el is None:
            mass_el = ET.SubElement(inertial, "mass")
        mass_el.set("value", f"{mass:.8f}")

        # Update <inertia> (Ic is symmetric 3x3 at CoM)
        inertia_el = inertial.find("inertia")
        if inertia_el is None:
            inertia_el = ET.SubElement(inertial, "inertia")
        inertia_el.set("ixx", f"{Ic[0, 0]:.8f}")
        inertia_el.set("ixy", f"{Ic[0, 1]:.8f}")
        inertia_el.set("ixz", f"{Ic[0, 2]:.8f}")
        inertia_el.set("iyy", f"{Ic[1, 1]:.8f}")
        inertia_el.set("iyz", f"{Ic[1, 2]:.8f}")
        inertia_el.set("izz", f"{Ic[2, 2]:.8f}")

    # ---- Update friction parameters in joints ----
    phi_friction = phi[n_inertia:n_inertia + n_friction]
    if friction_model == "symmetric":
        phi_friction = phi_friction.reshape(njoints, 2)
        for i in range(njoints):
            joint = joints[i]
            dyn = joint.find("dynamics")
            if dyn is None:
                dyn = ET.SubElement(joint, "dynamics")
            # coulomb = friction, viscous = damping
            dyn.set("friction", f"{phi_friction[i, 0]:.8f}")
            dyn.set("damping", f"{phi_friction[i, 1]:.8f}")
    else:
        phi_friction = phi_friction.reshape(njoints, 4)
        for i in range(njoints):
            joint = joints[i]
            dyn = joint.find("dynamics")
            if dyn is None:
                dyn = ET.SubElement(joint, "dynamics")
            # average positive/negative for URDF (URDF only supports symmetric)
            fc_avg = (abs(phi_friction[i, 0]) + abs(phi_friction[i, 2])) / 2
            fv_avg = (abs(phi_friction[i, 1]) + abs(phi_friction[i, 3])) / 2
            dyn.set("friction", f"{fc_avg:.8f}")
            dyn.set("damping", f"{fv_avg:.8f}")

    # Write output (default: overwrite source URDF)
    if output_path is None:
        output_path = urdf_file

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    logger.info(f"Identified URDF written to: {output_path} (inertia + friction)")


def load_data(data_dir):
    """Load sensor and command data from log_data directory.

    Returns:
        t: time array
        q: measured joint positions (n_samples, 7)
        v: measured joint velocities (n_samples, 7)
        a_fd: acceleration via central difference of v (n_samples, 7)
        tau: measured torques (n_samples, 7)
        a_cmd: expected acceleration from command trajectory (n_samples, 7)
    """
    data_dir = Path(data_dir)

    df_q = pd.read_csv(data_dir / "sensors_joint_q.csv")
    df_v = pd.read_csv(data_dir / "sensors_joint_v.csv")
    df_tau = pd.read_csv(data_dir / "sensors_joint_torque.csv")
    df_cmd = pd.read_csv(data_dir / "csv_trajectory_a.csv")

    # Extract left arm joints: sensors j12-j18 (first 7 of 14 columns)
    t = df_q["time"].values
    q = df_q.iloc[:, 1:8].values      # j12-j18
    v = df_v.iloc[:, 1:8].values      # j12-j18
    tau = df_tau.iloc[:, 1:8].values   # j12-j18

    # Compute acceleration via central difference of filtered velocity
    dt = np.diff(t)
    a_fd = np.zeros_like(v)
    a_fd[1:-1] = (v[2:] - v[:-2]) / (dt[1:] + dt[:-1])[:, None]
    # Apply Savitzky-Golay filter to smooth the central-difference acceleration
    a_fd = savgol_filter_acceleration(a_fd, window_length=801, polyorder=5)

    # Expected acceleration from command trajectory (csv_trajectory_a: j0-j6 are accelerations)
    t_cmd = df_cmd["time"].values
    a_cmd = df_cmd.iloc[:, 1:8].values  # j0-j6 = left arm expected acceleration

    return t, q, v, a_fd, tau, a_cmd


def build_reg_vector(njoints, friction_model, reg_config_path, fallback_lambda):
    """Build a per-parameter regularization weight vector from a YAML config.

    The parameter vector layout is:
      [inertia(10*nj) | friction | armature(nj)]
    where inertia is joint-major (10 params per joint):
      [mass, com_x, com_y, com_z, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    and friction is grouped by feature:
      symmetric:  [coulomb_j... , viscous_j...]            (2*nj)
      asymmetric: [coulomb+_j, viscous+_j, coulomb-_j, viscous-_j] (4*nj)

    YAML structure (see experiments/reg_config.yaml):
      regularization:
        default: <float>
        joints:
          - joint: 1
            mass: <float>
            com: <float>       # applied to all 3 com first-moment params
            inertia: <float>   # applied to all 6 inertia-tensor params
            coulomb: <float>   # applied to Fc+ and Fc- in asymmetric model
            viscous: <float>   # applied to Fv+ and Fv- in asymmetric model
            armature: <float>

    Resolution priority per parameter:
      1. explicit value in the matching joint entry
      2. `default` in the YAML
      3. `fallback_lambda` (the --lambda_reg CLI argument)

    Returns:
        lam: ndarray of shape (n_params,) with regularization weights.
    """
    n_inertia = 10 * njoints
    if friction_model == "symmetric":
        n_friction = 2 * njoints
    else:
        n_friction = 4 * njoints
    n_armature = njoints
    n_params = n_inertia + n_friction + n_armature

    lam = np.full(n_params, float(fallback_lambda), dtype=float)

    if reg_config_path is None or not Path(reg_config_path).exists():
        logger.warning(f"Regularization config not found ({reg_config_path}); "
                       f"using uniform lambda_reg={fallback_lambda} for all parameters")
        return lam

    with open(reg_config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    reg = cfg.get("regularization", {}) or {}
    default_val = float(reg.get("default", fallback_lambda))
    lam[:] = default_val

    quantities = ("mass", "com", "inertia", "coulomb", "viscous", "armature")
    joint_map = {}
    for entry in reg.get("joints", []) or []:
        ji = entry.get("joint")
        if ji is None:
            continue
        joint_map[int(ji) - 1] = entry

    for j in range(njoints):
        entry = joint_map.get(j, {}) or {}
        vals = {q: float(entry.get(q, default_val)) for q in quantities}

        # inertia block: [mass(1), com(3), inertia-tensor(6)]
        base = j * 10
        lam[base + 0] = vals["mass"]
        lam[base + 1: base + 4] = vals["com"]
        lam[base + 4: base + 10] = vals["inertia"]

        # friction block
        if friction_model == "symmetric":
            lam[n_inertia + j] = vals["coulomb"]                       # coulomb
            lam[n_inertia + njoints + j] = vals["viscous"]             # viscous
        else:
            lam[n_inertia + 0 * njoints + j] = vals["coulomb"]        # Fc+
            lam[n_inertia + 1 * njoints + j] = vals["viscous"]        # Fv+
            lam[n_inertia + 2 * njoints + j] = vals["coulomb"]        # Fc-
            lam[n_inertia + 3 * njoints + j] = vals["viscous"]        # Fv-

        # armature block
        lam[n_inertia + n_friction + j] = vals["armature"]

    logger.info(f"Loaded regularization config: {reg_config_path} (default={default_val})")
    for j in range(njoints):
        entry = joint_map.get(j, {}) or {}
        logger.info(
            f"  J{j+1}: mass={entry.get('mass', default_val)}, "
            f"com={entry.get('com', default_val)}, "
            f"inertia={entry.get('inertia', default_val)}, "
            f"coulomb={entry.get('coulomb', default_val)}, "
            f"viscous={entry.get('viscous', default_val)}, "
            f"armature={entry.get('armature', default_val)}"
        )
    return lam


def main():
    parser = argparse.ArgumentParser(description="System identification from logged data")
    parser.add_argument("--robot", type=str, default="biped_s49_left_arm")
    parser.add_argument("--friction_model", type=str, default="symmetric",
                        choices=["symmetric", "asymmetric"])
    parser.add_argument("--data_dir", type=str, default="./log_data")
    parser.add_argument("--val_dir", type=str, default="./test_data", help="Directory for validation data")
    parser.add_argument("--trim", type=int, default=1, help="Number of frames to trim from start and end")
    parser.add_argument("--output_urdf", type=str, default=None, help="Output URDF path (default: disabled)")
    parser.add_argument("--lambda_reg", type=float, default=0.0001,
                        help="Fallback regularization weight for parameters not specified in the YAML config")
    parser.add_argument("--reg_config", type=str,
                        default=str(Path(__file__).resolve().parent / "reg_config.yaml"),
                        help="Path to YAML with per-joint regularization weights (mass/com/inertia/coulomb/viscous/armature)")
    parser.add_argument("--vbrk_init", type=float, default=0.0001, help="Initial guess for vbrk (nonlinear friction smoothness)")
    parser.add_argument("--vbrk_lb", type=float, default=1e-6, help="Lower bound for vbrk")
    parser.add_argument("--vbrk_ub", type=float, default=0.1, help="Upper bound for vbrk")
    parser.add_argument("--val_acc_method", type=str, default="central",
                        choices=["central", "causal", "zero"],
                        help="Acceleration computation method for validation: "
                             "'central' = central difference + Savitzky-Golay (non-causal), "
                             "'causal' = causal backward difference + causal Savitzky-Golay, "
                             "'zero' = zero acceleration (gravity + friction only)")
    parser.add_argument("--save_params", type=str, default="experiments/identified_params.npz",
                        help="Save identified phi and vbrk to an NPZ file (default: experiments/identified_params.npz)")
    args = parser.parse_args()

    # Load data
    t, q, v, a_fd, tau_meas, a_cmd = load_data(args.data_dir)
    n_samples_full, njoints = q.shape
    logger.info(f"Loaded {n_samples_full} samples, {njoints} joints")

    # Trim boundary frames to avoid edge artifacts in acceleration
    n = args.trim
    t = t[n:-n]
    q = q[n:-n]
    v = v[n:-n]
    a_fd = a_fd[n:-n]
    a_cmd = a_cmd[n:-n]
    tau_meas = tau_meas[n:-n]
    logger.info(f"Trimmed {n} frames from each end, remaining: {t.shape[0]} samples")

    # Plot raw data: position, velocity, torque
    fig_data, axes_data = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    labels = ["Position (rad)", "Velocity (rad/s)", "Torque (N·m)"]
    colors = ["C0", "C1", "C2"]
    for k in range(3):
        ax = axes_data[k]
        for j in range(njoints):
            ax.plot(t, [q, v, tau_meas][k][:, j], color=colors[j] if j < len(colors) else None,
                    linewidth=1, label=f"J{j+1}" if k == 0 else None)
        ax.set_ylabel(labels[k])
    axes_data[0].legend(loc='upper right', ncol=njoints, fontsize=8)
    axes_data[-1].set_xlabel("time (s)")
    fig_data.suptitle("Sensor Data")
    plt.tight_layout()

    # Plot expected vs central-diff acceleration per joint
    fig_acc, axes_acc = plt.subplots(njoints, 1, sharex=True, figsize=(12, 2 * njoints))
    if njoints == 1:
        axes_acc = [axes_acc]
    for j in range(njoints):
        ax = axes_acc[j]
        ax.plot(t, a_cmd[:, j], color='C0', linewidth=1.5, label="Expected" if j == 0 else None)
        ax.plot(t, a_fd[:, j], color='C3', linewidth=1.0, alpha=0.7, label="Central Diff" if j == 0 else None)
        ax.set_ylabel(f"J{j+1}\n(rad/s²)")
    axes_acc[0].legend(loc='upper right', fontsize=8)
    axes_acc[-1].set_xlabel("time (s)")
    fig_acc.suptitle("Expected Acceleration (blue) vs Central Diff (red)")
    plt.tight_layout()

    # Use central-diff acceleration for identification
    a = a_fd
    n_samples = t.shape[0]

    # Build inertia model
    inertia_model = InertiaModel(args.robot)

    # Build regressor: inertia + armature + friction
    logger.info("Building regressor...")
    reg_inertia = inertia_model.regressor(q, v, a)

    # Armature regressor: armature_j * qddot[j], one param per joint
    reg_armature = feature2regressor([a], n_samples, njoints)

    # Flatten measured torque for least squares: shape (n_samples * njoints,)
    tau_flat = tau_meas.reshape(-1)

    # Build initial guess: inertia and friction from URDF, armature = 0
    n_params = 10 * njoints + (2 if args.friction_model == "symmetric" else 4) * njoints + njoints
    phi0 = np.zeros(n_params)
    phi0[:10 * njoints] = inertia_model.dyn_param
    if args.friction_model == "symmetric":
        phi0[10 * njoints:10 * njoints + njoints] = np.array(inertia_model.coulomb)
        phi0[10 * njoints + njoints:10 * njoints + 2 * njoints] = np.array(inertia_model.damping)
    else:
        for j in range(njoints):
            phi0[10 * njoints + j * 4] = inertia_model.coulomb[j]      # Fc+
            phi0[10 * njoints + j * 4 + 1] = inertia_model.damping[j]  # Fv+
            phi0[10 * njoints + j * 4 + 2] = inertia_model.coulomb[j]  # Fc-
            phi0[10 * njoints + j * 4 + 3] = inertia_model.damping[j]  # Fv-
    logger.info(f"Initial guess: inertia/friction from URDF, armature = 0, lambda_reg(fallback)={args.lambda_reg}")

    # Build per-parameter regularization weight vector from YAML config.
    # Layout follows the regressor columns: [inertia(10*nj) | friction | armature(nj)]
    lam_vec = build_reg_vector(njoints, args.friction_model, args.reg_config, args.lambda_reg)
    sqrt_lam = np.sqrt(lam_vec)  # precompute sqrt once; reused in linear solve and residual

    # Per-joint RMS normalization: scale each joint's rows by 1/rms(tau_j)
    # so all joints contribute equally to the loss regardless of torque magnitude
    tau_rms = np.sqrt(np.mean(tau_meas ** 2, axis=0))  # (njoints,)
    tau_rms = np.maximum(tau_rms, 1e-6)  # avoid division by zero
    logger.info(f"Per-joint torque RMS: {tau_rms}")
    tau_flat_norm = tau_flat / np.tile(tau_rms, n_samples)

    # Variable projection: optimize vbrk (nonlinear) with inner linear LS for phi
    from scipy.optimize import least_squares

    def solve_linear_given_vbrk(vbrk):
        """Given vbrk, build friction regressor and solve regularized linear LS for phi."""
        if args.friction_model == "symmetric":
            reg_friction = generateSymFrictionReg(v, vbrk=vbrk)
        else:
            reg_friction = generateAsymFrictionReg(v, vbrk=vbrk)
        regressor = np.hstack([reg_inertia, reg_friction, reg_armature])
        regressor_norm = regressor.reshape(n_samples, njoints, n_params) / tau_rms[:, None]
        regressor_norm = regressor_norm.reshape(n_samples * njoints, n_params)
        # Per-parameter Tikhonov regularization: diag(sqrt(lam)) * (phi - phi0)
        Y_aug = np.vstack([regressor_norm, np.diag(sqrt_lam)])
        tau_aug = np.concatenate([tau_flat_norm, sqrt_lam * phi0])
        phi, _, rank, sv = np.linalg.lstsq(Y_aug, tau_aug, rcond=None)
        return phi, regressor, rank, sv

    def residual_fn(vbrk_arr):
        vbrk = float(vbrk_arr[0])
        phi, regressor, _, _ = solve_linear_given_vbrk(vbrk)
        pred = regressor @ phi
        resid = (pred - tau_flat) / np.tile(tau_rms, n_samples)
        reg_resid = sqrt_lam * (phi - phi0)
        return np.concatenate([resid, reg_resid])

    # Initial QR analysis with initial vbrk
    logger.info("Building regressor and initial QR analysis...")
    phi_init, regressor_init, rank_init, sv_init = solve_linear_given_vbrk(args.vbrk_init)
    reduced_R, cond_num = QR_dim_reduction(regressor_init)
    logger.info(f"Regressor shape: {regressor_init.shape}, reduced R: {reduced_R.shape}, cond: {cond_num:.4f}")

    # Nonlinear optimization of vbrk
    logger.info(f"Solving nonlinear least squares (variable projection) for vbrk...")
    logger.info(f"  vbrk bounds: [{args.vbrk_lb}, {args.vbrk_ub}], init: {args.vbrk_init}")
    result = least_squares(
        residual_fn,
        x0=[args.vbrk_init],
        bounds=([args.vbrk_lb], [args.vbrk_ub]),
        method='trf',
        verbose=2,
    )
    vbrk_opt = float(result.x[0])
    logger.info(f"Optimal vbrk = {vbrk_opt:.8f} (vcoul = {vbrk_opt*2:.8f})")

    # Final solve with optimal vbrk
    phi, regressor, rank, sv = solve_linear_given_vbrk(vbrk_opt)
    logger.info(f"Final solution rank: {rank}, singular values range: [{sv.min():.4e}, {sv.max():.4e}]")

    # Save identified parameters (phi + vbrk) to NPZ file
    if args.save_params is not None:
        np.savez(
            args.save_params,
            phi=phi,
            vbrk=vbrk_opt,
            phi0=phi0,
            robot=args.robot,
            friction_model=args.friction_model,
            njoints=njoints,
            n_params=n_params,
        )
        logger.info(f"Saved identified parameters to: {args.save_params}")

    # Predict torque from identified params
    tau_pred = regressor @ phi
    tau_pred = tau_pred.reshape(n_samples, njoints)

    # Predict torque from nominal URDF model via RNEA (+ friction + armature)
    import pinocchio as pin
    from system_identification.utils import find_path
    urdf_file = find_path(f"{args.robot}.urdf", "./robot_description")
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

    # Print 3-way comparison: measured vs nominal vs identified
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

    # Print identified mass and CoM per joint
    n_inertia = 10 * njoints
    for j in range(njoints):
        phi_i = phi[j * 10: (j + 1) * 10]
        mass = phi_i[0]
        com = phi_i[1:4] / mass if mass > 1e-10 else np.zeros(3)
        logger.info(f"Joint {j+1}: Mass={mass:.6f}, CoM=[{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}]")

    # Print identified friction and armature parameters
    if args.friction_model == "symmetric":
        n_friction = 2 * njoints
    else:
        n_friction = 4 * njoints

    n_armature = njoints
    phi_friction = phi[n_inertia:n_inertia + n_friction]
    if args.friction_model == "symmetric":
        phi_friction = phi_friction.reshape(njoints, 2)
        for j in range(njoints):
            logger.info(f"Joint {j+1}: Coulomb={phi_friction[j,0]:.6f}, Viscous={phi_friction[j,1]:.6f}")
    else:
        phi_friction = phi_friction.reshape(njoints, 4)
        for j in range(njoints):
            logger.info(f"Joint {j+1}: Fc+={phi_friction[j,0]:.6f}, Fv+={phi_friction[j,1]:.6f}, "
                        f"Fc-={phi_friction[j,2]:.6f}, Fv-={phi_friction[j,3]:.6f}")

    phi_armature = phi[n_inertia + n_friction:n_inertia + n_friction + n_armature]
    for j in range(njoints):
        logger.info(f"Joint {j+1}: Armature={phi_armature[j]:.6f}")

    # Write identified inertia, friction, and armature back to URDF (if enabled)
    if args.output_urdf is not None:
        write_to_urdf(args.robot, phi, njoints, args.friction_model, args.output_urdf)

    # Plot measured vs nominal vs predicted torque
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
    fig.suptitle("Measured vs Nominal (RNEA) vs Identified Torque")
    plt.tight_layout()

    # ===== Validation on test_data =====
    logger.info("=" * 80)
    logger.info(f"Validating identified parameters on: {args.val_dir}")
    t_val, q_val, v_val, a_fd_val, tau_meas_val, a_cmd_val = load_data(args.val_dir)
    t_val = t_val[n:-n]; q_val = q_val[n:-n]; v_val = v_val[n:-n]
    a_fd_val = a_fd_val[n:-n]; tau_meas_val = tau_meas_val[n:-n]; a_cmd_val = a_cmd_val[n:-n]
    n_samples_val = t_val.shape[0]
    logger.info(f"Loaded {n_samples_val} validation samples, {njoints} joints")

    # Compute acceleration for validation: selectable between central-diff+SG, causal, or zero
    if args.val_acc_method == "central":
        a_val = a_fd_val
        acc_label = "Central Diff + Savgol"
    elif args.val_acc_method == "causal":
        a_val = compute_causal_acceleration(t_val, v_val)
        acc_label = "Causal Savgol"
    else:  # zero
        a_val = np.zeros_like(v_val)
        acc_label = "Zero Acceleration"
    reg_inertia_val = inertia_model.regressor(q_val, v_val, a_val)
    reg_armature_val = feature2regressor([a_val], n_samples_val, njoints)
    if args.friction_model == "symmetric":
        reg_friction_val = generateSymFrictionReg(v_val, vbrk=vbrk_opt)
    else:
        reg_friction_val = generateAsymFrictionReg(v_val, vbrk=vbrk_opt)
    regressor_val = np.hstack([reg_inertia_val, reg_friction_val, reg_armature_val])
    tau_pred_val = (regressor_val @ phi).reshape(n_samples_val, njoints)

    tau_nominal_val = np.zeros((n_samples_val, njoints))
    for i in range(n_samples_val):
        tau_nominal_val[i] = pin.rnea(nom_model, nom_data, q_val[i], v_val[i], a_val[i])
        tau_nominal_val[i] += nom_coulomb * np.tanh(v_val[i] / eps_nom) + nom_viscous * v_val[i] + nom_armature * a_val[i]

    logger.info("VALIDATION: 3-way RMS error comparison on test_data")
    logger.info(f"{'Joint':>6} | {'RMS meas':>10} | {'Nominal err':>12} {'ratio':>7} | {'Ident err':>12} {'ratio':>7}")
    logger.info("-" * 80)
    for j in range(njoints):
        rms_meas = np.sqrt(np.mean(tau_meas_val[:, j] ** 2))
        rms_nom_err = np.sqrt(np.mean((tau_meas_val[:, j] - tau_nominal_val[:, j]) ** 2))
        rms_id_err = np.sqrt(np.mean((tau_meas_val[:, j] - tau_pred_val[:, j]) ** 2))
        logger.info(f"J{j+1:>4}  | {rms_meas:10.4f} | {rms_nom_err:12.6f} {rms_nom_err/rms_meas:7.4f} | "
                    f"{rms_id_err:12.6f} {rms_id_err/rms_meas:7.4f}")
    logger.info("=" * 80)

    fig_val, axes_val = plt.subplots(njoints, 1, sharex=True, figsize=(12, 2 * njoints))
    if njoints == 1:
        axes_val = [axes_val]
    for j in range(njoints):
        ax = axes_val[j]
        ax.plot(t_val, tau_meas_val[:, j], 'b-', linewidth=1, label='Measured')
        ax.plot(t_val, tau_nominal_val[:, j], 'g-', linewidth=1, alpha=0.5, label='Nominal (RNEA)')
        ax.plot(t_val, tau_pred_val[:, j], 'r--', linewidth=1, label='Identified')
        ax.set_ylabel(f"Joint {j+1}\n(N·m)")
        if j == 0:
            ax.legend(loc='upper right', fontsize=8)
    axes_val[-1].set_xlabel("time (s)")
    fig_val.suptitle("VALIDATION (test_data): Measured vs Nominal (RNEA) vs Identified Torque")
    plt.tight_layout()

    # Plot validation acceleration: Expected (dataset) vs Causal Savgol (actually used)
    fig_acc_val, axes_acc_val = plt.subplots(njoints, 1, sharex=True, figsize=(12, 2 * njoints))
    if njoints == 1:
        axes_acc_val = [axes_acc_val]
    for j in range(njoints):
        ax = axes_acc_val[j]
        ax.plot(t_val, a_cmd_val[:, j], color='C0', linewidth=1.5, label="Expected" if j == 0 else None)
        ax.plot(t_val, a_val[:, j], color='C3', linewidth=1.0, alpha=0.7, label=acc_label if j == 0 else None)
        ax.set_ylabel(f"J{j+1}\n(rad/s²)")
    axes_acc_val[0].legend(loc='upper right', fontsize=8)
    axes_acc_val[-1].set_xlabel("time (s)")
    fig_acc_val.suptitle(f"VALIDATION (test_data): Expected Acceleration (blue) vs {acc_label} (red)")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
