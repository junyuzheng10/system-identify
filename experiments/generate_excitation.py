import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from system_identification.excitation_generator import (
    obtain_bounded_fourier_traj,
    obtain_fourier_traj,
    is_traj_valid,
)
from scipy.optimize import minimize
from system_identification.excitation_optimization import (
    params2cond, constraints, constraints_velocity_only,
    params2coverage, params2condFriction,
    params2condFrictionA, params2condFrictionD,
    generateSymFrictionReg, generateAsymFrictionReg,
    run_global_coarse_search,
)
from system_identification.utils import QR_dim_reduction, retrieve_robot_config
from system_identification.inertia_model import InertiaModel
from datetime import datetime
from pathlib import Path
import time


class TimeLimitCallback:
    """Callback to stop optimization after a time limit, storing the latest xk."""
    def __init__(self, max_seconds):
        self.max_seconds = max_seconds
        self.start_time = None
        self.latest_x = None

    def __call__(self, xk, *args, **kwargs):
        self.latest_x = np.copy(xk)
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            logger.info(f"Time limit reached ({elapsed:.1f}s > {self.max_seconds}s), stopping optimization.")
            raise StopIteration

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulator")
    parser.add_argument("--excite_type", type=str, default="condFriction")
    parser.add_argument("--robot", type=str, default="ar5_leftArm")
    parser.add_argument("--optimizer", type=str, default="trust-constr") # SLSQP, trust-constr
    parser.add_argument("--fourier_order", type=int, default=8)
    parser.add_argument("--fourier_duration", type=int, default=40)
    parser.add_argument("--friction_model", type=str, default="symmetric") # symmetric, asymmetric
    parser.add_argument("--use_bounded", action="store_true", default=True, help="Use tanh mapping to bound positions within joint limits")
    parser.add_argument("--coarse_threshold", type=float, default=80.0, help="Loss threshold for collecting candidates in coarse search")
    parser.add_argument("--coarse_pop", type=int, default=50, help="Population size for Differential Evolution")
    parser.add_argument("--coarse_gen", type=int, default=2, help="Max generations for Differential Evolution")
    parser.add_argument("--coarse_max_gen", type=int, default=3000, help="Hard generation limit for Differential Evolution")
    parser.add_argument("--soft_vel_penalty", type=float, default=5.0, help="Soft velocity penalty weight for global coarse search (0 to disable)")
    parser.add_argument("--max_local_time", type=int, default=180, help="Max seconds for local fine-tuning (0 = no limit)")
    parser.add_argument("--constr_penalty", type=float, default=10.0, help="Initial constraint penalty for trust-constr: higher = stricter constraint enforcement")
    parser.add_argument("--raw_limit", type=float, default=5.0, help="Max |raw(t)| to prevent tanh saturation (R_j = raw_limit / sqrt(order))")
    args = parser.parse_args()

    # setup gym env
    robot_config = retrieve_robot_config(args.robot)
    njoints = robot_config["njoints"]
    friction_model = args.friction_model

    logger.info("=== Joint limits ===")
    for i in range(njoints):
        logger.info(
            f"Joint {i+1}: "
            f"pos=[{robot_config['lower_joint_pos_limits'][i]:.4f}, "
            f"{robot_config['upper_joint_pos_limits'][i]:.4f}] rad, "
            f"vel_limit={robot_config['joint_vel_limits'][i]:.4f} rad/s"
        )
    # ===================================================================================
    # Generate exciting trajectories
    # ===================================================================================
    # parameterize trajectories as fourier series
    fourier_config = {"order": args.fourier_order, "duration": args.fourier_duration}

    # TODO: select loss and setting up parameters
    if args.excite_type == "cond":
        loss = params2cond
        inertia_model = InertiaModel(args.robot)
        optimizer_args = {"fourier_config": fourier_config, "robot_config": robot_config, "sysID": inertia_model, "use_bounded": args.use_bounded, "verbose": True}
        optimizer_input_args = (fourier_config, robot_config, inertia_model, args.use_bounded, True)
        optimize_options = {"maxiter": 100, "disp": True, "initial_constr_penalty": args.constr_penalty}
    elif args.excite_type == "coverage":
        loss = params2coverage
        optimize_options = {"maxiter": 100, "disp": True, "initial_constr_penalty": args.constr_penalty}
        optimizer_args = {"fourier_config": fourier_config, "robot_config": robot_config}
        optimizer_input_args = (fourier_config, robot_config)
    elif args.excite_type == "condFriction":
        loss = params2condFriction
        inertia_model = InertiaModel(args.robot)
        optimizer_args = {"fourier_config": fourier_config, "robot_config": robot_config, "sysID": inertia_model, "friction_model": friction_model, "use_bounded": args.use_bounded, "verbose": True}
        optimizer_input_args = (fourier_config, robot_config, inertia_model, friction_model, args.use_bounded, True)
        optimize_options = {"maxiter": 300, "disp": True, "initial_constr_penalty": args.constr_penalty}
    elif args.excite_type == "condFrictionA":
        loss = params2condFrictionA
        inertia_model = InertiaModel(args.robot)
        optimizer_args = {"fourier_config": fourier_config, "robot_config": robot_config, "sysID": inertia_model, "friction_model": friction_model, "use_bounded": args.use_bounded, "verbose": True}
        optimizer_input_args = (fourier_config, robot_config, inertia_model, friction_model, args.use_bounded, True)
        optimize_options = {"maxiter": 300, "disp": True, "initial_constr_penalty": args.constr_penalty}
    elif args.excite_type == "condFrictionD":
        loss = params2condFrictionD
        inertia_model = InertiaModel(args.robot)
        optimizer_args = {"fourier_config": fourier_config, "robot_config": robot_config, "sysID": inertia_model, "friction_model": friction_model, "use_bounded": args.use_bounded, "verbose": True}
        optimizer_input_args = (fourier_config, robot_config, inertia_model, friction_model, args.use_bounded, True)
        optimize_options = {"maxiter": 300, "disp": True, "initial_constr_penalty": args.constr_penalty}

    # TODO: starting optimization
    logger.info("=== Phase 1: Global Coarse Search (Differential Evolution) ===")
    # Use silent args for coarse search (no verbose logging, with soft velocity penalty)
    coarse_optimizer_args = dict(optimizer_args)
    coarse_optimizer_args["verbose"] = False
    coarse_optimizer_args["soft_vel_penalty"] = args.soft_vel_penalty
    candidates, candidate_losses = run_global_coarse_search(
        loss_func=loss,
        fourier_config=fourier_config,
        robot_config=robot_config,
        optimizer_args=coarse_optimizer_args,
        pop_size=args.coarse_pop,
        target_candidates=args.coarse_gen,
        loss_threshold=args.coarse_threshold,
        raw_limit=args.raw_limit,
        max_gen=args.coarse_max_gen
    )
    
    logger.info(f"=== Phase 2: Selecting Best Candidate from Pool (Size: {len(candidates)}) ===")
    # Evaluate boundary violation for each candidate to select the most boundary-friendly one
    vel_limits = np.array(robot_config["joint_vel_limits"])
    best_candidate = None
    min_violation = np.inf
    best_candidate_loss = np.inf

    for idx, cand in enumerate(candidates):
        # Reshape candidate to generate trajectory
        cand_reshaped = cand.reshape(2, njoints, fourier_config["order"])
        cand_reshaped = np.transpose(cand_reshaped, (0, 2, 1))
        if args.use_bounded:
            _, _, qds, _ = obtain_bounded_fourier_traj(cand_reshaped, fourier_config, robot_config)
        else:
            _, _, qds, _ = obtain_fourier_traj(cand_reshaped, fourier_config, robot_config)
        
        # Calculate max velocity violation ratio across all joints and time steps
        # Violation ratio = (max_observed_vel - limit) / limit
        max_vel_observed = np.max(np.abs(qds), axis=0)
        violations = (max_vel_observed - vel_limits) / vel_limits
        max_violation = np.max(violations)
        
        logger.info(f"Candidate {idx+1:2d}: Loss = {candidate_losses[idx]:.4f}, Max Vel Violation Ratio = {max_violation:.4f}")
        
        # We want to find the candidate that violates the boundary the least.
        # If multiple candidates have no violation (<= 0), we choose the one with the lowest loss.
        if max_violation < min_violation:
            min_violation = max_violation
            best_candidate = cand
            best_candidate_loss = candidate_losses[idx]
        elif max_violation <= 0 and min_violation <= 0:
            if candidate_losses[idx] < best_candidate_loss:
                min_violation = max_violation
                best_candidate = cand
                best_candidate_loss = candidate_losses[idx]

    logger.info(f"Selected Candidate with Max Vel Violation Ratio = {min_violation:.4f}, Loss = {best_candidate_loss:.4f}")
    init_params = best_candidate

    # Generate coarse trajectory for plotting comparison
    _coarse_params = init_params.reshape(2, njoints, fourier_config["order"])
    _coarse_params = np.transpose(_coarse_params, (0, 2, 1))
    if args.use_bounded:
        t_coarse, qs_coarse, qds_coarse, qdds_coarse = obtain_bounded_fourier_traj(_coarse_params, fourier_config, robot_config)
    else:
        t_coarse, qs_coarse, qds_coarse, qdds_coarse = obtain_fourier_traj(_coarse_params, fourier_config, robot_config)

    if args.use_bounded:
        # tanh mapping guarantees position bounds, only keep velocity constraints
        cons = constraints_velocity_only(init_params, fourier_config, robot_config, args.optimizer, raw_limit=args.raw_limit)
    else:
        cons = constraints(init_params, fourier_config, robot_config, args.optimizer)
    loss_init = loss(init_params, **optimizer_args)
    logger.info(
        f"Loss before local optimization: {loss_init}"
    )
    
    logger.info("=== Phase 3: Local Fine-Tuning (Scipy Minimize) ===")
    time1 = time.time()
    callback = TimeLimitCallback(args.max_local_time) if args.max_local_time > 0 else None
    res = None
    try:
        res = minimize(
            loss,
            init_params,
            args=optimizer_input_args,
            method=args.optimizer,
            constraints=cons,
            options=optimize_options,
            tol=1e-3,
            callback=callback
        )
    except StopIteration:
        res = callback.latest_x
        logger.info("Optimization stopped by time limit, using best result so far.")
    # If res is a plain array (time limit reached), wrap it for .x access
    if isinstance(res, np.ndarray):
        res = type('OptimizeResult', (), {'x': res})()
    time2 = time.time()
    logger.info(f"Time cost: {time2 - time1}")
    logger.info(
        f"Loss before optimization {loss_init}; Loss after optimization: {loss(res.x, **optimizer_args)}"
    )
    # generate joint trajectories parametrized by fourier basis. Since the
    # params are flatten into (2 x order x njoints) to constuct the constraints,
    # here we need to reshape and transpose it to recover the original shape
    params = res.x.reshape(2, njoints, fourier_config["order"])
    params = np.transpose(params, (0, 2, 1))
    if args.use_bounded:
        t, qs, qds, qdds = obtain_bounded_fourier_traj(params, fourier_config, robot_config, fps=500)
    else:
        t, qs, qds, qdds = obtain_fourier_traj(params, fourier_config, robot_config, fps=500)
    
    save = is_traj_valid(qs, qds, qdds, robot_config, safety_gain=2.0)
    logger.info(
        f"Trajectories vidality: {save}"
    )
    # Plot optimized trajectory (solid) and coarse search trajectory (light dashed) for comparison
    traj_opt = [qs, qds, qdds]
    traj_coarse = [qs_coarse, qds_coarse, qdds_coarse]
    traj_names = ["q", "qd", "qdd"]
    colors = ["C0", "C1", "C2"]
    fig, axes = plt.subplots(njoints, sharex=True)
    for i in range(njoints):
        ax = axes[i] if njoints != 1 else axes
        for j in range(3):
            # Coarse search trajectory in light color
            ax.plot(t_coarse, traj_coarse[j][:, i], color=colors[j], linewidth=1, alpha=0.3,
                    label=f"{traj_names[j]} (coarse)" if i == 0 else None)
            # Optimized trajectory in normal color
            ax.plot(t, traj_opt[j][:, i], color=colors[j], linewidth=1.5,
                    label=traj_names[j] if i == 0 else None)
        ax.set_ylabel(f"Joint {i+1}")
    plt.xlabel("time")
    if njoints != 1:
        fig.align_ylabels(axes[:])
        axes[0].legend(loc='upper right')
    else:
        fig.align_ylabels(axes)
        axes.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    if args.excite_type == "cond":
        # compute regressor and do dimensionality reduction
        regressor = inertia_model.regressor(qs, qds, qdds)
        reduced_R, cond_num = QR_dim_reduction(regressor)
        logger.info(f"condition number: {cond_num}")
    elif args.excite_type in ("condFriction", "condFrictionA", "condFrictionD"):
        regressor = inertia_model.regressor(qs, qds, qdds)
        if friction_model == "symmetric":
            regressorFriction = generateSymFrictionReg(qds)
        elif friction_model == "asymmetric":
            regressorFriction = generateAsymFrictionReg(qds)
        regressor = np.hstack((regressor, regressorFriction))
        reduced_R, cond_num = QR_dim_reduction(regressor)
        logger.info(f"condition number: {cond_num}")

    # save trajectories
    if save:
        _time = datetime.now().strftime("%d%m%Y%H%M%S")
        _dir = f"./traj_data"
        import csv
        _csv_dir = Path(_dir) / "csv_data"
        _csv_dir.mkdir(parents=True, exist_ok=True)
        _csv_path = _csv_dir / f"{args.excite_type}_{args.robot}_{_time}.csv"
        _header = ["t"]
        _header += [f"q{i+1}" for i in range(njoints)]
        _header += [f"v{i+1}" for i in range(njoints)]
        _header += [f"a{i+1}" for i in range(njoints)]
        with _csv_path.open("w", newline="") as _f:
            _w = csv.writer(_f)
            _w.writerow(_header)
            for _i in range(t.shape[0]):
                _row = [round(float(t[_i]), 3)]
                _row.extend([round(float(v), 6) for v in qs[_i]])
                _row.extend([round(float(v), 6) for v in qds[_i]])
                _row.extend([round(float(v), 6) for v in qdds[_i]])
                _w.writerow(_row)
        logger.info(f"Saved CSV: {_csv_path} ({t.shape[0]} samples, {njoints} joints)")
