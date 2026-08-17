import numpy as np
from system_identification.utils import QR_dim_reduction, feature2regressor
from loguru import logger
from system_identification.excitation_generator import (
    obtain_bounded_fourier_traj,
    obtain_fourier_traj,
    generate_random_param,
)
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

def _build_param_weights(regressor, njoints, friction_model, sysID=None, friction_weight=1.0):
    """Build per-column weight vector for weighted A/D-optimal criteria.
    
    Each joint has 10 inertial params: [0]=mass, [1,2,3]=CoM first moments, [4:10]=inertia tensor.
    Friction params are appended after inertial params.
    
    Weight strategy:
    - mass/CoM = joint_w[i], inertia tensor = 0.01 * joint_w[i] (de-emphasized)
    - friction = friction_weight (default 1.0, tunable)
    """
    n_inertia = 10 * njoints
    if friction_model == "symmetric":
        n_friction = 2 * njoints
    elif friction_model == "asymmetric":
        n_friction = 4 * njoints
    else:
        n_friction = 0
    n_total = n_inertia + n_friction
    
    w = np.ones(n_total)
    
    # 计算每个关节的质量反比权重，补偿低质量关节
    if sysID is not None and hasattr(sysID, 'masses') and len(sysID.masses) == njoints:
        masses = np.array(sysID.masses, dtype=float)
        mean_mass = np.mean(masses)
        joint_w = mean_mass / masses  # 低质量关节获得更大权重
    else:
        joint_w = np.ones(njoints)
    
    for i in range(njoints):
        base = i * 10
        w[base + 0] = joint_w[i]      # mass
        w[base + 1: base + 4] = joint_w[i]  # CoM first moments
        w[base + 4: base + 10] = 0.01 * joint_w[i]  # inertia tensor (de-emphasized but still joint-scaled)
    
    # friction columns use tunable weight
    if n_friction > 0:
        w[n_inertia:] = friction_weight
    
    return w[:regressor.shape[1]]

def params2cond(params, fourier_config, robot_config, sysID, use_bounded=False, verbose=False, soft_vel_penalty=0.0):
    order = fourier_config["order"]
    njoints = robot_config["njoints"]

    # since the params are flatten into (2 x order x njoints) to constuct
    # the constraints, here we need to reshape and transpose it to recover
    # the original shape
    params = params.reshape(2, njoints, order)
    params = np.transpose(params, (0, 2, 1))

    if use_bounded:
        t, qs, qds, qdds = obtain_bounded_fourier_traj(params, fourier_config, robot_config)
    else:
        t, qs, qds, qdds = obtain_fourier_traj(params, fourier_config, robot_config)
    regressor = sysID.regressor(qs, qds, qdds)
    reduced_R, cond_num = QR_dim_reduction(regressor)
    s = np.linalg.svd(reduced_R, compute_uv=False)
    
    # 按照公式 J₁ = k₁·cond(Y) + k₂·(1/σ_min) + k₃·σ_max 计算损失
    # k₁, k₂, k₃ 为权重系数，根据经验设置
    k1 = 1.0    # 条件数权重
    k2 = 1.0    # 最小奇异值倒数权重
    k3 = 1e-6   # 最大奇异值权重（需要较小以平衡量级）
    
    sigma_min = np.min(s)
    sigma_max = np.max(s)
    
    # 计算综合损失
    loss = k1 * cond_num + k2 * (1.0 / sigma_min) + k3 * sigma_max
    
    # Soft velocity penalty: push the global search toward boundary-friendly trajectories
    if soft_vel_penalty > 0.0:
        vel_limits = np.array(robot_config["joint_vel_limits"])
        max_vel_observed = np.max(np.abs(qds), axis=0)
        violations = np.maximum(0, max_vel_observed - vel_limits)
        # Violation ratio: punish proportional to how much we exceed the limit
        vel_penalty = soft_vel_penalty * np.sum((violations / vel_limits) ** 2)
        loss += vel_penalty
    
    if verbose:
        logger.info(f"current singular values: min={sigma_min:.4e}, max={sigma_max:.4e}")
        logger.info(f"condition number: {cond_num:.4f}, total loss: {loss:.4f}")
    return loss

def generateSymFrictionReg(dq, vbrk=0.00005):
    nq = dq.shape[1]
    ndata = dq.shape[0]
    vcoul = vbrk * 2
    feature_Fc = np.tanh(dq / vcoul)
    feature_viscous = dq
    features = [feature_Fc, feature_viscous]
    Y_reg = feature2regressor(features, ndata, nq)
    return Y_reg

def generateAsymFrictionReg(dq, vbrk=0.00005):
    nq = dq.shape[1]
    ndata = dq.shape[0]
    vcoul = vbrk * 2
    feature_Fc_pos = np.tanh(dq / vcoul) * (dq > 0)
    feature_viscous_pos = dq * (dq > 0)
    feature_Fc_neg = np.tanh(dq / vcoul) * (dq < 0)
    feature_viscous_neg = dq * (dq < 0)
    features = [feature_Fc_pos, feature_viscous_pos, feature_Fc_neg, feature_viscous_neg]
    Y_reg = feature2regressor(features, ndata, nq)
    return Y_reg

def params2condFriction(params, fourier_config, robot_config, sysID, friction_model, use_bounded=False, verbose=False, soft_vel_penalty=0.0):
    order = fourier_config["order"]
    njoints = robot_config["njoints"]

    # since the params are flatten into (2 x order x njoints) to constuct
    # the constraints, here we need to reshape and transpose it to recover
    # the original shape
    params = params.reshape(2, njoints, order)
    params = np.transpose(params, (0, 2, 1))

    if use_bounded:
        t, qs, qds, qdds = obtain_bounded_fourier_traj(params, fourier_config, robot_config)
    else:
        t, qs, qds, qdds = obtain_fourier_traj(params, fourier_config, robot_config)
    regressor = sysID.regressor(qs, qds, qdds)
    if friction_model == "symmetric":
        regressorFriction = generateSymFrictionReg(qds)
    elif friction_model == "asymmetric":
        regressorFriction = generateAsymFrictionReg(qds)
    else:
        raise ValueError("Invalid friction model")
    regressor = np.hstack([regressor, regressorFriction])
    reduced_R, cond_num = QR_dim_reduction(regressor)
    s = np.linalg.svd(reduced_R, compute_uv=False)
    
    # 按照公式 J₁ = k₁·cond(Y) + k₂·(1/σ_min) + k₃·σ_max 计算损失
    # k₁, k₂, k₃ 为权重系数，根据经验设置
    k1 = 1.0    # 条件数权重
    k2 = 1.0    # 最小奇异值倒数权重
    k3 = 1e-6   # 最大奇异值权重（需要较小以平衡量级）
    
    sigma_min = np.min(s)
    sigma_max = np.max(s)
    
    # 计算综合损失
    loss = k1 * cond_num + k2 * (1.0 / sigma_min) + k3 * sigma_max
    
    # Soft velocity penalty: push the global search toward boundary-friendly trajectories
    if soft_vel_penalty > 0.0:
        vel_limits = np.array(robot_config["joint_vel_limits"])
        max_vel_observed = np.max(np.abs(qds), axis=0)
        violations = np.maximum(0, max_vel_observed - vel_limits)
        # Violation ratio: punish proportional to how much we exceed the limit
        vel_penalty = soft_vel_penalty * np.sum((violations / vel_limits) ** 2)
        loss += vel_penalty
    
    if verbose:
        logger.info(f"current singular values: min={sigma_min:.4e}, max={sigma_max:.4e}")
        logger.info(f"condition number: {cond_num:.4f}, total loss: {loss:.4f}")
    return loss

def params2condFrictionA(params, fourier_config, robot_config, sysID, friction_model, use_bounded=False, verbose=False, soft_vel_penalty=0.0):
    """加权 A-最优准则 (inertia+friction): 最小化 trace((R^T R)^{-1})，
    对质量/质心/摩擦列权重=1.0，惯性张量列权重=0.01，集中改善关注参数的辨识精度。"""
    order = fourier_config["order"]
    njoints = robot_config["njoints"]

    params = params.reshape(2, njoints, order)
    params = np.transpose(params, (0, 2, 1))

    if use_bounded:
        t, qs, qds, qdds = obtain_bounded_fourier_traj(params, fourier_config, robot_config)
    else:
        t, qs, qds, qdds = obtain_fourier_traj(params, fourier_config, robot_config)
    regressor = sysID.regressor(qs, qds, qdds)
    if friction_model == "symmetric":
        regressorFriction = generateSymFrictionReg(qds)
    elif friction_model == "asymmetric":
        regressorFriction = generateAsymFrictionReg(qds)
    else:
        raise ValueError("Invalid friction model")
    regressor = np.hstack([regressor, regressorFriction])
    
    # 降维前对回归器列加权：质量/质心/摩擦=sqrt(1.0)，惯性张量=sqrt(0.01)
    # 低质量关节额外加权补偿数值幅度劣势
    w = _build_param_weights(regressor, njoints, friction_model, sysID, friction_weight=3.0)
    regressor_weighted = regressor * np.sqrt(w)[None, :]
    
    reduced_R, cond_num = QR_dim_reduction(regressor_weighted)
    s = np.linalg.svd(reduced_R, compute_uv=False)

    # A-最优准则: 最小化 trace((R^T R)^{-1}) = sum(1/s_i^2)
    s_safe = np.maximum(s, 1e-12)
    k_cond = 1e-3  # 条件数权重，保持 A-最优项主导
    loss = np.sum(1.0 / (s_safe ** 2)) + k_cond * cond_num

    # Soft velocity penalty
    if soft_vel_penalty > 0.0:
        vel_limits = np.array(robot_config["joint_vel_limits"])
        max_vel_observed = np.max(np.abs(qds), axis=0)
        violations = np.maximum(0, max_vel_observed - vel_limits)
        vel_penalty = soft_vel_penalty * np.sum((violations / vel_limits) ** 2)
        loss += vel_penalty

    if verbose:
        _, raw_cond = QR_dim_reduction(regressor)
        logger.info(f"current singular values (weighted): min={np.min(s):.4e}, max={np.max(s):.4e}")
        logger.info(f"weighted A-optimal loss: {loss:.4e}, weighted cond: {cond_num:.4f}, raw cond: {raw_cond:.4f}")
    return loss

def params2condFrictionD(params, fourier_config, robot_config, sysID, friction_model, use_bounded=False, verbose=False, soft_vel_penalty=0.0):
    """加权 D-最优准则 (inertia+friction): 最小化 -log(det(R^T R))，
    对质量/质心/摩擦列权重=1.0，惯性张量列权重=0.01，集中改善关注参数的辨识精度。"""
    order = fourier_config["order"]
    njoints = robot_config["njoints"]

    params = params.reshape(2, njoints, order)
    params = np.transpose(params, (0, 2, 1))

    if use_bounded:
        t, qs, qds, qdds = obtain_bounded_fourier_traj(params, fourier_config, robot_config)
    else:
        t, qs, qds, qdds = obtain_fourier_traj(params, fourier_config, robot_config)
    regressor = sysID.regressor(qs, qds, qdds)
    if friction_model == "symmetric":
        regressorFriction = generateSymFrictionReg(qds)
    elif friction_model == "asymmetric":
        regressorFriction = generateAsymFrictionReg(qds)
    else:
        raise ValueError("Invalid friction model")
    regressor = np.hstack([regressor, regressorFriction])
    
    # 降维前对回归器列加权：质量/质心/摩擦=sqrt(1.0)，惯性张量=sqrt(0.01)
    # 低质量关节额外加权补偿数值幅度劣势
    w = _build_param_weights(regressor, njoints, friction_model, sysID, friction_weight=3.0)
    regressor_weighted = regressor * np.sqrt(w)[None, :]
    
    reduced_R, cond_num = QR_dim_reduction(regressor_weighted)
    s = np.linalg.svd(reduced_R, compute_uv=False)

    # D-最优准则: 最小化 -log(det(R^T R)) = -2 * sum(log(s_i))
    s_safe = np.maximum(s, 1e-12)
    k_cond = 1e-3  # 条件数权重，保持 D-最优项主导
    loss = -2.0 * np.sum(np.log(s_safe)) + k_cond * cond_num

    # Soft velocity penalty
    if soft_vel_penalty > 0.0:
        vel_limits = np.array(robot_config["joint_vel_limits"])
        max_vel_observed = np.max(np.abs(qds), axis=0)
        violations = np.maximum(0, max_vel_observed - vel_limits)
        vel_penalty = soft_vel_penalty * np.sum((violations / vel_limits) ** 2)
        loss += vel_penalty

    if verbose:
        _, raw_cond = QR_dim_reduction(regressor)
        logger.info(f"current singular values (weighted): min={np.min(s):.4e}, max={np.max(s):.4e}")
        logger.info(f"weighted D-optimal loss: {loss:.4e}, weighted cond: {cond_num:.4f}, raw cond: {raw_cond:.4f}")
    return loss

def params2coverage(params, fourier_config, robot_config):
    order = fourier_config["order"]
    njoints = robot_config["njoints"]

    # since the params are flatten into (2 x order x njoints) to construct
    # the constraints, here we need to reshape and transpose it to recover
    # the original shape
    params = params.reshape(2, njoints, order)
    params = np.transpose(params, (0, 2, 1))

    t, qs, qds, qdds = obtain_fourier_traj(params, fourier_config, robot_config)
    qs_min = np.min(qs, axis=0)
    qs_max = np.max(qs, axis=0)
    loss = -np.sum((qs_max-qs_min)**2)
    return loss

def constraints(flatten_params, fourier_config, robot_config, optimizer):
    """
    Return constraints for parameter optimization, including
        1) Linear constraints: Initial position, velocity, accleration constraints: q(0)=init_pos, dq(0)=ddq(0)=0
        2) Nonlinear constraints: Extreme position, velocity constraints
        3) Parameter constraints

        The linear equality constraints can express as M.dot(flatten params), while M are the constant of the linear
        equality constraints, organized as a matrix, for a 2-dof robot with a fourier series trajectory of order 5,
        the position constraints matrx is
            [
        M=      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0., 1/1, 1/2., 1/3., 1/4., 1/5., 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0., 0, 0, 0, 0, 0, 1/1, 1/2., 1/3., 1/4., 1/5.],
            ]
        M.dot(params) = [
                            0,
                            0,
                        ]

    Refer to paper: Parameter Identification of the KUKA LBR iiwa Robot Including
        Constraints on Physical Feasibility
    """
    order = fourier_config["order"]
    duration = fourier_config["duration"]
    njoints = robot_config["njoints"]
    init_pos = robot_config["init_pos"]
    upperPosLimit, lowerPosLimit, velLimit = (
        robot_config["upper_joint_pos_limits"],
        robot_config["lower_joint_pos_limits"],
        robot_config["joint_vel_limits"],
    )
    upperPosLimit, lowerPosLimit, velLimit = (
        np.array(upperPosLimit),
        np.array(lowerPosLimit),
        np.array(velLimit),
    )
    
    # set to 95 of the limit to avoid violation
    bound = 0.8
    lowerPosLimit[1] = -1.0
    upperPosLimit[1] = 1.0
    upperPosLimit = upperPosLimit * bound
    lowerPosLimit = lowerPosLimit * bound
    logger.info(f"upperPosLimit: {upperPosLimit}")
    logger.info(f"lowerPosLimit: {lowerPosLimit}")
    logger.info(f"velLimit: {velLimit}")
    omega_f = 2 * np.pi / duration

    # linear constraints for initial positions, velocities and accelerations
    init_vel_constraint_mat = np.zeros((njoints, flatten_params.shape[0]))
    for i in range(njoints):
        init_vel_constraint_mat[i, i * order : (i + 1) * order] = np.ones(order)
    vel_lb = vel_ub = np.zeros(njoints)
    keep_feasible = np.array([True for i in range(njoints)])
    init_vel_constraint = LinearConstraint(
        init_vel_constraint_mat, vel_lb, vel_ub, keep_feasible=keep_feasible
    )

    start_idx = flatten_params.shape[0] // 2
    init_pos_constraint_mat = np.zeros((njoints, flatten_params.shape[0]))
    const = np.array([1 / i for i in range(1, order + 1)])
    for i in range(njoints):
        init_pos_constraint_mat[
            i, start_idx + i * order : start_idx + (i + 1) * order
        ] = const
    pos_lb = pos_ub = np.array(init_pos)
    init_pos_constraint = LinearConstraint(
        init_pos_constraint_mat, pos_lb, pos_ub, keep_feasible=keep_feasible
    )

    init_acc_constraint_mat = np.zeros((njoints, flatten_params.shape[0]))
    const = np.array([i for i in range(1, order + 1)])
    for i in range(njoints):
        init_acc_constraint_mat[
            i, start_idx + i * order : start_idx + (i + 1) * order
        ] = const
    acc_lb = acc_ub = np.zeros(njoints)
    init_acc_constraint = LinearConstraint(
        init_acc_constraint_mat, acc_lb, acc_ub, keep_feasible=keep_feasible
    )

    # nonlinear constraints for positions and velocities
    extreme_constraint_mat = np.zeros((njoints, order * njoints))
    for i in range(njoints):
        extreme_constraint_mat[i, i * order : (i + 1) * order] = np.ones(order)

    def extreme_vel_func(x):
        A = x[:start_idx]
        B = x[start_idx:]
        A2 = A**2
        B2 = B**2
        A2B2 = A2 + B2
        root_A2B2 = np.sqrt(A2B2)
        return np.dot(extreme_constraint_mat, root_A2B2)

    extreme_vel_lb = -velLimit * 0.95
    extreme_vel_ub = velLimit * 0.95 # here use a tighter bound for velocity to optimize
                                     # such that we can guarantee that after optimization
                                     # all trajectories are within the bound
    for i in range(len(extreme_vel_ub)):
        if extreme_vel_ub[i] > 10000:
            extreme_vel_ub[i] = 10000
    extreme_vel_constraint = NonlinearConstraint(
        extreme_vel_func, extreme_vel_lb, extreme_vel_ub
    )

    def extreme_pos_func(x):
        A = x[:start_idx]
        B = x[start_idx:]
        A2 = A**2
        B2 = B**2
        A2B2 = A2 + B2
        root_A2B2 = np.sqrt(A2B2)
        divider = np.array([1 / i for i in range(1, order + 1)] * njoints)
        root_A2B2_divided = root_A2B2 * divider
        return np.dot(extreme_constraint_mat, root_A2B2_divided)

    extreme_pos_lb = lowerPosLimit * omega_f
    extreme_pos_ub = upperPosLimit * omega_f
    extreme_pos_constraint = NonlinearConstraint(
        extreme_pos_func, extreme_pos_lb, extreme_pos_ub
    )

    # linear constraints for parameters
    const = np.array([i / order * omega_f for i in range(1, order + 1)] * njoints)
    const = np.hstack([const, const])

    q_max_extend = np.repeat(upperPosLimit, order)
    q_max_extend = np.hstack([q_max_extend, q_max_extend])

    dq_max_extend = np.repeat(velLimit, order)
    dq_max_extend = np.hstack([dq_max_extend, dq_max_extend])
    dq_min_extend = -dq_max_extend

    param_ub = np.minimum(const * q_max_extend, dq_max_extend)
    q_min_extend = np.repeat(lowerPosLimit, order)
    q_min_extend = np.hstack([q_min_extend, q_min_extend])
    param_lb = np.maximum(const * q_min_extend, dq_min_extend)

    keep_feasible = [True for i in range(flatten_params.shape[0])]
    A = np.identity(flatten_params.shape[0])
    extreme_param_constraint = LinearConstraint(
        A, param_lb, param_ub, keep_feasible=keep_feasible
    )

    # equality constraints for initial positions, velocities and accelerations
    init_vel_constraint_func = lambda x: np.dot(init_vel_constraint_mat, x)
    init_pos_constraint_func = lambda x: np.dot(init_pos_constraint_mat, x)
    init_acc_constraint_func = lambda x: np.dot(init_acc_constraint_mat, x)

    # inequality constraints for upper and lower bounds of positions and velocities
    upper_pos_constraint_func = lambda x: -(extreme_pos_func(x) - extreme_pos_ub)
    upper_vel_constraint_func = lambda x: -(extreme_vel_func(x) - extreme_vel_ub)
    lower_pos_constraint_func = lambda x: -extreme_pos_func(x) - extreme_pos_lb
    
    # inequality constraints for parameters from the paper, but it seems these constraints
    # are useless so we ommited them, according to the paper, I can't find the parameters
    # that satisfy with the joint limit constraints but not satisfy with the parameter
    # constraints
    # the parameter constraints are tighter bound compare with joint limit constraints
    # and it seems they are redundant compare with constraints in (14d) and (14e) in the paper
    
    extreme_param_upper_constraint_func = lambda x: -np.dot(A, x) + param_ub
    extreme_param_lower_constraint_func = lambda x: np.dot(A, x) - param_lb

    if optimizer == "SLSQP":
        # SLSQP seems to have problems with the constraints, it's not able to find a feasible solution
        cons = (
            {"type": "eq", "fun": init_vel_constraint_func},
            {"type": "eq", "fun": init_pos_constraint_func},
            {"type": "eq", "fun": init_acc_constraint_func},
            # {"type": "ineq", "fun": upper_pos_constraint_func},
            # {"type": "ineq", "fun": upper_vel_constraint_func},
            # {"type": "ineq", "fun": lower_pos_constraint_func},
            {"type": "ineq", "fun": extreme_param_upper_constraint_func},
            {"type": "ineq", "fun": extreme_param_lower_constraint_func},
        )
    elif (
        optimizer == "trust-constr"
    ):  
        cons = (
            init_pos_constraint,
            init_vel_constraint,
            init_acc_constraint,
            extreme_pos_constraint,
            extreme_vel_constraint,
            # extreme_param_constraint, # would make the optimization infeasible, it's a tighter bound compare with the joint limit constraints
        )
    return cons


def constraints_velocity_only(flatten_params, fourier_config, robot_config, optimizer, raw_limit=1.5):
    """
    Return constraints for bounded tanh trajectory optimization.
    Only velocity and acceleration constraints are enforced — position constraints
    are eliminated because the tanh mapping in BoundedOscillationGenerator guarantees
    positions stay within joint limits.

    Constraints:
        1) Linear equality: sum(a_i) = 0 per joint  (initial velocity = 0)
        2) Linear equality: sum(b_i) = 0 per joint  (initial position = center of joint limits)
        3) Nonlinear inequality: velocity bounds at sampled time points
        4) Nonlinear inequality: acceleration bounds at sampled time points
        5) Nonlinear inequality: coefficient norm bound per joint to prevent tanh saturation
           ||A_j||^2 + ||B_j||^2 <= (raw_limit / sqrt(order))^2
    """
    order = fourier_config["order"]
    duration = fourier_config["duration"]
    njoints = robot_config["njoints"]
    velLimit = np.array(robot_config["joint_vel_limits"])
    # derive acceleration limits from velocity limits (no explicit acc limit in config)
    accLimit = velLimit * 0.7
    accLimit[2:4] = velLimit[2:4] * 1.5
    accLimit[-3:] = velLimit[-3:] * 2.0
    omega_f = 2 * np.pi / duration
    start_idx = flatten_params.shape[0] // 2

    # ---- Linear equality: initial velocity = 0 ----
    # dq(0) = sum_l a_l * cos(0) - b_l * sin(0) = sum_l a_l = 0
    init_vel_mat = np.zeros((njoints, flatten_params.shape[0]))
    for i in range(njoints):
        init_vel_mat[i, i * order:(i + 1) * order] = np.ones(order)
    vel_eq_constraint = LinearConstraint(
        init_vel_mat, np.zeros(njoints), np.zeros(njoints),
        keep_feasible=np.ones(njoints, dtype=bool)
    )

    # ---- Linear equality: initial position = center of joint limits ----
    # raw(0) = sum_l b_l = 0  =>  q(0) = q_center + q_range * tanh(0) = q_center
    init_pos_mat = np.zeros((njoints, flatten_params.shape[0]))
    for i in range(njoints):
        init_pos_mat[i, start_idx + i * order:start_idx + (i + 1) * order] = np.ones(order)
    pos_eq_constraint = LinearConstraint(
        init_pos_mat, np.zeros(njoints), np.zeros(njoints),
        keep_feasible=np.ones(njoints, dtype=bool)
    )

    # ---- Nonlinear velocity & acceleration constraints (fully vectorized) ----
    n_samples = 50
    t_samples = np.linspace(0, duration, n_samples)
    wl = omega_f * np.arange(1, order + 1)  # (order,)
    sin_wl_t = np.sin(np.outer(wl, t_samples))  # (order, n_samples)
    cos_wl_t = np.cos(np.outer(wl, t_samples))  # (order, n_samples)
    vel_basis = wl[:, None] * cos_wl_t  # (order, n_samples)
    wl_sin = wl[:, None] * sin_wl_t     # (order, n_samples)
    # second derivative basis for raw_ddot
    acc_basis_a = -(wl ** 2)[:, None] * sin_wl_t  # (order, n_samples)  for A part
    acc_basis_b = -(wl ** 2)[:, None] * cos_wl_t  # (order, n_samples)  for B part

    upper_limits = np.array(robot_config["upper_joint_pos_limits"])
    lower_limits = np.array(robot_config["lower_joint_pos_limits"])
    q_range = 0.5 * (upper_limits - lower_limits) * 0.95  # (njoints,)

    def vel_constraint_func(x):
        A = x[:start_idx].reshape(njoints, order)
        B = x[start_idx:].reshape(njoints, order)
        # raw: (njoints, n_samples)
        raw = A @ sin_wl_t + B @ cos_wl_t
        # raw_dot: (njoints, n_samples)
        raw_dot = A @ vel_basis - B @ wl_sin
        sech2 = 1.0 - np.tanh(raw) ** 2
        result = q_range[:, None] * sech2 * raw_dot  # (njoints, n_samples)
        return result.flatten()

    def acc_constraint_func(x):
        A = x[:start_idx].reshape(njoints, order)
        B = x[start_idx:].reshape(njoints, order)
        raw = A @ sin_wl_t + B @ cos_wl_t
        raw_dot = A @ vel_basis - B @ wl_sin
        raw_ddot = A @ acc_basis_a + B @ acc_basis_b
        tanh_raw = np.tanh(raw)
        sech2 = 1.0 - tanh_raw ** 2
        # q_ddot = q_range * sech^2(raw) * [raw_ddot - 2*tanh(raw)*raw_dot^2]
        result = q_range[:, None] * sech2 * (raw_ddot - 2.0 * tanh_raw * (raw_dot ** 2))
        return result.flatten()

    # tighten velocity margin to 0.9 to be more conservative
    vel_margin = 0.9
    acc_margin = 0.9
    vel_lb = np.repeat(-velLimit * vel_margin, n_samples)
    vel_ub = np.repeat(velLimit * vel_margin, n_samples)
    acc_lb = np.repeat(-accLimit * acc_margin, n_samples)
    acc_ub = np.repeat(accLimit * acc_margin, n_samples)

    # ---- Coefficient norm constraint to prevent tanh saturation ----
    # ||A_j||^2 + ||B_j||^2 <= R_j^2, where R_j = raw_limit / sqrt(order)
    R_j = raw_limit / np.sqrt(order)
    R_j_sq = R_j ** 2

    def coeff_norm_func(x):
        """Return ||A_j||^2 + ||B_j||^2 per joint, shape (njoints,)"""
        A = x[:start_idx].reshape(njoints, order)
        B = x[start_idx:].reshape(njoints, order)
        return np.sum(A ** 2, axis=1) + np.sum(B ** 2, axis=1)

    coeff_lb = np.zeros(njoints)
    coeff_ub = np.full(njoints, R_j_sq)

    if optimizer == "SLSQP":
        init_vel_func = lambda x: np.dot(init_vel_mat, x)
        init_pos_func = lambda x: np.dot(init_pos_mat, x)
        vel_limit_scaled = velLimit * vel_margin
        acc_limit_scaled = accLimit * acc_margin
        vel_ineq_upper = lambda x: -(vel_constraint_func(x).reshape(njoints, n_samples).max(axis=1) - vel_limit_scaled)
        vel_ineq_lower = lambda x: -(-vel_limit_scaled - vel_constraint_func(x).reshape(njoints, n_samples).min(axis=1))
        acc_ineq_upper = lambda x: -(acc_constraint_func(x).reshape(njoints, n_samples).max(axis=1) - acc_limit_scaled)
        acc_ineq_lower = lambda x: -(-acc_limit_scaled - acc_constraint_func(x).reshape(njoints, n_samples).min(axis=1))
        coeff_ineq = lambda x: R_j_sq - coeff_norm_func(x)
        cons = (
            {"type": "eq", "fun": init_vel_func},
            {"type": "eq", "fun": init_pos_func},
            {"type": "ineq", "fun": vel_ineq_upper},
            {"type": "ineq", "fun": vel_ineq_lower},
            {"type": "ineq", "fun": acc_ineq_upper},
            {"type": "ineq", "fun": acc_ineq_lower},
            {"type": "ineq", "fun": coeff_ineq},
        )
    elif optimizer == "trust-constr":
        vel_constraint = NonlinearConstraint(vel_constraint_func, vel_lb, vel_ub)
        acc_constraint = NonlinearConstraint(acc_constraint_func, acc_lb, acc_ub)
        coeff_constraint = NonlinearConstraint(coeff_norm_func, coeff_lb, coeff_ub)
        cons = (
            vel_eq_constraint,
            pos_eq_constraint,
            vel_constraint,
            acc_constraint,
            coeff_constraint,
        )
    return cons


def run_global_coarse_search(loss_func, fourier_config, robot_config, optimizer_args, 
                             pop_size=50, target_candidates=30, loss_threshold=100.0, raw_limit=1.5, max_gen=500):
    """Run Differential Evolution to search for low-condition-number parameters.

    Continues evolving until at least `target_candidates` parameter sets achieve
    a loss below `loss_threshold`. Coefficient norm constraint is enforced via
    rejection sampling: mutated individuals violating ||A_j||^2+||B_j||^2 <= R_j^2
    are rejected.

    Args:
        target_candidates: number of candidates that must satisfy loss < loss_threshold
        raw_limit: max |raw(t)|, R_j = raw_limit / sqrt(order)
    """
    order = fourier_config["order"]
    njoints = robot_config["njoints"]
    param_dim = 2 * order * njoints
    start_idx = param_dim // 2

    # Coefficient norm bound, same as local optimization
    R_j = raw_limit / np.sqrt(order)
    R_j_sq = R_j ** 2

    def check_coeff_norm(x):
        """Return True if all joints satisfy ||A_j||^2 + ||B_j||^2 <= R_j^2"""
        A = x[:start_idx].reshape(njoints, order)
        B = x[start_idx:].reshape(njoints, order)
        norms_sq = np.sum(A ** 2, axis=1) + np.sum(B ** 2, axis=1)
        return np.all(norms_sq <= R_j_sq)

    logger.info(f"Starting Differential Evolution global search (pop={pop_size}, target_candidates={target_candidates}, threshold={loss_threshold}, raw_limit={raw_limit})...")

    # Initialize population using generate_random_param to respect initial fourier constraints (sum=0)
    population = []
    for _ in range(pop_size):
        p = generate_random_param(order, njoints)
        p = np.transpose(p, (0, 2, 1)).flatten()
        population.append(p)
    population = np.array(population)

    # Evaluate initial population
    losses = np.array([loss_func(ind, **optimizer_args) for ind in population])

    candidate_pool = []
    candidate_losses = []

    # Helper to collect candidates
    def collect_candidates(pop, pop_losses):
        for ind, l in zip(pop, pop_losses):
            if l < loss_threshold:
                # Avoid duplicates
                if not any(np.allclose(ind, c, atol=1e-3) for c in candidate_pool):
                    candidate_pool.append(ind.copy())
                    candidate_losses.append(l)

    collect_candidates(population, losses)

    # DE parameters
    F = 0.8  # mutation factor
    CR = 0.9  # crossover probability

    gen = 0

    while len(candidate_pool) < target_candidates and gen < max_gen:
        gen += 1
        for i in range(pop_size):
            # 1. Mutation: select 3 random distinct individuals different from i
            candidates = [idx for idx in range(pop_size) if idx != i]
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            # Mutated vector
            mutated = population[r1] + F * (population[r2] - population[r3])
            
            # 2. Crossover
            cross_points = np.random.rand(param_dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, param_dim)] = True
            
            trial = np.where(cross_points, mutated, population[i])
            
            # Check coefficient norm constraint, reject if violated
            if not check_coeff_norm(trial):
                continue
            
            # 3. Selection
            trial_loss = loss_func(trial, **optimizer_args)
            if trial_loss < losses[i]:
                population[i] = trial
                losses[i] = trial_loss

        collect_candidates(population, losses)
        best_idx = np.argmin(losses)
        logger.info(f"DE Gen {gen}: Loss = {losses[best_idx]:.4f}, Candidates collected = {len(candidate_pool)}/{target_candidates}")

    if len(candidate_pool) < target_candidates:
        logger.warning(f"Reached max generations ({max_gen}) with only {len(candidate_pool)}/{target_candidates} candidates. "
                       f"Consider lowering --coarse_threshold or increasing --coarse_pop.")
        # Fill remaining with best individuals from final population
        sorted_indices = np.argsort(losses)
        for idx in sorted_indices:
            if len(candidate_pool) >= target_candidates:
                break
            ind = population[idx]
            if not any(np.allclose(ind, c, atol=1e-3) for c in candidate_pool):
                candidate_pool.append(ind.copy())
                candidate_losses.append(losses[idx])

    logger.info(f"Global coarse search complete: collected {len(candidate_pool)} candidates in {gen} generations.")
    return candidate_pool, candidate_losses