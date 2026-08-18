import numpy as np


def generate_constraints(order: int):
    r"""
    Trajectories are parameterized as
        q = \sum_k A_k*1/wk sin(k*wt) - B_k*1/wk cos(k*wt)
        qd = \sum_k A_k cos(k*wt) + B_k sin(k*wt)
        qdd = -\sum_k A_k*wk sin(k*wt) + B_k*wk cos(k*wt)

    Initial position, velocity, acceleration constraints: q(0) = qd(0) = qdd(0) = 0
        cA:  sum_k A_1 + ... + A_k + ... + A_L = 0
        cB1: sum_k B_1/1 + ... B_k/k + ... + B_L/L = 0
        cB2: sum_k B_1*1 + ... B_k*k + ... + B_L*L = 0
    """
    cA = np.ones(order)
    cB = np.array(
        [[1 / i for i in range(1, order + 1)], [i for i in range(1, order + 1)]]
    )
    cB_reduced_echelon_form = np.array([(cB[0] - cB[1]) / (cB[0] - cB[1])[1], cB[1]])
    return cA, cB_reduced_echelon_form


def generate_random_param(order: int, njoints: int):
    """
    Generate constant terms for the Fourier basis, each joint has
    different frequencies.

    Return:
        params: shape (2, order, njoints)
    """
    params = (
        np.random.randn(2, order, njoints) * 0.05
    )

    cA, cB = generate_constraints(order)
    A, B = params[0], params[1]
    A[-1] = -np.dot(cA[:-1], A[:-1])
    B[1] = -np.dot(cB[0][2:], B[2:])
    B[0] = -np.dot(cB[1][1:], B[1:])
    params = np.array([A, B])
    return params


def generate_fourier_traj(
    order, duration, njoints, params, init_pos=None, init_vel=None, fps=100
):
    if init_pos is None:
        init_pos = np.zeros(njoints)
    if init_vel is None:
        init_vel = np.zeros(njoints)
    step_size = 1 / fps

    omega_f = 2 * np.pi / duration
    num_samples = int(duration / step_size)
    t = np.linspace(0, duration, num_samples+1)
    q = np.zeros((t.shape[0], njoints)) + init_pos
    dq = np.zeros_like(q) + init_vel
    ddq = np.zeros_like(q)
    A, B = params[0], params[1]

    for k in range(1, order + 1):
        q += np.outer(np.sin(omega_f * k * t), A[k - 1] / (omega_f * k)) - np.outer(
            np.cos(omega_f * k * t), B[k - 1] / (omega_f * k)
        )
        dq += np.outer(np.cos(omega_f * k * t), A[k - 1]) + np.outer(
            np.sin(omega_f * k * t), B[k - 1]
        )
        ddq += -np.outer(np.sin(omega_f * k * t), A[k - 1] * (omega_f * k)) + np.outer(
            np.cos(omega_f * k * t), B[k - 1] * (omega_f * k)
        )
    return t, q, dq, ddq


def is_traj_valid(q, dq, ddq, robot_config, skip_pos=False, safety_gain=1.0):
    """Check if trajectory is within joint limits.

    Args:
        safety_gain: gain factor for tighter limits. gain=1 uses full limits
    """
    upperPosLimit, lowerPosLimit, velLimit = (
        robot_config["upper_joint_pos_limits"],
        robot_config["lower_joint_pos_limits"],
        robot_config["joint_vel_limits"],
    )
    if safety_gain != 1.0:
        velLimit = velLimit * safety_gain
    for q_i, dq_i, ddq_i in zip(q, dq, ddq):
        if not skip_pos:
            for j in range(len(q_i)):
                if q_i[j] > upperPosLimit[j]:
                    return False
                elif q_i[j] < lowerPosLimit[j]:
                    return False
        if np.any(np.abs(dq_i) - velLimit > 0):
            return False
    return True


def obtain_fourier_traj(params, fourier_config, robot_config):
    order = fourier_config["order"]
    duration = fourier_config["duration"]
    njoints = robot_config["njoints"]
    init_pos = robot_config["init_pos"]
    init_vel = robot_config["init_vel"]
    t, q, dq, ddq = generate_fourier_traj(
        order, duration, njoints, params, init_pos, init_vel
    )
    return t, q, dq, ddq


def obtain_bounded_fourier_traj(params, fourier_config, robot_config, fps=10):
    """Generate trajectories using tanh mapping to guarantee joint limits.

    Fully vectorized: no per-joint or per-time-step Python loops.

    Args:
        params: shape (2, order, njoints), params[0]=a (sin coeffs), params[1]=b (cos coeffs)
        fourier_config: dict with 'order' and 'duration'
        robot_config: dict with joint limits and init_pos
        fps: samples per second
    Returns:
        t, q, dq, ddq arrays
    """
    order = fourier_config["order"]
    duration = fourier_config["duration"]
    njoints = robot_config["njoints"]
    upper_limits = np.array(robot_config["upper_joint_pos_limits"])
    lower_limits = np.array(robot_config["lower_joint_pos_limits"])
    omega_f = 2 * np.pi / duration

    A = params[0]  # (order, njoints) sin coefficients
    B = params[1]  # (order, njoints) cos coefficients

    step_size = 1.0 / fps
    num_samples = int(duration / step_size)
    t = np.linspace(0, duration, num_samples + 1)
    n_time = len(t)

    # Pre-compute all sin/cos basis: shape (order, n_time)
    wl = omega_f * np.arange(1, order + 1)  # (order,)
    wt = np.outer(wl, t)                     # (order, n_time)
    sin_vals = np.sin(wt)                    # (order, n_time)
    cos_vals = np.cos(wt)                    # (order, n_time)

    # raw(t) for all joints, all time steps: shape (n_time, njoints)
    raw = (A.T @ sin_vals + B.T @ cos_vals).T

    # raw_dot(t): shape (n_time, njoints)
    wl_cos = wl[:, None] * cos_vals   # (order, n_time)
    wl_sin = wl[:, None] * sin_vals   # (order, n_time)
    raw_dot = (A.T @ wl_cos - B.T @ wl_sin).T

    # raw_ddot(t): shape (n_time, njoints)
    wl2_sin = (wl**2)[:, None] * sin_vals  # (order, n_time)
    wl2_cos = (wl**2)[:, None] * cos_vals  # (order, n_time)
    raw_ddot = (-A.T @ wl2_sin - B.T @ wl2_cos).T

    # tanh mapping for all joints at once
    q_center = 0.5 * (lower_limits + upper_limits)   # (njoints,)
    half_range = 0.5 * (upper_limits - lower_limits)  # (njoints,)
    q_range = half_range * 0.85                       # (njoints,)

    th = np.tanh(raw)           # (n_time, njoints)
    sech2 = 1.0 - th ** 2       # (n_time, njoints)

    q = q_center + q_range * th
    dq = q_range * sech2 * raw_dot
    ddq = q_range * (sech2 * raw_ddot - 2.0 * th * sech2 * raw_dot ** 2)

    return t, q, dq, ddq
