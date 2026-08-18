import os
import numpy as np
import scipy
from copy import copy
from scipy.signal import savgol_filter, savgol_coeffs
from loguru import logger


def find_path(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
    raise Exception(f"Can't find {name} in directory {path}!")


def QR_dim_reduction(A):
    """
    Perform Reduced QR decomposition, and remove the columns with small diagonal elements in R
    The resulted matrix has the same rank as the original matrix, the return include the reduced
    R matrix and the condition number of the reduced R matrix
    """
    q, r = scipy.linalg.qr(A, mode='economic')
    M, N = A.shape[0], A.shape[1]
    rank = np.linalg.matrix_rank(r)

    diag_r = abs(np.diagonal(r))
    cols = np.arange(N)
    f = lambda x: diag_r[x]
    cols = sorted(cols, key=f)
    del_cols = cols[: N - rank]
    del_cols = sorted(del_cols)

    r_reduced = copy(r)
    r_reduced = np.delete(r_reduced, del_cols, 1)

    cond = np.linalg.cond(r_reduced)
    return r_reduced, cond


def feature2regressor(list_of_features, n_datapoints, njoints):
    """
    Transfer a list of features to regressor
        features: (n_datapoints, njoints)
        regressor: (n_datapoints*njoints, njoints)
    """
    check_features(list_of_features, n_datapoints, njoints)
    list_of_features = [
        np.repeat(feature, njoints, axis=0) for feature in list_of_features
    ]
    identities_filter = []
    for i in range(n_datapoints):
        identities_filter.append(np.identity(njoints))
    identities_filter = np.vstack(identities_filter)
    list_of_regressor = [feature * identities_filter for feature in list_of_features]
    regressor = np.hstack(list_of_regressor)
    return regressor


def check_features(list_of_features, n_datapoints, njoints):
    for i in range(len(list_of_features)):
        feature = list_of_features[i]
        assert (
            feature.shape[0] == n_datapoints and feature.shape[1] == njoints
        ), f"Feature{i} is not compatible with the given dataset."


def retrieve_geo_fromCAD(robot_name=None):
    import pinocchio as pin
    if robot_name is not None:
        urdf_file = f"{robot_name}.urdf"
    else:
        urdf_file = f"iiwas14_cad.urdf"
    urdf_file = find_path(urdf_file, "./robot_description")
    model = pin.buildModelFromUrdf(urdf_file)
    njoints = model.njoints - 1
    system_inertia = model.inertias.tolist()[1 : 1 + njoints]
    GoMs_lever = []
    masses_CAD = []
    inertias_CAD = []
    for i in range(njoints):
        GoMs_lever.append(system_inertia[i].lever)
        masses_CAD.append(system_inertia[i].mass)
        inertias_CAD.append(system_inertia[i].inertia)
    return masses_CAD, GoMs_lever, inertias_CAD


def pin_joint_config(robot_name):
    import pinocchio as pin
    urdf_file = f"{robot_name}.urdf"
    urdf_file = find_path(urdf_file, "./robot_description")
    model = pin.buildModelFromUrdf(urdf_file)
    njoints = model.njoints - 1
    joint_configs = model.nqs[1:]
    idx_continuous_joint = []
    for i in range(njoints):
        if joint_configs[i] > 1:
            idx_continuous_joint.append(i)

    upper_joint_pos_limits = []
    lower_joint_pos_limits = []
    for joint_idx in range(njoints):
        if joint_idx in idx_continuous_joint:
            upper_pos_limit = np.pi * 4
            lower_pos_limit = -np.pi * 4
        else:
            upper_pos_limit = model.upperPositionLimit[joint_idx]
            lower_pos_limit = model.lowerPositionLimit[joint_idx]
            # URDF 中某些关节的 lower/upper 标签可能写反（lower > upper），
            # 此时报警告并交换两者，保证 lower <= upper，否则 is_traj_valid 对任何 q 都会失败。
            if lower_pos_limit > upper_pos_limit:
                logger.warning(
                    f"joint {joint_idx + 1}: lower({lower_pos_limit}) > upper({upper_pos_limit}), "
                    "自动交换 lower/upper。请检查 URDF 中该关节的 limit 标签。"
                )
                lower_pos_limit, upper_pos_limit = upper_pos_limit, lower_pos_limit
        upper_joint_pos_limits.append(upper_pos_limit)
        lower_joint_pos_limits.append(lower_pos_limit)
    joint_vel_limits = model.velocityLimit
    return njoints, upper_joint_pos_limits, lower_joint_pos_limits, joint_vel_limits


def retrieve_robot_config(robot_name):
    (
        njoints,
        upper_joint_pos_limits,
        lower_joint_pos_limits,
        joint_vel_limits,
    ) = pin_joint_config(robot_name)
    init_pos = (
        np.array(upper_joint_pos_limits) + np.array(lower_joint_pos_limits)
    ) / 2.0
    init_vel = np.zeros(shape=len(upper_joint_pos_limits))
    config = {
        "njoints": njoints,
        "upper_joint_pos_limits": upper_joint_pos_limits,
        "lower_joint_pos_limits": lower_joint_pos_limits,
        "joint_vel_limits": joint_vel_limits,
        "init_pos": init_pos,
        "init_vel": init_vel,
    }
    return config


def savgol_filter_acceleration(a, window_length=21, polyorder=3):
    """Apply Savitzky-Golay filter to acceleration data.

    Args:
        a: acceleration array, shape (n_samples, njoints)
        window_length: odd integer, length of the filter window
        polyorder: order of the polynomial used to fit the samples
    Returns:
        Filtered acceleration array, same shape as input
    """
    if a.ndim == 1:
        return savgol_filter(a, window_length, polyorder)
    return np.column_stack([savgol_filter(a[:, j], window_length, polyorder) for j in range(a.shape[1])])


def causal_savgol_smooth(a, window_length=801, polyorder=5):
    """Apply causal Savitzky-Golay smoothing (only past and current samples).

    Uses savgol_coeffs with pos=window_length-1 so the filter window only
    spans [i-window_length+1, i], i.e. no future samples are used.

    Args:
        a: acceleration array, shape (n_samples, njoints)
        window_length: odd integer, length of the filter window
        polyorder: order of the polynomial used to fit the samples
    Returns:
        Filtered acceleration array, same shape as input
    """
    coeffs = savgol_coeffs(window_length, polyorder, pos=window_length - 1, use='dot')
    if a.ndim == 1:
        xp = np.pad(a, (window_length - 1, 0), mode='edge')
        return np.convolve(xp, coeffs[::-1], mode='valid')[:len(a)]
    return np.column_stack([
        causal_savgol_smooth(a[:, j], window_length, polyorder)
        for j in range(a.shape[1])
    ])


def compute_causal_acceleration(t, v, savgol_window=201, savgol_poly=5):
    """Compute acceleration using causal backward difference + causal Savitzky-Golay smoothing.

    No future samples are used. Returns the smoothed acceleration array.

    Args:
        t: time array, shape (n_samples,)
        v: velocity array, shape (n_samples, njoints)
        savgol_window: odd integer, length of the filter window
        savgol_poly: order of the polynomial used to fit the samples
    Returns:
        Filtered acceleration array, shape (n_samples, njoints)
    """
    dt = np.diff(t)  # shape (n-1,)

    # 1st-order backward difference: a[k] = (v[k] - v[k-1]) / (t[k] - t[k-1])
    a_backward = np.zeros_like(v)
    a_backward[1:] = (v[1:] - v[:-1]) / dt[:, None]

    # Causal Savitzky-Golay: smooth backward diff using only past + current samples
    a_causal = causal_savgol_smooth(a_backward, window_length=savgol_window, polyorder=savgol_poly)
    return a_causal


def skew_symmetric(vec):
    if type(vec) == type([]):
        vec = np.array(vec)
    return np.array(
        [[0.0, -vec[2], vec[1]], [vec[2], 0.0, -vec[0]], [-vec[1], vec[0], 0]]
    )


def inertiaVecToPinertia(pi_dyn_i):
    Ibar = np.array(
        [
            [pi_dyn_i[4], pi_dyn_i[5], pi_dyn_i[7]],
            [pi_dyn_i[5], pi_dyn_i[6], pi_dyn_i[8]],
            [pi_dyn_i[7], pi_dyn_i[8], pi_dyn_i[9]],
        ]
    )
    h = np.array([pi_dyn_i[1], pi_dyn_i[2], pi_dyn_i[3]])
    m = pi_dyn_i[0]
    eye3 = np.eye(3)
    Sigma = 1 / 2 * np.trace(Ibar) * eye3 - Ibar
    tmp1 = np.hstack([Sigma, np.reshape(h, (3, 1))])
    tmp2 = np.hstack([h, m])
    J = np.vstack([tmp1, tmp2])
    return J


def inertiaVecToIcQs(pi_dyn_i):
    Ibar = np.array(
        [
            [pi_dyn_i[4], pi_dyn_i[5], pi_dyn_i[7]],
            [pi_dyn_i[5], pi_dyn_i[6], pi_dyn_i[8]],
            [pi_dyn_i[7], pi_dyn_i[8], pi_dyn_i[9]],
        ]
    )
    h = np.array([pi_dyn_i[1], pi_dyn_i[2], pi_dyn_i[3]])
    m = pi_dyn_i[0]
    c = h / m
    Sc = skew_symmetric(c)
    Ic = Ibar - m * Sc @ Sc.T

    eye3 = np.eye(3)
    SigmaC = 1 / 2 * np.trace(Ic) * eye3 - Ic
    Qs = SigmaC / m
    return Ic, Qs


def inertiaVecToQ(pi_dyn_i):
    h = np.array([pi_dyn_i[1], pi_dyn_i[2], pi_dyn_i[3]])
    m = pi_dyn_i[0]
    c = h / m
    c = c.reshape(-1, 1)

    _, Qs = inertiaVecToIcQs(pi_dyn_i)
    Qsinv = np.linalg.inv(Qs)
    QsinvTXs = Qsinv.T @ c
    XsTQsinvXs = c.T @ Qsinv @ c

    Q_col1 = np.hstack([-Qsinv, QsinvTXs])
    Q_col2 = np.hstack([QsinvTXs.T, 1 - XsTQsinvXs])
    Q = np.vstack([Q_col1, Q_col2])
    return Q
