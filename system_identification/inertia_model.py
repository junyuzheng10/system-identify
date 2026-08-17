import numpy as np
import pinocchio as pin
from system_identification.utils import find_path, retrieve_geo_fromCAD, \
    inertiaVecToPinertia, inertiaVecToIcQs, inertiaVecToQ


class InertiaModel(object):
    def __init__(self, robot_name, **kwargs):
        self.name = "inertiaCoM"
        urdf_file = f"{robot_name}.urdf"
        urdf_file = find_path(urdf_file, "./robot_description")
        self._friction = kwargs.get("friction", False)

        self.model = pin.buildModelFromUrdf(urdf_file)
        self.data = self.model.createData()
        self.njoints = self.model.njoints - 1
        system_inertia = self.model.inertias.tolist()[1 : 1 + self.njoints]
        self.damping = self.model.damping
        self.coulomb = self.model.friction
        self._gravity = np.array(self.model.gravity.linear, copy=True)
        self.total_mass = 0
        self.iden_masses = []
        self.masses = []
        self.CoMs_lever = []
        self.inertia_CoM = []

        self.idx_continuous_joint = []
        joint_configs = self.model.nqs[1:]
        for i in range(self.njoints):
            if joint_configs[i] > 1:
                self.idx_continuous_joint.append(i)
        self._joint_limit()

        self._proj = []
        self.dyn_param_0 = []

        for i in range(self.njoints):
            _dyn_param_i = system_inertia[i].toDynamicParameters()
            self.dyn_param_0.append(_dyn_param_i)
            self.total_mass += system_inertia[i].mass
            self.masses.append(system_inertia[i].mass)
            self.CoMs_lever.append(system_inertia[i].lever)
            self.inertia_CoM.append(system_inertia[i].inertia)
        self.masses_CAD, self.GoMs_lever, self.inertias_CAD = retrieve_geo_fromCAD(robot_name)

        self._dyn_param_0 = np.vstack(self.dyn_param_0)
        self.dyn_param = self._dyn_param_0.reshape(-1)
        self.iden_param = None
        self._init_ref_param()
        self._init_data()

    def _init_ref_param(self):
        self.inertia_param_per_joint = 10
        self.num_param_total = 10 * self.njoints
        self.ref_param = self.dyn_param

    def _init_data(self):
        self.J_prior = []
        self.Q = []
        self.Qs = []
        for i in range(self.njoints):
            pi_dyn_i = self.dyn_param[
                i
                * self.inertia_param_per_joint : (i + 1)
                * self.inertia_param_per_joint
            ]
            J = inertiaVecToPinertia(pi_dyn_i)
            self.J_prior.append(J)
            Ic, Qs = inertiaVecToIcQs(pi_dyn_i)
            self.Qs.append(Qs)
            Q = inertiaVecToQ(pi_dyn_i)
            self.Q.append(Q)

    def _joint_limit(self):
        joint_pos_limits = []
        for joint_idx in range(self.njoints):
            if joint_idx in self.idx_continuous_joint:
                pos_limit = np.pi * 10
            else:
                pos_limit = self.model.upperPositionLimit[joint_idx]
            joint_pos_limits.append(pos_limit)
        self.joint_pos_limits = joint_pos_limits
        self.joint_vel_limits = self.model.velocityLimit

    def _api_regressor(self, q, qd, qdd):
        reg_without_friction = pin.computeJointTorqueRegressor(
            self.model, self.data, q, qd, qdd
        )
        if self._friction:
            qds = np.zeros(shape=(self.njoints, self.njoints))
            for i in range(self.njoints):
                qds[i][i] = qd[i]
            reg = np.hstack((reg_without_friction, qds))
        else:
            reg = reg_without_friction
        return reg

    def _construct_regressor(self, q, qd, qdd):
        q = self._preprocess_q(q)
        return self._api_regressor(q, qd, qdd)

    def _preprocess_q(self, q):
        if len(self.idx_continuous_joint) != 0:
            for idx in self.idx_continuous_joint:
                q = q.tolist()
                theta = q[idx]
                q.pop(idx)
                cos_theta, sin_theta = np.cos(theta), np.sin(theta)
                q.insert(idx, sin_theta)
                q.insert(idx, cos_theta)
        q = np.array(q)
        return q

    def _regressor(self, q_qd_qdd):
        q, qd, qdd = q_qd_qdd
        reg = self._construct_regressor(q, qd, qdd)
        return reg

    def regressor(self, qs, qds, qdds):
        qs_qds_qdds = zip(qs, qds, qdds)
        reg_list = list(map(self._regressor, qs_qds_qdds))
        reg_batch = np.vstack(reg_list)
        return reg_batch
