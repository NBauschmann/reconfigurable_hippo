import numpy as np
import rospy
import threading


class ExtendedKalmanFilter(object):

    def __init__(self, x0, p0_mat, v, w_dist, w_yaw, joint_length):
        self.dim_state = 2
        # self.dim_meas = dim_meas
        self._x_est_0 = x0
        self._x_est = self._x_est_0
        self._last_time_stamp_update = rospy.get_time()
        self._last_time_stamp_prediction = rospy.get_time()
        self._p0_mat = p0_mat
        self._p_mat = self._p0_mat
        self._v_mat = v
        self._w_mat_dist = w_dist
        self._w_mat_yaw = w_yaw
        self._joint_length = joint_length  # in m from CoM of vehicle # 0.29m
        self.lock = threading.Lock()

    def get_x_est(self):
        return np.copy(self._x_est)

    def get_x_est_0(self):
        return np.copy(self._x_est_0)

    def get_p_mat(self):
        return np.copy(self._p_mat)

    def get_x_est_last(self):
        return np.copy(self._x_est_last)

    def reset(self, x_est_0=None, p0_mat=None):
        if x_est_0:
            self._x_est = x_est_0
        else:
            self._x_est = self._x_est_0
        if p0_mat:
            self._p_mat = p0_mat
        else:
            self._p_mat = self._p0_mat

    def predict(self, dt):
        self._x_est_last = self._x_est
        a_mat = np.array([[1, dt], [0, 1]])
        self._x_est = np.matmul(a_mat, self.get_x_est())
        self._p_mat = np.matmul(np.matmul(a_mat, self.get_p_mat()),
                                a_mat.transpose()) + self._v_mat
        return True

    def update_yaw_data(self, z_yaw):
        # measurement is difference in vehicles' yaw angle
        self._x_est_last = self._x_est

        z_est_yaw = np.asarray(self.get_x_est()[0, 0]).reshape((-1, 1))
        h_mat_yaw = np.array([[1.0, 0]])

        # innovation
        y_yaw = z_yaw - z_est_yaw
        # wrap angle innovations between -pi and pi
        y_yaw = np.arctan2(np.sin(y_yaw), np.cos(y_yaw))

        self._x_est, self._p_mat = self._update(self.get_x_est(),
                                                self.get_p_mat(), y_yaw,
                                                h_mat_yaw, self._w_mat_yaw)
        return True

    def h_fun_dist(self, x_est):
        z_est = 2 * self._joint_length * np.sin(0.5 * (np.pi - x_est[0, 0]))
        return z_est

    def h_jacobian_dist(self, x_est):
        h_mat = np.array(
            [[-self._joint_length * np.cos(0.5 * (np.pi - x_est[0, 0])), 0]])
        return h_mat

    def update_dist_data(self, z_dist):
        # measurement is: distance between vehicles' center of mass
        self._x_est_last = self._x_est

        # estimated distance measurement
        z_est_dist = self.h_fun_dist(self.get_x_est())
        h_mat_dist = self.h_jacobian_dist(self.get_x_est())

        # innovation
        y_dist = np.asarray(z_dist - z_est_dist).reshape((-1, 1))

        self._x_est, self._p_mat = self._update(self.get_x_est(),
                                                self.get_p_mat(), y_dist,
                                                h_mat_dist, self._w_mat_dist)

        return True

    def _update(self, x_est, p_mat, y, h_mat, w_mat):
        """ helper function for general update """

        # compute K gain
        tmp = np.matmul(np.matmul(h_mat, p_mat), h_mat.transpose()) + w_mat
        k_mat = np.matmul(np.matmul(p_mat, h_mat.transpose()),
                          np.linalg.inv(tmp))

        # update state
        x_est = x_est + np.matmul(k_mat, y)

        # update covariance
        p_tmp = np.eye(self.dim_state) - np.matmul(k_mat, h_mat)
        p_mat = np.matmul(p_tmp, p_mat)
        
        return x_est, p_mat
