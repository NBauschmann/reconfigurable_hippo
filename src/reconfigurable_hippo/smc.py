from os import stat
import threading
import numpy as np
import scipy.optimize

import rospy
import tf.transformations
from dynamic_reconfigure.server import Server
from hippocampus_common.node import Node
from reconfigurable_hippo.cfg import SlidingModeControllerConfig
from std_msgs.msg import Float64MultiArray

# numpy printing options
float_formatter = "{:.5f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})


class SmcNode(Node):
    def __init__(self, name) -> None:
        super().__init__(name=name)

        self.data_lock = threading.RLock()
        self.number_vehicles = self.get_param("~number_vehicles")
        self.controller = SlidingModeController()
        self.setpoint = None
        self.transforms_initialized = False

        self.motor_setpoint_pub = rospy.Publisher("motor_setpoints",
                                                  Float64MultiArray,
                                                  queue_size=1)

        self.dyn_reconf_smc = Server(SlidingModeControllerConfig,
                                     self.on_smc_reconfigure)

        rospy.Subscriber("state_setpoints",
                         Float64MultiArray,
                         self.on_setpoint,
                         queue_size=1)

        rospy.Subscriber("multi_hippo/state",
                         Float64MultiArray,
                         self.on_states,
                         queue_size=1)

    def on_setpoint(self, sp_msg):
        with self.data_lock:
            setpoint = np.asarray(sp_msg.data).reshape(-1, 1)
            self.setpoint = np.copy(setpoint)
            # rospy.loginfo(("[{}] setpoint: {}").format(rospy.get_name(),
            #                                            self.setpoint))

            if self.transforms_initialized:
                u = self.controller.update(setpoint)
                # u = np.zeros((self.number_vehicles * 4, 1))
                self._publish_motor_setpoints(u)

    def on_states(self, state_msg):
        with self.data_lock:
            state_vector = state_msg.data
            self.controller.update_state(state_vector)
            self.transforms_initialized = True

    def on_smc_reconfigure(self, config, level):
        """Callback for the dynamic reconfigure service to set SMC control
        specific parameters.

        Args:
            config (dict): Holds parameters and values of the dynamic
                reconfigure config file.
            level (int): Level of the changed parameters

        Returns:
            dict: The actual parameters that are currently applied.
        """
        with self.data_lock:
            self.controller.k_11 = config["k_11"]
            self.controller.k_12 = config["k_12"]
            self.controller.k_2 = config["k_2"]
            self.controller.k_3 = config["k_3"]
            self.controller.K_1 = config["K_1"]
            self.controller.Khat_1 = config["Khat_1"]
            self.controller.K_2 = config["K_2"]
            self.controller.Khat_2 = config["Khat_2"]
            self.controller.K_3 = config["K_3"]
            self.controller.Khat_3 = config["Khat_3"]
        return config

    def _publish_motor_setpoints(self, u):
        with self.data_lock:
            msg = Float64MultiArray(data=u)
            self.motor_setpoint_pub.publish(msg)


class SlidingModeController():
    def __init__(self,
                 k_11=1.0,
                 k_12=1.0,
                 k_2=1.0,
                 k_3=1.0,
                 K_1=1.0,
                 Khat_1=1.0,
                 K_2=1.0,
                 Khat_2=1.0,
                 K_3=1.0,
                 Khat_3=1.0) -> None:
        self.k_11 = k_11
        self.k_12 = k_12
        self.k_2 = k_2
        self.k_3 = k_3
        self.K_1 = K_1
        self.Khat_1 = Khat_1
        self.K_2 = K_2
        self.Khat_2 = Khat_2
        self.K_3 = K_3
        self.Khat_3 = Khat_3
        self.r_1 = 10
        self.r_2 = 10
        self.r_3 = 10

        self.state_helper = StateHelper()
        self.frames = ['uuv00', 'uuv01', 'uuv02']  # vehicle frames
        self.a = [6, 1, 1]  # number of velocities per link

        self.beta = {
            'map': {
                'map': {
                    'map': np.zeros((6, 1)),
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv00': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv01': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv02': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
            },
            'uuv00': {
                'map': {
                    'map': np.zeros((6, 1)),
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv00': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv01': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv02': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
            },
            'uuv01': {
                'map': {
                    'map': np.zeros((6, 1)),
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv00': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv01': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv02': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
            },
            'uuv02': {
                'map': {
                    'map': np.zeros((6, 1)),
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv00': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv01': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv02': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
            }
        }

        self.dbeta = {
            'map': {
                'map': {
                    'map': np.zeros((6, 1)),
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv00': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv01': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv02': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
            },
            'uuv00': {
                'map': {
                    'map': np.zeros((6, 1)),
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv00': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv01': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv02': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
            },
            'uuv01': {
                'map': {
                    'map': np.zeros((6, 1)),
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv00': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv01': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv02': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
            },
            'uuv02': {
                'map': {
                    'map': np.zeros((6, 1)),
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv00': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv01': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv02': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
            }
        }

        self.pos_uuv00_err = None
        self.eps_quat_err = None
        self.nu_uuv00_uuv00des = None

        self.joint_1_angle_des = None
        self.joint_1_vel_des = None
        self.joint_2_angle_des = None
        self.joint_2_vel_des = None

        self.sigma = {'uuv00': None, 'uuv01': None, 'uuv02': None}
        self.s = {'uuv00': None, 'uuv01': None, 'uuv02': None}

        self.tau_des = {'uuv00': None, 'uuv01': None, 'uuv02': None}
        self.eta_des = {'uuv00': None, 'uuv01': None, 'uuv02': None}

        self.model_helper = ModelHelper()

    def update_state(self, state_vector):
        self.state_helper.update_state(state_vector)

    def update(self, sp_vector):
        """
        Compute control output
        """
        # process setpoint
        self._process_setpoint(sp_vector)

        # compute desired forces on each body
        self._compute_forces_des()

        # compute needed thruster forces to achieve des. forces on each body
        thruster_forces = self._compute_thruster_forces()

        # compute resulting control input
        u = self._compute_u(thruster_forces)
        return u

    def _process_setpoint(self, sp_vector):
        # sp_vector: [pos_des, quat_des, twist_des, joint_1_angle_des, joint_1_vel_des, ...]
        # quat_des in map frame, twist_des in (desired) body frame
        pos_des = np.array([sp_vector[0], sp_vector[1], sp_vector[2]]).reshape(
            (-1, 1))
        quat_des_map = np.array(
            [sp_vector[3], sp_vector[4], sp_vector[5], sp_vector[6]]).reshape(
                (-1, 1))
        twist_des_desbody = np.array([
            sp_vector[7], sp_vector[8], sp_vector[9], sp_vector[10],
            sp_vector[11], sp_vector[12]
        ]).reshape((-1, 1))

        A_map_uuv00des = self.state_helper.compute_A(pos_des, quat_des_map)
        A_uuv00_uuv00des = np.matmul(self.state_helper.get_A('uuv00', 'map'),
                                     A_map_uuv00des)
        # tf transformations.quaternion_from_matrix needs a 4x4 rotation matrix
        R = np.zeros((4, 4))
        R[:3, :3] = A_uuv00_uuv00des[:3, :3]
        R[3, 3] = 1.0

        # imaginary part of unit quaternion describing orientation of
        # frame uvv00des in frame uuv00
        quat_uuv00_uuv00des = tf.transformations.quaternion_from_matrix(
            R).reshape((-1, 1))  # [x,y,z,w]

        # eps_1_1des in paper
        quat_xyz_uuv00_uuv00des = quat_uuv00_uuv00des[:3, :]
        self.eps_quat_err = quat_xyz_uuv00_uuv00des

        # position of frame uuv00des in frame uuv00, p_1_1des in paper
        self.pos_uuv00_err = np.matmul(
            self.state_helper.get_A('map', 'uuv00')[:3, :3],
            (pos_des - self.state_helper.get_theta('uuv00')[:3, :]))

        # associated velocity error in current body frame
        self.nu_uuv00_uuv00des = np.matmul(A_uuv00_uuv00des, twist_des_desbody)

        ## Joint angles and velocities
        self.joint_1_angle_des = np.array([sp_vector[13]]).reshape((-1, 1))
        self.joint_1_vel_des = np.array([sp_vector[14]]).reshape((-1, 1))
        self.joint_2_angle_des = np.array([sp_vector[15]]).reshape((-1, 1))
        self.joint_2_vel_des = np.array([sp_vector[16]]).reshape((-1, 1))

        self._compute_sigma()

    def _compute_forces_des(self):
        self.beta['map']['map']['map'] = np.zeros((6, 1))
        self.dbeta['map']['map']['map'] = np.zeros((6, 1))

        for frame in self.frames:
            parent = self.state_helper.get_parent_name(frame)
            nu = self.state_helper.get_nu(frame, frame, 'map')
            self.beta[frame][frame][parent] = np.matmul(
                self.state_helper.get_Phi(frame), self.get_sigma(frame))
            # dbeta from product rule, derivative of Phi is zero, dsigma so far assumed to be zero-> TODO: calculate from finite differences?
            self.dbeta[frame][frame][parent] = np.zeros((6, 1))
            self.beta[frame][parent]['map'] = np.matmul(
                self.state_helper.get_A(frame, parent),
                self.get_beta(parent, parent, 'map'))

            self.dbeta[frame][frame]['map'] = np.matmul(
                self.state_helper.get_A(frame, parent),
                self.get_dbeta(parent, parent, 'map')) + self.get_dbeta(
                    frame, frame, parent) + np.matmul(
                        self.state_helper.get_se3(
                            self.get_beta(frame, parent, 'map')),
                        self.state_helper.get_nu(frame, frame, parent))

            # print('Current Frame: ', frame)
            # print('nu: ', self.state_helper.get_nu(frame, frame, parent))
            # print(self.get_beta(frame, parent, 'map'))
            # print(self.state_helper.get_se3(self.get_beta(frame, parent,
            #                                               'map')))
            # #print('beta: ', self.beta[frame][parent]['map'])
            # print('DBeta: ', self.get_dbeta(frame, frame, 'map'))

            self.beta[frame][frame]['map'] = self.get_beta(
                frame, parent, 'map') + self.get_beta(frame, frame, parent)

            # print('Beta: ', self.get_beta(
            #    frame, parent, 'map') + self.get_beta(frame, frame, parent))
            # print('Beta: ', self.get_beta(frame, frame, 'map'))
            self.tau_des[frame] = np.matmul(
                self.model_helper.get_mass(), self.get_dbeta(
                    frame, frame, 'map')) + np.matmul(
                        self.model_helper.get_C_RB(nu),
                        self.get_beta(frame, frame, 'map')) + np.matmul(
                            self.model_helper.get_C_A(nu),
                            self.get_beta(frame, frame, 'map')) + np.matmul(
                                self.model_helper.get_D(nu),
                                self.get_beta(frame, frame, 'map'))
            # print('tau_des: ', self.tau_des[frame])

    def _compute_thruster_forces(self):

        # for frame in self.frames[::-1]:
        #     self.eta_des[frame] = np.matmul(
        #         self.state_helper.get_Phi[frame].transpose(),
        #         self.tau_des[frame]) + self....

        # vehicle 3:
        self.eta_des['uuv02'] = np.matmul(
            self.state_helper.get_Phi('uuv02').transpose(),
            self.tau_des['uuv02']
        ) + self.K_3 * self.s['uuv02'] + self.Khat_3 * self.s['uuv02'] / (max(
            np.linalg.norm(self.s['uuv02']), self.r_3))
        # add to parent
        self.tau_des['uuv01'] += np.matmul(
            self.state_helper.get_A('uuv02', 'uuv01').transpose(),
            self.tau_des['uuv02'])

        # vehicle 2:
        self.eta_des['uuv01'] = np.matmul(
            self.state_helper.get_Phi('uuv01').transpose(),
            self.tau_des['uuv01']
        ) + self.K_2 * self.s['uuv01'] + self.Khat_2 * self.s['uuv01'] / (max(
            np.linalg.norm(self.s['uuv01']), self.r_2))
        # add to parent
        self.tau_des['uuv00'] += np.matmul(
            self.state_helper.get_A('uuv01', 'uuv00').transpose(),
            self.tau_des['uuv01'])

        # vehicle 1:
        self.eta_des['uuv00'] = np.matmul(
            self.state_helper.get_Phi('uuv00').transpose(),
            self.tau_des['uuv00']
        ) + self.K_1 * self.s['uuv00'] + self.Khat_1 * self.s['uuv00'] / (max(
            np.linalg.norm(self.s['uuv00']), self.r_1))

        # Thrust allocation:
        # solving for motor forces/thrusts
        # bound = 1N, since that is the maximal value possible in motor model (when u = 1)
        B = self._get_thruster_matrix()
        # print('Thruster Matrix: ', B)
        eta_des_arr = np.concatenate((self.eta_des['uuv00'], self.eta_des['uuv01'], self.eta_des['uuv02'])).reshape((-1,))
        # print(eta_des_arr)
        opt_res = scipy.optimize.lsq_linear(B, eta_des_arr, bounds=(-1, 1))
        # print('optimize results : ', opt_res)
        nu_thrusters = opt_res.x
        # print('nu_thruster: ', nu_thrusters)
        return nu_thrusters


    def _compute_u(self, nu_mot):
        cw = [1, -1, 1, -1]
        cw = np.tile(cw, (1, 3)).reshape((-1,))
        # print('cw: ', cw)
        u = np.sqrt(np.abs(nu_mot)/self.model_helper.KF)*np.sign(nu_mot)*np.sign(cw)
        # print('u: ', u)
        return u.reshape((-1,1))

    def _get_thruster_matrix(self):
        psi_entire_vehicle = self.model_helper.get_psi_thruster()

        B = np.zeros((np.sum(self.a), 12))
        for i, frame in enumerate(self.frames):
            # row: dof of link
            start_idx_row = int(np.sum(self.a[:i]))
            end_idx_row = int(np.sum(self.a[:i+1]))

            # column: number of thruster
            start_idx_col = int(4 * i)
            end_idx_col = int(4 * (i + 1))

            x = psi_entire_vehicle
            B[start_idx_row:end_idx_row, start_idx_col:end_idx_col] = np.matmul(
                self.state_helper.get_Phi(frame).transpose(), x)
            j = i
            while self.state_helper.get_parent_name(frame) != 'map':
                x = np.matmul(
                    self.state_helper.get_A(
                        frame,
                        self.state_helper.get_parent_name(frame)).transpose(),
                    x)
                frame = self.state_helper.get_parent_name(frame)
                j -= 1
                start_idx_row = int(np.sum(self.a[:j]))
                end_idx_row = int(np.sum(self.a[:j+1]))
                B[start_idx_row:end_idx_row,
                  start_idx_col:end_idx_col] = np.matmul(
                      self.state_helper.get_Phi(frame).transpose(), x)
        return B

    def _compute_sigma(self):
        self.sigma['uuv00'] = self.nu_uuv00_uuv00des + np.concatenate(
            (self.k_11 * self.pos_uuv00_err, self.k_12 * self.eps_quat_err),
            axis=0)
        self.sigma['uuv01'] = self.joint_1_vel_des - self.k_2 * (
            self.state_helper.get_theta('uuv01') - self.joint_1_angle_des)
        self.sigma['uuv02'] = self.joint_2_vel_des - self.k_3 * (
            self.state_helper.get_theta('uuv02') - self.joint_2_angle_des)

        for frame in self.frames:
            self.s[frame] = self.sigma[frame] - self.state_helper.get_zeta(
                frame)

    def get_sigma(self, frame):
        return np.copy(self.sigma[frame])

    def get_beta(self, in_frame, from_frame, to_frame):
        return np.copy(self.beta[in_frame][from_frame][to_frame])

    def get_dbeta(self, in_frame, from_frame, to_frame):
        return np.copy(self.dbeta[in_frame][from_frame][to_frame])


class StateHelper():
    def __init__(self) -> None:
        self.frames = ['uuv00', 'uuv01', 'uuv02']  # vehicle frames
        self.A = {
            'map': {
                'map': np.eye(6),
                'uuv00': None,
                'uuv01': None,
                'uuv02': None
            },
            'uuv00': {
                'map': None,
                'uuv00': np.eye(6),
                'uuv01': None,
                'uuv02': None
            },
            'uuv01': {
                'map': None,
                'uuv00': None,
                'uuv01': np.eye(6),
                'uuv02': None
            },
            'uuv02': {
                'map': None,
                'uuv00': None,
                'uuv01': None,
                'uuv02': np.eye(6)
            }
        }

        self.Phi = {
            'uuv00': np.eye(6),
            'uuv01': self.get_Phi_revolute('z'),
            'uuv02': self.get_Phi_revolute('z')
        }

        self.nu = {
            'map': {
                'map': {
                    'map': np.zeros((6, 1)),
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv00': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv01': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv02': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
            },
            'uuv00': {
                'map': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv00': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv01': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv02': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
            },
            'uuv01': {
                'map': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv00': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv01': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv02': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
            },
            'uuv02': {
                'map': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv00': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv01': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
                'uuv02': {
                    'map': None,
                    'uuv00': None,
                    'uuv01': None,
                    'uuv02': None
                },
            },
        }

        self.theta = {'uuv00': None, 'uuv01': None, 'uuv02': None}
        self.pos = None
        self.quat = None

        self.zeta = {'uuv00': None, 'uuv01': None, 'uuv02': None}

        self.offset_parent_to_joint = np.array([0.25, 0, 0]).reshape((-1, 1))
        self.offset_joint_to_child = np.array([0.25, 0, 0]).reshape((-1, 1))

        self.skew_offset_parent_joint = self.get_skew(
            self.offset_parent_to_joint)
        self.skew_offset_joint_child = self.get_skew(self.offset_joint_to_child)

    def _process_states(self, state_vector):
        # theta uuv00/joint 1
        pos = np.array([state_vector[0], state_vector[1],
                        state_vector[2]]).reshape((-1, 1))
        quat = np.array([
            state_vector[4], state_vector[5], state_vector[6], state_vector[3]
        ])
        # self.theta['uuv00'] = np.concatenate((pos, quat.reshape(-1, 1)), axis=0)
        self.theta['uuv00'] = np.array([
            state_vector[0], state_vector[1], state_vector[2], state_vector[3],
            state_vector[4], state_vector[5], state_vector[6]
        ]).reshape((-1, 1))

        # zeta uuv00/joint 1
        twist = np.array([
            state_vector[7], state_vector[8], state_vector[9], state_vector[10],
            state_vector[11], state_vector[12]
        ]).reshape((-1, 1))
        self.zeta['uuv00'] = twist

        # theta + zeta uuv01/joint 2
        self.theta['uuv01'] = np.array([state_vector[7 + 6]]).reshape((-1, 1))
        self.zeta['uuv01'] = np.array([state_vector[7 + 6 + 1]]).reshape((-1, 1))

        # theta + zeta uuv02/joint 3
        self.theta['uuv02'] = np.array([state_vector[7 + 6 + 2]]).reshape((-1, 1))
        self.zeta['uuv02'] = np.array([state_vector[7 + 6 + 2 + 1]]).reshape((-1, 1))

        return pos, quat  # quat in order: [x, y, z, w]

    def update_state(self, state_vector):

        pos, quat = self._process_states(state_vector)

        self._update_A_6DOF(pos, quat)
        self._update_A_revolute_joint('uuv00',
                                      'uuv01',
                                      angle=self.get_theta('uuv01'))
        self._update_A_revolute_joint('uuv01',
                                      'uuv02',
                                      angle=self.get_theta('uuv01'))
        self._update_nu()

    def _update_nu(self):

        for frame in self.frames:
            parent = self.get_parent_name(frame)
            self.nu[frame][frame][parent] = np.matmul(
                self.get_Phi(frame),
                self.get_zeta(frame))
            self.nu[frame][frame]['map'] = np.matmul(
                self.get_A(frame, parent), self.get_nu(
                    parent, parent, 'map')) + self.get_nu(frame, frame, parent)

    def _update_A_revolute_joint(self,
                                 parent_frame,
                                 child_frame,
                                 angle,
                                 axis='z'):
        if axis == 'z':
            direction = np.array([0, 0, 1])
        elif axis == 'y':
            direction = np.array([0, 1, 0])
        elif axis == 'x':
            direction = np.array([1, 0, 0])
        else:
            rospy.logwarn(
                "[{}]: Choose a valid rotation axis for joint!".format(
                    rospy.get_name()))
        # from parent to child
        R = tf.transformations.rotation_matrix(angle=angle,
                                               direction=direction)[:3, :3]
        A_p_j = np.block([[
            np.copy(R),
            np.matmul(np.copy(self.skew_offset_parent_joint), np.copy(R))
        ], [np.zeros((3, 3)), np.copy(R)]])
        A_j_c = np.block([[
            np.eye(3),
            np.matmul(np.copy(self.skew_offset_joint_child), np.eye(3))
        ], [np.zeros((3, 3)), np.eye(3)]])
        A_p_c = np.matmul(A_p_j, A_j_c)

        self.A[parent_frame][child_frame] = A_p_c

        # Inverse transformation: from child to parent
        A_j_p = np.block([[
            np.copy(R).transpose(),
            -np.matmul(np.copy(R), np.copy(self.skew_offset_parent_joint))
        ], [np.zeros((3, 3)), np.copy(R).transpose()]])
        A_c_j = np.block([[
            np.eye(3),
            -np.matmul(np.eye(3), np.copy(self.skew_offset_joint_child))
        ], [np.zeros((3, 3)), np.eye(3)]])
        A_c_p = np.matmul(A_c_j, A_j_p)

        self.A[child_frame][parent_frame] = A_c_p

    def _update_A_6DOF(self, pos, quat):
        R = tf.transformations.quaternion_matrix(quat)[:3, :3]
        pos_skew = self.get_skew(np.asarray(pos).reshape(-1, 1))
        A_map_0 = np.block([[R, np.matmul(pos_skew, R)], [np.zeros((3, 3)), R]])
        A_0_map = np.block([[R.transpose(), -np.matmul(R, pos_skew)],
                            [np.zeros((3, 3)), R.transpose()]])

        self.A['map']['uuv00'] = A_map_0
        self.A['uuv00']['map'] = A_0_map

    def compute_A(self, p_j_i, quat_j_i):
        # Returns transformation matrix A_j_i, which transforms velocities from
        # Frame i to Frame j
        # p_j_i: Position of Frame i in frame j
        # quat_j_i: Orientation of frame i in frame j
        R_j_i = tf.transformations.quaternion_matrix(quaternion=[
            quat_j_i[1, 0], quat_j_i[2, 0], quat_j_i[3, 0], quat_j_i[0, 0]
        ])[:3, :3]
        A_j_i = np.block([[R_j_i, np.matmul(self.get_skew(p_j_i), R_j_i)],
                          [np.zeros((3, 3)), R_j_i]])
        return A_j_i

    def get_A(self, from_frame, to_frame):
        if self.A[from_frame][to_frame] is not None:
            return np.copy(self.A[from_frame][to_frame])
        else:
            rospy.logwarn(
                "[{}]: Transform from {} to {} not defined yet.".format(
                    rospy.get_name(), from_frame, to_frame))

    def get_Phi_revolute(self, axis='z'):
        if axis == 'x':
            Phi = np.array([0, 0, 0, 1, 0, 0]).reshape((-1, 1))
        elif axis == 'y':
            Phi = np.array([0, 0, 0, 0, 1, 0]).reshape((-1, 1))
        elif axis == 'z':
            Phi = np.array([0, 0, 0, 0, 0, 1]).reshape((-1, 1))
        else:
            rospy.logwarn(
                "[{}]: Choose a valid rotation axis for Phi matrix of joint".
                format(rospy.get_name()))
        return Phi

    def get_Phi(self, frame):
        return np.copy(self.Phi[frame])

    def get_theta(self, frame):
        return np.copy(self.theta[frame])

    def get_zeta(self, frame):
        return np.copy(self.zeta[frame].reshape((-1, 1)))

    def get_nu(self, in_frame, from_frame, to_frame):
        return np.copy(self.nu[in_frame][from_frame][to_frame])

    def get_skew(self, array):
        if np.shape(array) == (3, 1):
            return np.array([[0, -array[2, 0], array[1, 0]],
                             [array[2, 0], 0, -array[0, 0]],
                             [-array[1, 0], array[0, 0], 0]])
        else:
            rospy.logwarn(
                "[{}]: Check array shape of input to get_skew, expected (3,1) but got {}"
                .format(rospy.get_name(), np.shape(array)))

    def get_se3(self, nu):
        v1 = float(np.copy(nu[0, 0]))
        v2 = float(np.copy(nu[1, 0]))
        v3 = float(np.copy(nu[2, 0]))
        w1 = float(np.copy(nu[3, 0]))
        w2 = float(np.copy(nu[4, 0]))
        w3 = float(np.copy(nu[5, 0]))

        matrix = np.array([[0.0, -w3, w2, 0.0, -v3, v2],
                           [w3, 0.0, -w1, v3, 0.0, -v1],
                           [-w2, w1, 0.0, -v2, v1, 0.0],
                           [0.0, 0.0, 0.0, 0.0, -w3, w2],
                           [0.0, 0.0, 0.0, w3, 0.0, -w1],
                           [0.0, 0.0, 0.0, -w2, w1, 0.0]])
        return matrix

    def get_parent_name(self, frame):
        if frame == 'uuv00':
            return 'map'
        elif frame == 'uuv01':
            return 'uuv00'
        elif frame == 'uuv02':
            return 'uuv01'


class ModelHelper():
    def __init__(self) -> None:
        self.MASS = 1.47
        self.IXX = 0.002408
        self.IYY = 0.010717
        self.IZZ = 0.010717
        # added Mass params
        self.XU = -1.11
        self.YV = -2.8
        self.ZW = -2.8
        self.KP = -0.00451
        self.MQ = -0.0163
        self.NR = -0.0163
        # damping params
        self.DLX = 5.39
        self.DLY = 17.36
        self.DLZ = 17.63
        self.DAX = 0.00114
        self.DAY = 0.007
        self.DAZ = 0.007
        # propeller thrust and drag coefficient
        self.KF = 1
        self.KM = 0.25
        # thruster position
        self.H = 0.055
        # mass matrix of rigid body
        self.M_RB_11 = self.MASS * np.eye(3)
        self.M_RB_12 = np.zeros((3, 3))
        self.M_RB_21 = np.zeros((3, 3))
        self.M_RB_22 = np.diag([self.IXX, self.IYY, self.IZZ])
        self.M_RB = np.block([[self.M_RB_11, self.M_RB_12],
                              [self.M_RB_21, self.M_RB_22]])
        # added mass
        self.M_A_11 = -np.diag([self.XU, self.YV, self.ZW])
        self.M_A_12 = np.zeros((3, 3))
        self.M_A_21 = np.zeros((3, 3))
        self.M_A_22 = -np.diag([self.KP, self.MQ, self.NR])
        self.M_A = np.block([[self.M_A_11, self.M_A_12],
                             [self.M_A_21, self.M_A_22]])
        self.M = self.M_RB + self.M_A

        self.PSI = self.get_psi_thruster()

    def get_mass(self):
        return np.copy(self.M)

    def get_C_A(self, nu):
        C_A = np.block([[
            np.zeros((3, 3)),
            -self.MASS * self.get_skew(np.matmul(self.M_A_11, nu[:3, :]))
        ],
                        [
                            -self.MASS *
                            self.get_skew(np.matmul(self.M_A_11, nu[:3, :])),
                            self.get_skew(np.matmul(self.M_A_22, nu[3:, :]))
                        ]])
        return C_A

    def get_C_RB(self, nu):
        #print('In get C_RB')
        #print('omega: ', nu[3:, :])
        #print('M_RB_22*nu: ', np.matmul(self.M_RB_22, nu[3:, :]))
        C_RB = np.block(
            [[np.zeros((3, 3)), -self.MASS * self.get_skew(nu[:3, :])],
             [
                 -self.MASS * self.get_skew(nu[:3, :]),
                 self.get_skew(np.matmul(self.M_RB_22, nu[3:, :]))
             ]])
        return C_RB

    def get_D(self, nu):
        D = np.diag([
            self.DLX * abs(nu[0, 0]), self.DLY * abs(nu[1, 0]),
            self.DLZ * abs(nu[2, 0]), self.DAX * abs(nu[3, 0]),
            self.DAY * abs(nu[4, 0]), self.DAZ * abs(nu[5, 0])
        ])
        return D

    def get_psi_thruster(self):
        # mapping matrix thrust force -> forces on body
        return np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                         [
                             self.KM / self.KF, -self.KM / self.KF,
                             self.KM / self.KF, -self.KM / self.KF
                         ], [-self.H, -self.H, self.H, self.H],
                         [self.H, -self.H, -self.H, self.H]])

    def get_skew(self, array):
        if np.shape(array) == (3, 1):
            return np.array([[0, -array[2, 0], array[1, 0]],
                             [array[2, 0], 0, -array[0, 0]],
                             [-array[1, 0], array[0, 0], 0]])
        else:
            rospy.logwarn(
                "[{}]: Check array shape of input to get_skew, expected (3,1) but got {}"
                .format(rospy.get_name(), np.shape(array)))
