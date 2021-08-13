import threading
import numpy as np

import rospy
from dynamic_reconfigure.server import Server
from hippocampus_common.node import Node
from reconfigurable_hippo.cfg import SlidingModeControllerConfig
from std_msgs.msg import Float64MultiArray


class SmcNode(Node):
    def __init__(self, name) -> None:
        super().__init__(name=name)

        self.data_lock = threading.RLock()
        self.number_vehicles = self.get_param("~number_vehicles")
        self.controller = SlidingModeController()
        self.setpoint = None
        self.transformation_helper = TransformationHelper()

        self.motor_setpoint_pub = rospy.Publisher(
            "motor_setpoints", Float64MultiArray, queue_size=1)

        self.dyn_reconf_smc = Server(SlidingModeControllerConfig,
                                     self.on_smc_reconfigure)

        rospy.Subscriber("state_setpoints",
                         Float64MultiArray,
                         self.on_setpoint,
                         queue_size=1)

        rospy.Subscriber("multi_hippo/state", Float64MultiArray, self.on_states, queue_size=1)

    def on_setpoint(self, sp_msg):
        self.setpoint = np.asarray(sp_msg.data).reshape(-1, 1)
        rospy.loginfo(("[{}] setpoint: {}").format(rospy.get_name(), self.setpoint))

        u = np.zeros((self.number_vehicles*4, 1))
        self.publish_motor_setpoints(u)

    def on_states(self, state_msg):
        self.transformation_helper.update_state(state_msg)
        
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

    def publish_motor_setpoints(self, u):
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

        self.Phi_revolute = np.array([0, 0, 0, 0, 0, 1]).reshape(
            (-1, 1))  # z-axis
        self.Phi_6DOF = np.eye(6)


class TransformationHelper():
    def __init__(self) -> None:
        
        pass

    def update_state():
