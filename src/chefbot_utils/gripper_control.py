#!/usr/bin/env python

import rospy
import rospkg
import numpy as np

from control_wrapper.srv import SetPose
from control_wrapper.srv import GetPose
from control_wrapper.srv import SetJoints
from control_wrapper.srv import SetTrajectory

from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest

from tri_star.msg import TargetPositions
from geometry_msgs.msg import Point, Quaternion, PoseStamped, Pose
from tri_star.robot_util import Robot, pose_matrix_to_msg
from std_msgs.msg import String, Bool


class Gripper(object):
    def __init__(self):
        "docstring"
        self.topic = "/" + "ur" + "/control_wrapper/" + "left" + "/"
        # self.gripper_pub = rospy.Publisher(self.topic + "gripper", Bool, queue_size=10)
        # while self.gripper_pub.get_num_connections() == 0:
        #     rospy.loginfo("Waiting for subscriber to connect")
        #     rospy.sleep(.2)
        print("Waiting for service...")
        rospy.wait_for_service("/ur/control_wrapper/left/gripper_srv")
        self.gripper_sp = rospy.ServiceProxy("/ur/control_wrapper/left/gripper_srv", SetBool)
        print("Service started!")

    def open_gripper(self):
        # self.gripper_pub.publish(True)
        self.rsp = self.gripper_sp(True)
        print(self.rsp.success)
        print("Publsihing gripper open!")

    def close_gripper(self):
        self.rsp = self.gripper_sp(False)
        print(self.rsp.success)
        print("Publsihing gripper close!")

class Gripper_backup:
    def __init__(self):
        # to update?? open/close with https://github.com/ctaipuj/luc_control/blob/master/robotiq_85_control/src/gripper_ur_control.cpp

        self.is_simulator = rospy.get_param("sim")

        self.gripper_pub = rospy.Publisher("/ur_hardware_interface/script_command", String, queue_size=10)
        self.connect_pub = rospy.Publisher("/ur_hardware_interface/connect", String, queue_size=10)
        self.gripper_commands = self.get_gripper_command()
        self.command = "{{GRIPPER_COMMAND}}"

        self.activate_gripper()

        rospy.Subscriber("/ur/control_wrapper/left/gripper", Bool, self.control)

    def get_gripper_command(self):
        commands = ""

        rospack = rospkg.RosPack()
        with open(rospack.get_path("control_wrapper") + "/resources/ur_gripper.script", "r") as command_file:
            commands = command_file.read()

        return commands + "\n"

    def activate_gripper(self):
        if not self.is_simulator:
            command = self.gripper_commands.replace(self.command, "rq_activate_and_wait()")
            self.gripper_pub.publish(command)
            rospy.sleep(0.1)
            self.connect_pub.publish(True)

    def control(self, data):
        if not self.is_simulator:
            if data.data:
                self.open_gripper()
            else:
                self.close_gripper()
            rospy.sleep(0.1)
            self.connect_pub.publish(True)

    def open_gripper(self):
        command = self.gripper_commands.replace(self.command, "rq_open()")
        self.gripper_pub.publish(command)

    def close_gripper(self):
        command = self.gripper_commands.replace(self.command, "rq_close()")
        self.gripper_pub.publish(command)

    def deactivate_gripper(self):
        if not self.is_simulator:
            command = self.gripper_commands.replace(self.command, "")
            self.gripper_pub.publish(command)
            rospy.sleep(0.1)
            self.connect_pub.publish(True) 
