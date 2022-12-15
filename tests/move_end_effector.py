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
# import sys
# sys.path.append('/home/scazlab/chefbot_catkin_ws/src/chef-bot')
# from include.gripper_control import Gripper
import rtde_control
from rtde_control import RTDEControlInterface
from robotiq_preamble import ROBOTIQ_PREAMBLE
import time


# class RobotiqGripper(object):
#     """
#     RobotiqGripper is a class for controlling a robotiq gripper using the
#     ur_rtde robot interface.

#     Attributes:
#         rtde_c (rtde_control.RTDEControlInterface): The interface to use for the communication
#     """
#     def __init__(self, rtde_c):
#         """
#         The constructor for RobotiqGripper class.

#         Parameters:
#            rtde_c (rtde_control.RTDEControlInterface): The interface to use for the communication
#         """
#         self.rtde_c = rtde_c

#     def call(self, script_name, script_function):
#         return self.rtde_c.sendCustomScriptFunction(
#             "ROBOTIQ_" + script_name,
#             ROBOTIQ_PREAMBLE + script_function
#         )

#     def activate(self):
#         """
#         Activates the gripper. Currently the activation will take 5 seconds.

#         Returns:
#             True if the command succeeded, otherwise it returns False
#         """
#         import pdb
#         pdb.set_trace()
#         ret = self.call("ACTIVATE", "rq_activate()")
#         time.sleep(5)  # HACK
#         return ret

#     def set_speed(self, speed):
#         """
#         Set the speed of the gripper.

#         Parameters:
#             speed (int): speed as a percentage [0-100]

#         Returns:
#             True if the command succeeded, otherwise it returns False
#         """
#         return self.call("SET_SPEED", "rq_set_speed_norm(" + str(speed) + ")")

#     def set_force(self, force):
#         """
#         Set the force of the gripper.

#         Parameters:
#             force (int): force as a percentage [0-100]

#         Returns:
#             True if the command succeeded, otherwise it returns False
#         """
#         return self.call("SET_FORCE", "rq_set_force_norm(" + str(force) + ")")

#     def move(self, pos_in_mm):
#         """
#         Move the gripper to a specified position in (mm).

#         Parameters:
#             pos_in_mm (int): position in millimeters.

#         Returns:
#             True if the command succeeded, otherwise it returns False
#         """
#         # import pdb
#         # pdb.set_trace()
#         value = self.call("MOVE", "rq_move_and_wait_mm(" + str(pos_in_mm) + ")")
#         # rospy.sleep(5)
#         return value


#     def get_closed_mm(self, gripper_socket):
#         gripper_closed_mm = [0, 0, 0, 0]
#         return gripper_closed_mm[gripper_socket - 1]


#     def detect_object(self):
#         """
#         Detect if object was grasped

#         Returns:
#             True if the command succeeded, otherwise it returns False
#         """

#         # value = self.call("", "get_closed_mm(1)")
#         value = self.get_closed_mm(1)
#         print("Detect object: ", value)
#         return value

#     def open(self):
#         """
#         Open the gripper.

#         Returns:
#             True if the command succeeded, otherwise it returns False
#         """
#         return self.call("OPEN", "rq_open_and_wait()")

#     def close(self):
#         """
#         Close the gripper.

#         Returns:
#             True if the command succeeded, otherwise it returns False
#         """
#         return self.call("CLOSE", "rq_close_and_wait()")


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

    
class Gripper_backup(object):
    def __init__(self):
        # to update?? open/close with https://github.com/ctaipuj/luc_control/blob/master/robotiq_85_control/src/gripper_ur_control.cpp

        self.is_simulator = rospy.get_param("sim")

        self.gripper_pub = rospy.Publisher("/ur_hardware_interface/script_command", String, queue_size=10)

        self.connect_pub = rospy.Publisher("/ur_hardware_interface/connect", String, queue_size=10)
        self.robot_connect_pub = rospy.Publisher("/ur/control_wrapper/left/connect", Bool, queue_size=10)
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
            self.robot_connect_pub.publish(True)

    def open_gripper(self):
        command = self.gripper_commands.replace(self.command, "rq_open()")
        self.gripper_pub.publish(command)

    def close_gripper(self):
        command = self.gripper_commands.replace(self.command, "rq_close()")
        self.gripper_pub.publish(command)

    # def detect_object(self):
    #     import pdb
    #     pdb.set_trace()
    #     command = self.gripper_commands.replace(self.command, "rq_is_object_detected()")
    #     # self.gripper_pub.publish(command)

    def deactivate_gripper(self):
        if not self.is_simulator:
            command = self.gripper_commands.replace(self.command, "")
            self.gripper_pub.publish(command)
            rospy.sleep(0.1)
            self.connect_pub.publish(True)

class MoveRobot(object):
    def __init__(self):

        # Object position transformed w.r.t. robot
        self.transformed_object_pos_sub = rospy.Subscriber('/chefbot/test/aruco_object_position', PoseStamped, self.object_pos_cb)

        self.connect_pub = rospy.Publisher("/ur/control_wrapper/left/connect", Bool, queue_size=10)

        # Robot IK services

        rospy.wait_for_service("/ur/control_wrapper/left/get_pose")
        self.get_robot_pose_sp = rospy.ServiceProxy("/ur/control_wrapper/left/get_pose", GetPose)

        rospy.wait_for_service("/ur/control_wrapper/left/set_pose")
        self.set_robot_pose_sp = rospy.ServiceProxy("/ur/control_wrapper/left/set_pose", SetPose)
        self.current_pose = self.get_robot_pose_sp()
        self.gripper = Gripper()
        # self.gripper.activate_gripper()
        # print("Gripper Open!")
        self.gripper.open_gripper()
        # self.connect_pub.publish(True)
        #rtde_c = RTDEControlInterface("192.168.1.146")
        # self.rgripper = RobotiqGripper(rtde_c)
        # self.rgripper.activate()
        # self.connect_pub.publish("True")
        # value = self.rgripper.move(1)
        # print(value)
        # detect_val = self.rgripper.detect_object()
        # self.gripper.activate_gripper()
        # self.gripper.close_gripper()
        # self.gripper.detect_object()
        # self.gripper.deactivate_gripper()


    def object_pos_cb(self, msg):
        self.object_depth = 0.1
        self.object_pose = msg.pose
        start_pose = self.current_pose
        rospy.sleep(0.5)

        ############## Move to current xy position of object ####################

        # print("Current Pose: ", self.current_pose)

        xy_pick_pose = self.assign_pose(self.object_pose.position.x,
                                   self.object_pose.position.y,
                                   start_pose.pose.position.z,
                                   start_pose.pose.orientation)

        # print("Goal Pose XY: ", xy_pose)
        reached_xy_pick = self.set_robot_pose_sp(request_pose=xy_pick_pose)
        rospy.sleep(0.5)

        z_pick_down_pose = self.assign_pose(xy_pick_pose.position.x,
                                  xy_pick_pose.position.y,
                                  self.object_depth,
                                  xy_pick_pose.orientation)

        # print("Goal Pose Z: ", z_pose)

        ############## Drop down to depth of the grasping surface ####################
        if reached_xy_pick.is_reached:
            reached_z_pick_down = self.set_robot_pose_sp(request_pose=z_pick_down_pose)
            # print(reached_z)
            rospy.sleep(0.5)

            if reached_z_pick_down.is_reached:
                # self.gripper.activate_gripper()
                self.gripper.close_gripper()
                # self.gripper.deactivate_gripper()
                print("Gripper Closed!")
                self.connect_pub.publish(True)
                rospy.sleep(0.5)

                ############## Lift object to safe height #####################################
                z_pick_up_pose = self.assign_pose(xy_pick_pose.position.x,
                                                  xy_pick_pose.position.y,
                                                  start_pose.pose.position.z,
                                                  xy_pick_pose.orientation)

                reached_z_pick_up = self.set_robot_pose_sp(request_pose=z_pick_up_pose)

                destination_pose = start_pose

                if reached_z_pick_up.is_reached:
                    print("Moved object up!!")


                    ############## Go to the xy coordinate of the destination #####################

                    xy_place_up_pose = self.assign_pose(destination_pose.pose.position.x,
                                                    destination_pose.pose.position.y,
                                                    destination_pose.pose.position.z,
                                                    start_pose.pose.orientation)

                    reached_xy_place_up = self.set_robot_pose_sp(request_pose=xy_place_up_pose)

                    if reached_xy_place_up:
                        print("Moved to place location!!")

                        ############## Keep object down ###############################################

                        z_place_down_pose = self.assign_pose(destination_pose.pose.position.x,
                                                            destination_pose.pose.position.y,
                                                            self.object_depth,
                                                            start_pose.pose.orientation)

                        reached_z_place_down = self.set_robot_pose_sp(request_pose=z_place_down_pose)

                        if reached_z_place_down.is_reached:
                            print("Placed object!")
                            self.gripper.open_gripper()
                            self.connect_pub.publish(True)
                            rospy.sleep(0.5)







    def assign_pose(self, x, y, z, orientation):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation = orientation
        return pose


    def run(self):
        pass


if __name__ == '__main__':
    try:
            rospy.init_node('move_robot', anonymous=True)

            move_robot = MoveRobot()
            while not rospy.is_shutdown():
                move_robot.run()


    except rospy.ROSInterruptException:
        pass
