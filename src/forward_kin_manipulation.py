#!/usr/bin/env python

import rospy
import rospkg
import numpy as np
import os

from control_wrapper.srv import SetPose
from control_wrapper.srv import GetPose
from control_wrapper.srv import SetJoints
from control_wrapper.srv import SetTrajectory
from chefbot.srv import PickPlace, PickPlaceRequest, PickPlaceResponse, ActionCommand, ActionCommandRequest, ActionCommandResponse
from control_wrapper.srv import GetJoints

from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from tri_star.msg import TargetPositions
from geometry_msgs.msg import Point, Quaternion, PoseStamped, Pose
from tri_star.robot_util import Robot, pose_matrix_to_msg
from std_msgs.msg import String, Bool

import os
import sys
import pickle as pkl
import rospkg

import actionlib
import time
import roslib

roslib.load_manifest('ur_driver')
import control_msgs
import trajectory_msgs
from trajectory_msgs.msg import *
# from control_msgs import FollowJointTrajectory
from control_msgs.msg import (
    FollowJointTrajectoryGoal,
    FollowJointTrajectoryActionResult,
    FollowJointTrajectoryActionGoal,
    FollowJointTrajectoryAction,
    FollowJointTrajectoryActionFeedback,
    JointTrajectoryAction
)


# Hyper-parameters go here
SLEEP_RATE = 10
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
GRIPPER_CMD = "gripper_open"

PKG_NAME = "chefbot"

class Gripper(object):
    def __init__(self):
        self.topic = "/" + "ur" + "/control_wrapper/" + "left" + "/"
        print("Waiting for service...")
        rospy.wait_for_service("/ur/control_wrapper/left/gripper_srv")
        self.gripper_sp = rospy.ServiceProxy("/ur/control_wrapper/left/gripper_srv", SetBool)
        print("Service started!")

    def open_gripper(self):
        self.rsp = self.gripper_sp(True)
        print(self.rsp.success)
        print("Publsihing gripper open!")

    def close_gripper(self):
        self.rsp = self.gripper_sp(False)
        print(self.rsp.success)
        print("Publsihing gripper close!")


class ExecuteTrajectory(object):
    def __init__(self, fn="traj_dict_path.pkl"):
        rospack = rospkg.RosPack()
        save_dir = "demo/trajectories"
        pkg_path = rospack.get_path(PKG_NAME)
        self._save_path = os.path.join(pkg_path, save_dir)
        self.traj_dict_path = os.path.join(self._save_path, fn)

        self.client = actionlib.SimpleActionClient('/scaled_pos_traj_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        self.joint_position_dict = self._load_trajectory()  # {'trajectory1_name': [{joint1_name: joint1_state, joint2_name: joint2_state, ...},
                                                            #                      {joint1_name: joint1_state, joint2_name: joint2_state, ...},
                                                            #                       .............],
                                                            # 'trajectory2_name': [{joint1_name: joint1_state, joint2_name: joint2_state, ...},
                                                            #                      {joint1_name: joint1_state, joint2_name: joint2_state, ...},
                                                            #                       .............]
        self.gripper = Gripper()

        print("Waiting for /ur/control_wrapper/left/get_joints")
        rospy.wait_for_service("/ur/control_wrapper/left/get_joints")
        self._get_cur_joints_sp = rospy.ServiceProxy("/ur/control_wrapper/left/get_joints", GetJoints)


    def _load_trajectory(self):
        ret_dict = None

        with open(self.traj_dict_path, "rb") as f:
            ret_dict = pkl.load(f)

        return ret_dict

    def _execute_trajectory(self, traj_name):
        print("traj_name: ", traj_name)
        START_DUR = 0.0
        MAX_VEL = 191 # degree per sec.
        TIME_BUFF = .3 # sec
        cur_speed = MAX_VEL * .35 # CHECK TABLET FOR TRUE SCALING VAL.
        cur_joints = self._get_cur_joints_sp()
        cur_joints = np.array(cur_joints.joints.position)
        # extract joint names and angles from the dict
        # Select a trajectory
        # from cmdline input and retrieve corresponding data.
        traj = self.joint_position_dict[traj_name]
        cmd_list = []   # Will store joint cmds transformed into ros data

        # We must break the overall trajectory into sub trajectories
        # As cmds to the gripper are issued through a completely diff
        # process. These values help keep track of this.
        start_time = START_DUR
        new_sub_traj = True
        traj_idx = 0
        for cmd in traj:
            # If its a gripper cmd and not a postition, just append to list
            # and we'll execute it later
            # if GRIPPER_CMD in cmd or TTS_CMD in cmd:
            if GRIPPER_CMD in cmd:
                cmd_list.append(cmd)
                new_sub_traj = True
                traj_idx = len(cmd_list)
                start_time = 1
            else:
                if new_sub_traj:
                    cmd_list.append([])
                    new_sub_traj = False
                    start_time = START_DUR   #

                positions = [cmd[j] for j in JOINT_NAMES]
                new_joints = np.array(positions)
                max_deg_dist = max(abs(new_joints - cur_joints))
                # print("max_deg_dist: ", max_deg_dist)
                # print("calc time: ", np.rad2deg(max_deg_dist / cur_speed))
                start_time += np.rad2deg(max_deg_dist / cur_speed) + TIME_BUFF # t = d / v


                cmd_list[traj_idx].append(
                    JointTrajectoryPoint(positions=positions,
                                         time_from_start=rospy.Duration(
                                             start_time)
                                         ))
                cur_joints = new_joints
            # start_time += 4.0

        for cmd in cmd_list:
            if isinstance(cmd, list):
                g = FollowJointTrajectoryGoal()
                g.trajectory = JointTrajectory()
                g.trajectory.joint_names = JOINT_NAMES
                g.trajectory.points = cmd

                rospy.loginfo("Executing sub trajectory!")
                for i, p in enumerate(cmd):
                    print("point {} time: {}".format(i, p.time_from_start))
                self.client.send_goal(g)
                self.client.wait_for_result()
                print("got result")

            elif GRIPPER_CMD in cmd:
                rospy.loginfo("Commanding the gripper!")
                if cmd[GRIPPER_CMD]:
                    self.gripper.open_gripper()
                else:
                    self.gripper.close_gripper()

            # NOTE: This sleep is REQUIRED for robot to reconnect ROS
            # afer using the gripper.
            rospy.sleep(.5)

    def _keyboard_select_trajectories(self):
        """Keyboard interface for selecting one of the saved trajectories"""
        traj_by_num = dict(enumerate(self.joint_position_dict))
        traj_chosen = False
        while not traj_chosen:
            print("Available trajectories: ")
            for k in traj_by_num:
                print("\t{}: {}".format(k, traj_by_num[k]))

            try:
                # choice = int(input("Enter trajectory number (s): "))
                choice_str = input("Enter trajectory number (s): ").strip()
                choice = [int(c) for c in choice_str.split()]
                ret = [traj_by_num[c] for c in choice]
                traj_chosen = True
            except KeyError:
                print("{} not a valid trajectory choice, try again!\n".format(choice))
            except ValueError:
                print("Not a number, try again!\n")

        return ret

    def _select_trajectories(self, traj_id):
        traj_by_num = dict(enumerate(self.joint_position_dict))
        print("Available trajectories: ")
        for k in traj_by_num:
            print("\t{}: {}".format(k, traj_by_num[k]))
        ret = [traj_by_num[traj_id]]
        return ret

    def run(self, traj_id):
        traj_by_num = dict(enumerate(self.joint_position_dict))
        print("Available trajectories: ")
        for k in traj_by_num:
            print("\t{}: {}".format(k, traj_by_num[k]))
        # traj_names = self._select_trajectories(traj_id)
        traj_names = [traj_by_num[traj_id]]
        print("run", traj_names)
        for traj in traj_names:
            self._execute_trajectory(traj)

class ForwardKinematics(object):
    def __init__(self):

        self.connect_pub = rospy.Publisher("/ur/control_wrapper/left/connect", Bool, queue_size=10)

        # # Robot IK service proxies
        # rospy.wait_for_service("/ur/control_wrapper/left/get_pose")
        # self.get_robot_pose_sp = rospy.ServiceProxy("/ur/control_wrapper/left/get_pose", GetPose)

        # rospy.wait_for_service("/ur/control_wrapper/left/set_pose")
        # self.set_robot_pose_sp = rospy.ServiceProxy("/ur/control_wrapper/left/set_pose", SetPose)

        # Action Command Service
        self.action_command_srv = rospy.Service("/chefbot/learning/action_command", ActionCommand, self.action_command_srv_cb)

        # Open gripper initially
        self.gripper = Gripper()
        self.gripper.open_gripper()

        # Execute trajectory
        self.trajectory_dict = "fk_trajectories.pkl"
        self.execute_trajectory = ExecuteTrajectory(self.trajectory_dict)

    def action_command_srv_cb(self, srv):
        print("action_command")
        command_name = srv.command.data
        self.execute_trajectory._execute_trajectory(command_name)
        return ActionCommandResponse(exec_success=True)



    def run(self):
        pass


if __name__ == '__main__':
    try:
            rospy.init_node('forward_kinematics', anonymous=True)

            forward_kinematics = ForwardKinematics()
            while not rospy.is_shutdown():
                forward_kinematics.run()


    except rospy.ROSInterruptException:
        pass
