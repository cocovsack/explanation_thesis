#! /usr/bin/env python

import rospy
import numpy as np

from chefbot_utils.gripper_control import Gripper
# Message imports go here
from std_msgs.msg import String

from control_wrapper.srv import GetJoints
# Service imports go here
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse
from sensor_msgs.msg import JointState

from chefbot_utils.util import TTS
# All other imports go here
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
TTS_CMD = "tts"

PKG_NAME = "chefbot"


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
        self.tts = TTS()
        rospy.wait_for_service("/ur/control_wrapper/left/get_joints")
        self._get_cur_joints_sp = rospy.ServiceProxy("/ur/control_wrapper/left/get_joints", GetJoints)

    def _load_trajectory(self):
        ret_dict = None

        with open(self.traj_dict_path, "rb") as f:
            ret_dict = pkl.load(f)

        return ret_dict

    def _select_trajectories(self):
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

    def _execute_trajectory(self, traj_name):
        START_DUR = 0.0
        MAX_VEL = 191 # degree per sec.
        TIME_BUFF = .3 # sec
        cur_speed = MAX_VEL * .25 # CHECK TABLET FOR TRUE SCALING VAL.
        cur_joints = self._get_cur_joints_sp()
        cur_joints = np.array(cur_joints.joints.position)
        # extract joint names and angles from the dict # Select a trajectory
        # from cmdline input and retrieve corresponding data.
        traj = self.joint_position_dict[traj_name]
        cmd_list = []   # Will store joint cmds transformed into ros data

        # joint_velocities = [0.01] * len(JOINT_NAMES)
        # joint_accelerations = [0.0] * len(JOINT_NAMES)
        # We must break the overall trajectory into sub trajectories
        # As cmds to the gripper are issued through a completely diff
        # process. These values help keep track of this.
        start_time = START_DUR   #
        new_sub_traj = True
        traj_idx = 0
        for cmd in traj:
            # If its a gripper cmd and not a postition, just append to list
            # and we'll execute it later
            # if GRIPPER_CMD in cmd or TTS_CMD in cmd:
            if GRIPPER_CMD in cmd or TTS_CMD in cmd:
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
                                         # velocities=joint_velocities,
                                         # accelerations=joint_accelerations,
                                         time_from_start=rospy.Duration(
                                             start_time)
                                         ))
                cur_joints = new_joints
            # start_time += 1.0



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

            elif TTS_CMD in cmd:
                rospy.loginfo("Speaking!")
                txt = cmd[TTS_CMD]
                self.tts.say(txt)
                rospy.sleep(3.5)

            # NOTE: This sleep is REQUIRED for robot to reconnect ROS
            # afer using the gripper.
            rospy.sleep(.5)

    def run(self):
        traj_names = self._select_trajectories()

        for traj in traj_names:
            self._execute_trajectory(traj)

        # self.clip_5()

    def _execute_traj_by_num(self, traj_idx_list):
        """
        Execute recorded trajectories indicated by traj_idx_list
        @traj_idx_list: list of ints, where ints correspond to recorded
        trajectory in self.joint_position_dict
        """
        traj_by_num = dict(enumerate(self.joint_position_dict))
        for traj_idx in traj_idx_list:
            # input("Press enter to continue...")
            traj = traj_by_num[traj_idx]
            rospy.loginfo("Executing {}".format(traj))
            self._execute_trajectory(traj)

    def clip_1(self):
        traj_num_list = [15, 17, 2, 0, 10, 17, 12, 13, 14 ] # TODO: Put traj numbers in
        #traj_num_list = [14]
        self._execute_traj_by_num(traj_num_list)

    def clip_2(self):
        traj_num_list = [2, 12, 13, 14, 0, 10, 7, 11] # TODO: Put traj numbers in
        #traj_num_list = [7]

        self._execute_traj_by_num(traj_num_list)

    def clip_3(self):
        traj_num_list = [3, 12, 13, 14, 16] # TODO: Put traj numbers in
        self._execute_traj_by_num(traj_num_list)

    def clip_4(self):
        traj_num_list = [14, 16, 0, 10, 7, 11] # TODO: Put traj numbers in
        self._execute_traj_by_num(traj_num_list)

    def clip_5(self):
        traj_num_list = [2, 14, 0, 10] # TODO: Put traj numbers in
        self._execute_traj_by_num(traj_num_list)


def main():
    rospy.init_node("execute_trajectory")

    fn = "fk_trajectories.pkl"
    execute_trajectory = ExecuteTrajectory(fn)
    print("Waiting for server...")
    execute_trajectory.client.wait_for_server()
    print("Connected to server...")

    rate = rospy.Rate(SLEEP_RATE)

    # This while loop is what keeps the node from dying
    while not rospy.is_shutdown():
        execute_trajectory.run()
        rate.sleep()


if __name__ == '__main__':
    main()
