#!/usr/bin/env python

import rospy
import rospkg
import numpy as np

from control_wrapper.srv import SetPose
from control_wrapper.srv import GetPose
from control_wrapper.srv import SetJoints
from control_wrapper.srv import SetTrajectory
from chefbot.srv import PickPlace, PickPlaceRequest, PickPlaceResponse

from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from tri_star.msg import TargetPositions
from geometry_msgs.msg import Point, Quaternion, PoseStamped, Pose 
from tri_star.robot_util import Robot, pose_matrix_to_msg
from std_msgs.msg import String,Bool, Float32


class PerceptionPickPlace(object):
    def __init__(self):

        # Object position transformed w.r.t. robot
        self.transformed_object_pos_sub = rospy.Subscriber('chefbot/perception/transformed_grasp_point', PoseStamped, self.object_pos_cb)

        # Destination position transformed w.r.t robot
        self.transformed_destination_pos_sub = rospy.Subscriber('/chefbot/test/destination_position', PoseStamped, self.object_destination_cb)

        # Pick Place service proxy
        rospy.wait_for_service("/chefbot/manipulation/pick_place")
        self.pick_place_sp = rospy.ServiceProxy("/chefbot/manipulation/pick_place", PickPlace)
        
        # Robot position service proxy
        rospy.wait_for_service("/ur/control_wrapper/left/get_pose")
        self.get_robot_pose_sp = rospy.ServiceProxy("/ur/control_wrapper/left/get_pose", GetPose)

        # Obtain the current robot pose
        self.current_pose = self.get_robot_pose_sp()


    def object_pos_cb(self, msg):
        self.object_depth = float(0.025) # temporary
        self.object_pose = msg.pose
        self.start_pose = self.current_pose.pose
        rospy.sleep(0.5)
        response = self.pick_place_sp(start_pose = self.start_pose,
                                                object_pose = self.object_pose,
                                                destination_pose = self.start_pose, # temporary
                                                object_depth = self.object_depth)
        
        
    def object_destination_cb(self, msg):
        pass
    


    def run(self):
        pass


if __name__ == '__main__':
    try:
            rospy.init_node('perception_pick_and_place', anonymous=True)

            perception_pick_place = PerceptionPickPlace()
            while not rospy.is_shutdown():
                perception_pick_place.run()


    except rospy.ROSInterruptException:
        pass
