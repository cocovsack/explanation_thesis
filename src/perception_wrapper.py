#!/usr/bin/env python

import cv2
import numpy as np
import rospy
import ros_numpy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool
import message_filters
from sensor_msgs.msg import CameraInfo, RegionOfInterest
from std_msgs.msg import Int64MultiArray, Float64MultiArray, MultiArrayDimension
import ros_numpy
import matplotlib.pyplot as plt
from PIL import Image as ImageShow
import os
import math
from collections import deque
from chefbot.srv import ItemRequest, ItemRequestRequest, ItemRequestResponse, MasterItemRequirement, MasterItemRequirementRequest, MasterItemRequirementResponse
from item_perception.item_perception_logic import handle_item_request
from geometry_msgs.msg import PointStamped, PoseStamped



class ItemRequirement():
    """
    ROS node
    """

    def __init__(self):

        # Initialize the node
        print("initializing node")
        rospy.init_node('item_requirement')

        self.robot_pos = None

        # Subscriber
        rospy.Subscriber('chefbot/perception/transformed_grasp_point',PoseStamped,self.get_grasp_point_cb)


        # Item Request Service
        self.item_request_srv = rospy.Service("/chefbot/learning/master_item_requirement", MasterItemRequirement, self.master_item_requirement_srv_cb)


        rospy.wait_for_service('/chefbot/learning/item_request')
        self.item_request_sp = rospy.ServiceProxy('/chefbot/learning/item_request', ItemRequest)

        self.grasp_point = None

    def get_grasp_point_cb(self, grasp_point):
        print("inside get grasp point cb", grasp_point)
        self.grasp_point = grasp_point

    def master_item_requirement_srv_cb(self, srv):
        item_request = srv.item.data
        print("Request in perception wrapper: ",item_request)
        req = ItemRequestRequest()
        req.item.data = item_request
        self.item_request_sp(req)

        rospy.sleep(1)

        msg = MasterItemRequirementResponse()

        if self.grasp_point is not None:
            msg.header = self.grasp_point.header
            msg.pose = self.grasp_point.pose
            msg.found = True
            self.grasp_point = None

            print("Response in perception wrapper if grasp point is found: ",msg)
            return msg
        else:
            msg.pose.position.x = -1
            msg.pose.position.y = -1
            msg.pose.position.z = -1
            msg.pose.orientation.x = -1
            msg.pose.orientation.y = -1
            msg.pose.orientation.z = -1
            msg.pose.orientation.w = -1
            msg.found = False
            print("Response in perception wrapper if grasp point is not found: ",msg)
            return msg


    def run(self):
        pass



if __name__ == '__main__':
    try:
        item_requirement = ItemRequirement()

        while not rospy.is_shutdown():
            item_requirement.run()
    except rospy.ROSInterruptException:
        pass
