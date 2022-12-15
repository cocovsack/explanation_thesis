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
from chefbot.srv import ItemRequest, ItemRequestRequest, ItemRequestResponse
from item_perception.item_perception_logic import handle_item_request
from geometry_msgs.msg import PointStamped

from chefbot.srv import ItemRequest, ItemRequestRequest, ItemRequestResponse


class GetDepthForPixel():
    """
    ROS node
    """

    def __init__(self):

        # Initialize the node
        print("initializing node")
        rospy.init_node('get_depth_for_pixel')
        self.rgb_image = None
        self.depth_image = None
        
        # Subscribers with synchronization
        self.rgb_img_sub = message_filters.Subscriber('/master/rgb/image_raw/compressed', CompressedImage)
        self.depth_img_sub = message_filters.Subscriber('/master/depth_to_rgb/image_raw/compressed',CompressedImage)
        
        self.synchronizer = message_filters.TimeSynchronizer(
            [self.rgb_img_sub,self.depth_img_sub], 10)
        self.synchronizer.registerCallback(self.get_image_synch_cb)
        
        self.pixel_grasp_sub = message_filters.Subscriber('chefbot/perception/grasp_pixel', PointStamped)
        
        rospy.Subscriber('chefbot/perception/grasp_pixel', PointStamped, self.get_pixel_depth_cb)
        
        # Publisher
        self.grap_pixel_depth_pub = rospy.Publisher('chefbot/perception/grasp_pixel_depth',PointStamped,queue_size=5)
        

        
    def get_image_synch_cb(self,ros_rgb_image,ros_depth_image):
        self.rgb_image = np.frombuffer(ros_rgb_image.data,np.uint8)
        self.rgb_image_timestamp = ros_rgb_image.header.stamp
        self.depth_image = np.frombuffer(ros_depth_image.data,np.uint8)
        self.depth_image_timestamp = ros_depth_image.header.stamp
        
        
    def get_pixel_depth_cb(self,point):
        pixel_x = point.point.x
        pixel_y = point.point.y
        
        depth_matrix = cv2.imdecode(self.depth_image,cv2.IMREAD_UNCHANGED)
        #print(depth_matrix.shape)
        #print(pixel_x)
        #print(pixel_y)
        depth_at_pixel = depth_matrix[int(pixel_y)][int(pixel_x)]
        #print(depth_matrix[int(pixel_y-10):int(pixel_y+10),int(pixel_y-10):int(pixel_y+10)])
        
        pixel_point = PointStamped()
        pixel_point.point.x = pixel_x
        pixel_point.point.y = pixel_y
        pixel_point.point.z = 890.0 #depth_at_pixel
        pixel_point.header = point.header
        
        self.grap_pixel_depth_pub.publish(pixel_point)
        return pixel_point
    
    def run(self):
        pass
        
    

if __name__ == '__main__':
    try:
        get_depth_for_pixel = GetDepthForPixel()
        
        while not rospy.is_shutdown():
            get_depth_for_pixel.run()
    except rospy.ROSInterruptException:
        pass