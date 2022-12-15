#!/usr/bin/env python2

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
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker


import tf
import tf2_ros
import tf2_geometry_msgs

from chefbot.srv import ItemRequest, ItemRequestRequest, ItemRequestResponse

X_OFFSET = 0.05

def get_Trobot_camera(frame_id):
    robot_usbcam_world_matrix = [-0.08750836445666606, -0.9960946289977985, -0.01173781205575175, -0.33490360917941986, -0.9961426640035954, 0.08757716308957976, -0.005480278916361513, -0.025361254610593603, 0.006486840674719038, 0.011212965126053243, -0.9999160916352645, 0.8689006641118059, 0.0, 0.0, 0.0, 1.0]
    Trobot_camera = np.array([robot_usbcam_world_matrix]).reshape((4, 4))
    return Trobot_camera

class TransformedObjGraspPoint():
    """
    ROS node
    """

    def __init__(self):

        # Initialize the node
        print("initializing node")
        rospy.init_node('pixel_depth_to_robot_pose')
        
        # Subscribers
        rospy.Subscriber('chefbot/perception/grasp_pixel_depth', PointStamped, self.get_robot_pos_cb)
        rospy.Subscriber('/master/rgb/camera_info', CameraInfo, self.get_camera_info_cb)
        
        # Publisher
        self.transformed_obj_grasp_point_pub = rospy.Publisher('chefbot/perception/transformed_grasp_point',PoseStamped,queue_size=1)
        self.visualize_markers_pub = rospy.Publisher('/chefbot/perception/object_marker_array', Marker, queue_size=1)
        
        # get camera info
        self.Trobot_camera = get_Trobot_camera("workspace")
        
        # 
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(720.0))
        self.listener = tf2_ros.TransformListener(self.tf_buffer)    
        
        self.object_id = 0

    def get_camera_info_cb(self,rgb_camera_info):
        # Extract camera parameters
        self.cx = rgb_camera_info.K[2]
        self.fx = rgb_camera_info.K[0]
        self.cy = rgb_camera_info.K[5]
        self.fy = rgb_camera_info.K[4]
        
    def get_robot_pos_cb(self,point):
        self.object_id += 1
        pixel_x = point.point.x
        pixel_y = point.point.y
        pixel_depth = point.point.z 
        print("pixel x,y, depth: ",pixel_x, " ", pixel_y, " ", pixel_depth)
        
        # Use the pin hole camera model to calculate the real world position of the objects
        big_x = ((pixel_x * pixel_depth) - (self.cx * pixel_depth)) / self.fx
        big_y = ((pixel_y * pixel_depth) - (self.cy * pixel_depth)) / self.fy
        big_z = pixel_depth
        print("pinhole: ",big_x, " ", big_y, " ", big_z)
        # Assign the real world position of the grasp point (in meters) to a Point message
        ps = PoseStamped()
        ps.pose.position.x = big_x / 1000
        ps.pose.position.y = big_y / 1000 
        ps.pose.position.z = big_z / 1000 
        ps.header = point.header
        print("CHECK DISTANCE: ",self.get_distance(ps))
        
        # Transform to robot frame
        grasp_ps = np.array([ps.pose.position.x, ps.pose.position.y, ps.pose.position.z, 1])
        '''
        transformed_points = self.Trobot_camera.dot(grasp_ps)
        transformed_pos = PoseStamped()
        
        
        transformed_pos = PoseStamped()
        transformed_pos.header = point.header
        transformed_pos.pose.position.x = transformed_points[0]
        transformed_pos.pose.position.y = transformed_points[1] 
        transformed_pos.pose.position.z = transformed_points[2]
        transformed_pos.header = point.header
        '''
        transformed_pos = self.tf_buffer.transform(ps,"base_link",rospy.Duration(1))
        print("before: ",transformed_pos)
        # offset = 0.05
        transformed_pos.pose.position.x -= X_OFFSET
        # print("offset: ",offset)
        
        self.transformed_obj_grasp_point_pub.publish(transformed_pos)
        print("Transformed: ",transformed_pos)
        #transformed_pos.header.frame_id = "base_link"
        
        # Visualize the real world position of the object in RVIZ
        marker = self.get_ros_marker(transformed_pos, self.object_id, 1200000, 1200000)
        #marker.header = point.header
        self.visualize_markers_pub.publish(marker)
        print("PUB")
        return transformed_pos
    
    
    @staticmethod
    def get_distance(pose):
        '''
        Helper function to calculate distance between the center of the camera's frame and the predicted position of the objects
        Parameters:
                    pose (message): Message in the PoseStamped format
        Returns:
                    distance between the pose of the object the center of the camera_color_optical_frames assumed to be at (0, 0, 0)
        '''
        return math.sqrt(pose.pose.position.x ** 2 + pose.pose.position.y ** 2 + pose.pose.position.z ** 2)
    
    def get_ros_marker(self, ps, object_id, width, height):
        '''
        Helper function to vizualize the real-world position of the object in RVIZ

        Parameters:
                    ps (message): PoseStamped message with the real-world position of the object
                    object_id (int): Unique ID of the object
                    width (int): Width of the object in the pixel-space
                    height (int): Height of the object in the pixel-space
        Return:
                    marker (message): Marker message representing the vizualization of the object in RVIZ
        '''
        self.marker = Marker()
        self.marker.type = 1
        self.marker.action = 0
        #self.marker.lifetime = rospy.Duration(0.2)
        self.marker.text = "Detected Object"
        print("FI: ",ps.header.frame_id)
        self.marker.header.frame_id = ps.header.frame_id
        self.marker.header.stamp = ps.header.stamp
        self.marker.id = 0

        self.marker.scale.x = 0.01 #.05#height/1000
        self.marker.scale.y = 0.01 # .05#width/1000
        self.marker.scale.z = 0.01 # .05#0.005 # half a centimeter tall

        self.marker.color.r = 0
        self.marker.color.g = 255
        self.marker.color.b = 0
        self.marker.color.a = 1

        self.marker.pose.position.x = ps.pose.position.x
        self.marker.pose.position.y = ps.pose.position.y
        self.marker.pose.position.z = ps.pose.position.z
        print("PS: ",ps)

        self.marker.pose.orientation.x = 0
        self.marker.pose.orientation.y = 0
        self.marker.pose.orientation.z = 0
        self.marker.pose.orientation.w = 1   

        return self.marker
    
    def run(self):
        pass
        
    

if __name__ == '__main__':
    try:
        transformed_obj_grasp_point = TransformedObjGraspPoint()
        
        while not rospy.is_shutdown():
            transformed_obj_grasp_point.run()
    except rospy.ROSInterruptException:
        pass