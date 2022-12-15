#!/usr/bin/env python

import cv2
import numpy as np
import rospy
from collections import deque
from copy import deepcopy
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
from multiprocessing import Process, Lock
import scipy.misc

from chefbot.srv import ItemRequest, ItemRequestRequest, ItemRequestResponse


class ItemPerception():
    """
    ROS node
    """

    def __init__(self):

        # Initialize the nodevisualize_detected_object(object_mask)
        print("initializing node")
        rospy.init_node('item_perception')

        self.rgb_image = None

        # Subscriber
        # rospy.Subscriber('/master/rgb/image_raw/compressed', CompressedImage, self.get_image_cb)

        # Item Request Service
        self.item_request_srv = rospy.Service("/chefbot/learning/item_request", ItemRequest, self.item_request_srv_cb)

        # Publisher
        self.grasp_pixel_pub = rospy.Publisher('chefbot/perception/grasp_pixel',PointStamped,queue_size=5)
        self.detection_image_pub = rospy.Publisher('/chefbot/visualization/detection_image', Image, queue_size=5)
        self._num_frames = 5
        self._image_deque = deque()
        self.lock = Lock()


    # def get_image_cb(self, ros_rgb_image):
    #     if len(self._image_deque) >= self._num_frames:
    #         self._image_deque.popleft()

    #     self.rgb_image = np.frombuffer(ros_rgb_image.data,np.uint8)
    #     np_i = cv2.imdecode(self.rgb_image,cv2.IMREAD_COLOR)
    #     self.rgb_image_timestamp = ros_rgb_image.header.stamp
    #     self.rgb_image_header = ros_rgb_image.header

    #     # import pdb; pdb.set_trace()

    #     # with self.lock:
    #     print("[cb] locking deque")
    #     self.lock.acquire()
    #     self._image_deque.append(np_i)
    #     self.lock.release()
    #     print("[cb] releasing deque")
    #     rospy.sleep(.5)

    # def _block_until_full_image_deque(self):

    #     with self.lock:
    #         while len(self._image_deque) < self._num_frames:
    #             rospy.sleep(.1)

    #         rospy.loginfo("Image deque full!")
    #         return deepcopy(self._image_deque)



    def _get_image_from_msg(self, data):
        rgb_image = np.frombuffer(data.data,np.uint8)
        self.rgb_image_timestamp = data.header.stamp
        print("timestamp: {}".format(self.rgb_image_timestamp))
        self.rgb_image_header = data.header
        return cv2.imdecode(rgb_image,cv2.IMREAD_COLOR)

    def item_request_srv_cb(self, srv):
        item_request = srv.item.data
        # Collect self._num_frames frames to turn into a mask
        image_list = []
        while len(image_list) < self._num_frames:
            img = rospy.wait_for_message('/master/rgb/image_raw/compressed',
                                         CompressedImage)
            image_list.append(self._get_image_from_msg(img))
        print("Request: ",item_request)
        # Create one mask from these images and get grasp point.
        point, object_mask = handle_item_request(image_list,
                                                 item_request)
        self.visualize_detected_object(point, object_mask)
        print("Response: ",point)
        # import pdb; pdb.set_trace()
        msg = ItemRequestResponse()
        point_to_pub = PointStamped()
        if point is None:
            msg.point.x = -1
            msg.point.y = -1
            msg.found = False
            return msg
        else:
            # import pdb; pdb.set_trace()
            msg.point.x = point[0]
            msg.point.y = point[1]
            msg.found = True
            # publish point
            point_to_pub.point.x = point[0]
            point_to_pub.point.y = point[1]
            point_to_pub.header = self.rgb_image_header
            self.grasp_pixel_pub.publish(point_to_pub)

            return msg

    def visualize_detected_object(self, point, object_mask):
        print("Converting to color!")
        color_image = np.zeros((object_mask.shape[0], object_mask.shape[1], 3), dtype=np.uint8)
        color_image[:, :, 0] = object_mask
        color_image[:, :, 1] = object_mask
        color_image[:, :, 2] = object_mask
        if point is not None:
            print("Publishing image with grasp point superimposed")
            color_image = self.plot_grasp_point(color_image, point)
        # cv2.imwrite('/home/scazlab/test.png', color_image)
        # print("Saved!!")
        try:
            detection_image_msg = self.cv2_to_imgmsg(color_image)
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        detection_image_msg.header = self.rgb_image_header
        self.detection_image_pub.publish(detection_image_msg)
        print("Visualizing!")

    def plot_grasp_point(self, image, point):
        print(point, "grasp point inside plotter")
        x = point[0]
        y = point[1]
        grasp_point_image = cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return grasp_point_image



    def cv2_to_imgmsg(self, cv_image):
        '''
        Helper function to publish a cv2 image as a ROS message (without using ROS cv2 Bridge)
        https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/
        Parameters:
                    cv_image (image): Image to publish to a ROS message
        Returns:
                    img_msg (message): Image message published to a topic
        '''
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
        return img_msg

    def run(self):
        pass



if __name__ == '__main__':
    try:
        item_perception = ItemPerception()

        while not rospy.is_shutdown():
            item_perception.run()
    except rospy.ROSInterruptException:
        pass
