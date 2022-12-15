#!/usr/bin/env python

from __future__ import print_function
import rospy
import cv2
import numpy as np
from item_perception.src import hsv_slider as hsv
from sensor_msgs.msg import Image, CompressedImage
import ros_numpy



# global
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Image'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


cv2.namedWindow(window_capture_name)
cv2.namedWindow(window_detection_name)

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)
    print(f"{{\"lower_H\": {low_H}, \"upper_H\": {high_H}, \"lower_S\": {low_S}, \"upper_S\": {high_S}, \"lower_V\": {low_V}, \"upper_V\": {high_V}}}")

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)
    print(f"{{\"lower_H\": {low_H}, \"upper_H\": {high_H}, \"lower_S\": {low_S}, \"upper_S\": {high_S}, \"lower_V\": {low_V}, \"upper_V\": {high_V}}}")

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)
    print(f"{{\"lower_H\": {low_H}, \"upper_H\": {high_H}, \"lower_S\": {low_S}, \"upper_S\": {high_S}, \"lower_V\": {low_V}, \"upper_V\": {high_V}}}")

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)
    print(f"{{\"lower_H\": {low_H}, \"upper_H\": {high_H}, \"lower_S\": {low_S}, \"upper_S\": {high_S}, \"lower_V\": {low_V}, \"upper_V\": {high_V}}}")

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
    print(f"{{\"lower_H\": {low_H}, \"upper_H\": {high_H}, \"lower_S\": {low_S}, \"upper_S\": {high_S}, \"lower_V\": {low_V}, \"upper_V\": {high_V}}}")

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)
    print(f"{{\"lower_H\": {low_H}, \"upper_H\": {high_H}, \"lower_S\": {low_S}, \"upper_S\": {high_S}, \"lower_V\": {low_V}, \"upper_V\": {high_V}}}")

cv2.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

class ItemPerception():
    """
    ROS node
    """

    def __init__(self):

        # Initialize the nodevisualize_detected_object(object_mask)
        print("initializing node")
        rospy.init_node('hsv')

        self.rgb_image = None

        # Subscriber
        self.cv2_img = None
        rospy.Subscriber('/master/rgb/image_raw/compressed', CompressedImage, self.get_image_cb)


    def get_image_cb(self, ros_rgb_image):

        # self.rgb_image = np.frombuffer(ros_rgb_image.data,np.uint8)
        self.cv2_img = self._get_image_from_msg(ros_rgb_image)
        # cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        # import pdb; pdb.set_trace()


    def _get_image_from_msg(self, data):
        rgb_image = np.frombuffer(data.data,np.uint8)
        self.rgb_image_timestamp = data.header.stamp
        self.rgb_image_header = data.header
        return cv2.imdecode(rgb_image,cv2.IMREAD_COLOR)

    def run(self):
        # frame_HSV, frame_threshold = hsv.ros_main(self.cv2_img)
        # cv2.imshow("img", frame_HSV)

        frame_HSV = cv2.cvtColor(self.cv2_img, cv2.COLOR_BGR2HSV)
        frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
        resize_img = cv2.resize(self.cv2_img, (1280, 720))
        resize_frame_thresh = cv2.resize(frame_threshold, (1280, 720))
        # cv2.imshow(window_capture_name, self.cv2_img)
        # cv2.imshow(window_detection_name, frame_threshold)
        cv2.imshow(window_capture_name, resize_img)
        cv2.imshow(window_detection_name, resize_frame_thresh)

        cv2.waitKey(1)
        # cv2.destroyAllWindows()



if __name__ == '__main__':
    try:
        item_perception = ItemPerception()

        rospy.sleep(.5)
        while not rospy.is_shutdown():
            item_perception.run()
    except rospy.ROSInterruptException:
        pass
