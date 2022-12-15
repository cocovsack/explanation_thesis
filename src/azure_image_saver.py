#!/usr/bin/env python

import rospy
import numpy as np
# ROS Image message
from sensor_msgs.msg import CompressedImage
# OpenCV for saving an image
import cv2
import calendar
import time
import keyboard

image_prefix = "/home/scazlab/chefbot_catkin_ws/src/chef-bot/src/item_perception/images/image_"

class ImageSaver(object):
    def __init__(self):
        # Set up subscriber with it's callback function for saving images
        rospy.Subscriber('/master/rgb/image_raw/compressed', CompressedImage, self.get_azure_camera_image)
        self.image = None
        
    def get_azure_camera_image(self, msg):
        "Obtains images from the Azure Kinetic Camera and saves them"
        # Direct conversion to CV2
        np_arr = np.fromstring(msg.data, np.uint8)
        self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
    
    def run(self):
        while True:
            prompt = input("Press \'Enter\' to save an image:")
            cv2.imwrite(image_prefix + str(calendar.timegm(time.gmtime())) + ".jpeg", self.image)
            print("Saved an image!")
            
    
if __name__ == '__main__':
    try:
        rospy.init_node('image_listener', anonymous=True)

        image_saver = ImageSaver()
        while not rospy.is_shutdown():
            image_saver.run()
    except rospy.ROSInterruptException:
        pass