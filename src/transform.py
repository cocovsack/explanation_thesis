#!/usr/bin/env python2

import rospy  # If you are doing ROS in python, then you will always need this import
import os
# Message imports go here
import geometry_msgs.msg
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import PoseStamped
# Service imports go here

# All other imports go here
import tf
import tf2_ros
import tf2_geometry_msgs


# Hyper-parameters go here
SLEEP_RATE = 10


class TransformPose():
    """
    Note: In the init function, I will first give a description of each part, and then I will give an example
    """

    def __init__(self):
        # Everything besides pubs, subs, services, and service proxies go here
        print("Starting!!!")

        

        # Subscribers go here
        self.image_sub = rospy.Subscriber('/master/rgb/image_raw/compressed', CompressedImage, self.get_image_cb)
        #self.real_pose_sub = rospy.Subscriber('/recycling_stretch/real_pose', RealPose, self.real_pose_cb)
        #self.marker_pose_sub = rospy.Subscriber('/recycling_stretch/marker_position', ArucoRefMarker, self.aruco_marker_cb)
        self.object_pose_time = rospy.Time.now()
        # Services go here

    
        rospy.spin()

    def get_image_cb(self, msg):
        
        self.image_time = msg.header.stamp
        #self.static_transform_broadcaster("base_link", "master_rgb_camera_link", -0.715254900591562, -0.021675664628654803, 0.4947174605478077, -0.503332, 0.5402831, -0.4871362, 0.4663148)
        #self.static_transform_broadcaster("base_link", "master_depth_camera_link", -0.288514225711, -0.00633827855991, 1.05471662968, 3.14, 0, -1.57)
        #self.static_transform_broadcaster("base_link", "master_depth_camera_link", -0.354166414551, 0.0160639336734, 0.905782344544, 3.05, 0, -1.57)
        #self.static_transform_broadcaster("base_link", "master_depth_camera_link", -0.354166414551, 0.00300639336734, 0.902782344544, 3.05, 0, -1.57)
        self.static_transform_broadcaster("base_link", "master_depth_camera_link", -0.337754000796, -0.00542571810713, 0.917021089607, 3.05, 0, -1.57)
    
    def static_transform_broadcaster(self, parent_frame, child_frame, x, y, z, roll,pitch,yaw):#q0, q1, q2, q3):
        broadcaster = tf2_ros.StaticTransformBroadcaster()

        static_transformStamped = geometry_msgs.msg.TransformStamped()

        static_transformStamped.header.stamp = rospy.Time(0)
        static_transformStamped.header.frame_id = parent_frame
        static_transformStamped.child_frame_id = child_frame

        static_transformStamped.transform.translation.x = x
        static_transformStamped.transform.translation.y = y
        static_transformStamped.transform.translation.z = z

        quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        q0,q1,q2,q3 = quat
        static_transformStamped.transform.rotation.x = q0
        static_transformStamped.transform.rotation.y = q1
        static_transformStamped.transform.rotation.z = q2
        static_transformStamped.transform.rotation.w = q3
        # print(static_transformStamped)
        broadcaster.sendTransform(static_transformStamped)
            


def main():
    rospy.init_node("transform_pose")

    transform_pose = TransformPose()

    rate = rospy.Rate(SLEEP_RATE)

    # This while loop is what keeps the node from dieing
    while not rospy.is_shutdown():
        # If I wanted my node to constantly publish something, I would put the publishing here

        rate.sleep()


if __name__ == '__main__':
    main()