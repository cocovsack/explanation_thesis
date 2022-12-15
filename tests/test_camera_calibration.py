#!/usr/bin/env python

import rospy
import numpy as np

from control_wrapper.srv import SetPose
from control_wrapper.srv import GetPose
from control_wrapper.srv import SetJoints
from control_wrapper.srv import SetTrajectory

from tri_star.msg import TargetPositions
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from tri_star.robot_util import Robot, pose_matrix_to_msg




def get_Trobot_camera(frame_id):
    robot_usbcam_world_matrix = rospy.get_param("{}_cam_matrix".format(frame_id))
    Trobot_camera = np.array([robot_usbcam_world_matrix]).reshape((4, 4))
    return Trobot_camera

class Controller(object):
    def __init__(self):
        #rospy.Subscriber('/aruco_single_usbcam_world_goal/pose', PoseStamped, self.get_points)
        self.pose_sub = rospy.Subscriber('/aruco_single_usbcam_world_goal/pose', PoseStamped, self.get_point_gt_diffs)
        self.transformed_object_pos_pub = rospy.Publisher('/chefbot/test/aruco_object_position', PoseStamped, queue_size=10)

        self.point = np.zeros(4)
        # 4x4 Transformation matrix
        self.Trobot_camera = get_Trobot_camera("workspace")  
        self.robot = Robot()     
        
    def get_aruco_point(self, data):
        "Get the centroid coords of the aruco code in camera frame"
        self.aruco_pos = data
        pos = data.pose.position
        print("aruco pos: ",pos)
        # Add 1 padding to make dimension consistent with Transformation matrix
        #self.point = np.array([pos.x,pos.y,pos.z, 1])
        
        # try this"
        self.point = np.array([pos.x,pos.y,.1, 1])
        print(".1")
        return self.point
        
    def get_end_effector_pos(self):
        "Get end effector coords in robot frame"
        pos = pose_matrix_to_msg(self.robot.get_robot_pose()).position
        # Add 1 padding to make dimension consistent with Transformation matrix
        return np.array([pos.x, pos.y, pos.z, 1])
    
    def get_point_gt_diffs(self, data):
        "Prints Comparison between aruco location in robot space (estimated) compared to ground truth"
        # Get aruco centroid and map it to robot frame
        aruco_pos = self.get_aruco_point(data)
        transformed_points = self.Trobot_camera.dot(aruco_pos)
        print("aruco: ", )
        transformed_pos = PoseStamped()
        transformed_pos.header = self.aruco_pos.header
        transformed_pos.pose.position.x = transformed_points[0]
        transformed_pos.pose.position.y = transformed_points[1] 
        transformed_pos.pose.position.z = transformed_points[2]
        self.transformed_object_pos_pub.publish(transformed_pos)
        # Robot should be holding aruco code so end effector should be close to a ground truth
        # location
        end_effector_pos = self.get_end_effector_pos()
        diffs = np.round(np.linalg.norm(transformed_points - end_effector_pos), 4)
        print("Error: {} m\n\t aruco: {}\n\t end effector: {}".format(diffs,
                                                                    transformed_points,
                                                                    end_effector_pos
                                                                            ))
                
    def run(self):
        pass


if __name__ == '__main__':
    try:
            rospy.init_node('test_calibration', anonymous=True)

            controller = Controller()
            while not rospy.is_shutdown():
                controller.run()


    except rospy.ROSInterruptException:
        pass
        
