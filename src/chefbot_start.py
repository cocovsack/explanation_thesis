#!/usr/bin/env python2.7

import numpy as np

import rospy
import rospkg
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3

from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents, AllowedCollisionMatrix, AllowedCollisionEntry

from control_wrapper.msg import SceneObjectAllowCollision

from control_wrapper.srv import SetPose
from control_wrapper.srv import GetPose
from control_wrapper.srv import GetJoints
from control_wrapper.srv import SetJoints
from control_wrapper.srv import AddObject, AddObjectRequest
from control_wrapper.srv import AttachObject, AttachObjectRequest
from control_wrapper.srv import DetachObject, DetachObjectRequest
from control_wrapper.srv import RemoveObject, RemoveObjectRequest

from control_wrapper.srv import Reset, ResetRequest

class ChefBot():
    def __init__(self,side):
        #self.sub = rospy.Subscriber()
        self.boxes = [{'name':'desk_b',
                       'pose':Pose(Point(0.0946981512439, -0.758731188065, -0.00952404719815), 
                                   Quaternion(0, 0, 0, 1)),
                       'vector':Vector3(0.8626133208009, 1.486000809943, 0.00806735339044)},
                      
                      
                    #   {'name':'desk5',
                    #   'pose':Pose(Point(0.5, 0.5, -0.1), Quaternion(0.0, 0.0, 0.0, 1.0)),
                    #   'vector':Vector3(0.7, 0.4, 0.1)}
                      ]
        self.topic = "/ur/control_wrapper/" + side + "/"
        self.free_drive_pub = rospy.Publisher(self.topic + "enable_freedrive", Bool, queue_size=10)
        self.gripper_pub = rospy.Publisher(self.topic + "gripper", Bool, queue_size=10)
        self.connect_pub = rospy.Publisher(self.topic + "connect", Bool, queue_size=10)
        self.collision_pub = rospy.Publisher(self.topic + 'scene_allow_collision', SceneObjectAllowCollision, queue_size=10)
    

    def add_boxes(self):
        for box in self.boxes:
            rospy.wait_for_service(self.topic + "add_object")
            add_box = rospy.ServiceProxy(self.topic + "add_object", AddObject)
            try:
                name = box['name']
                pose = box['pose'] #Pose(Point(0.5, 0.5, -0.1), Quaternion(0.0, 0.0, 0.0, 1.0))
                size = box['vector'] #Vector3(0.7, 0.4, 0.1)
                object_type = AddObjectRequest.TYPE_BOX
                mesh_filename = ""
                response = add_box(name, pose, size, mesh_filename, object_type).is_success
                if response:
                    print("successfully added: ",name)
                else:
                    print("did not add successfully: ", name)
            except rospy.ServiceException as exc:
                print(("Service did not process request: " + str(exc)))   

    def set_initial_pos(self):
        print("attempting to move to chefbot start")
        rospy.wait_for_service(self.topic + "set_joints")
        set_joints = rospy.ServiceProxy(self.topic + "set_joints", SetJoints)
        try:
            joints = JointState()
            joint_names = ["shoulder_pan_joint", "shoulder_lift_joint",
                           "elbow_joint","wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
            joint_angles = np.deg2rad([64.0, -115.0, -62.0, 0.0, -6.0,  0.0]).tolist()

            joints.name = joint_names
            joints.position = joint_angles
            is_reached = set_joints(joints).is_reached

        except rospy.ServiceException as exc:
            print(("Service did not process request: " + str(exc)))
    
    def set_default_angles(self):
        print("attempting to move to default")
        rospy.wait_for_service(self.topic + "reset")
        reset = rospy.ServiceProxy(self.topic + "reset", Reset)
        try:
            response = reset().is_reached
        except rospy.ServiceException as exc:
            print(("Service did not process request: " + str(exc)))  

    def run(self):
        #self.set_initial_pos()
        #self.set_default_angles()
        self.add_boxes()
        pass

if __name__ == '__main__':
    try:
        rospy.init_node('chefbot_start',anonymous=True)
        chefbot = ChefBot("left")
        chefbot.run()

    except rospy.ROSInterruptException:
        print(rospy.ROSInterruptException)