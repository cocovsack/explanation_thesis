#!/usr/bin/env python

import rospy
import rospkg
import numpy as np

from control_wrapper.srv import SetPose
from control_wrapper.srv import GetPose
from control_wrapper.srv import SetJoints
from control_wrapper.srv import SetTrajectory
from chefbot.srv import PickPlace, PickPlaceRequest, PickPlaceResponse, \
                        ActionCommand, ActionCommandRequest, ActionCommandResponse

from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from tri_star.msg import TargetPositions
from geometry_msgs.msg import Point, Quaternion, PoseStamped, Pose 
from tri_star.robot_util import Robot, pose_matrix_to_msg
from std_msgs.msg import String, Bool, Float32


class ManipulationController(object):
    def __init__(self):

        # Action Command Service
        self.action_command_srv = rospy.Service("/chefbot/learning/action_command", ActionCommand, self.action_command_srv_cb)

        # Service Proxies to call all the manipulation services
        
        # Open Microwave Service Proxy
        rospy.wait_for_service("/chefbot/manipulation/open_microwave")
        self.open_microwave_sp = rospy.ServiceProxy("/chefbot/manipulation/open_microwave", SetBool)

        # Get Water Service Proxy
        rospy.wait_for_service("/chefbot/manipulation/get_water")
        self.get_water_sp = rospy.ServiceProxy("/chefbot/manipulation/get_water", SetBool)

                
        # close microwave
        
        # pour
        
        

    def action_command_srv_cb(self, srv):
        action_command = srv.command.data
        
        print(action_command)
        # import pdb
        # pdb.set_trace()
        if action_command == "open_microwave":
            response = self.open_microwave_sp()
        if action_command == "get_water":
            response = self.get_water_sp()
        
        
    
        


    def run(self):
        pass


if __name__ == '__main__':
    try:
            rospy.init_node('manipulation_controller', anonymous=True)

            manipulation_controller = ManipulationController()
            while not rospy.is_shutdown():
                manipulation_controller.run()


    except rospy.ROSInterruptException:
        pass
