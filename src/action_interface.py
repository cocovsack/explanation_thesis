#!/usr/bin/env python

import os
import rospy
import rospkg
import json
from sensor_msgs.msg import CompressedImage
from chefbot_utils.learning import DQNAgent, load_action_space
from geometry_msgs.msg import Point, Quaternion, PoseStamped, Pose
from std_msgs.msg import String, Header
from visualization_msgs.msg import Marker
from control_wrapper.srv import SetPose
from control_wrapper.srv import GetPose
from control_wrapper.srv import SetJoints
from control_wrapper.srv import SetTrajectory

from chefbot.srv import AbstractActionCommand, AbstractActionCommandRequest, AbstractActionCommandResponse, \
                        ActionCommand, ActionCommandRequest, AbstractActionCommandResponse, \
                        MasterItemRequirement, MasterItemRequirementRequest, MasterItemRequirementResponse, \
                        PickPlace, PickPlaceRequest, PickPlaceResponse, \
                        QueryFeedback, QueryFeedbackRequest, QueryFeedbackResponse

PKG_PATH = rospkg.RosPack().get_path("chefbot")
CONFIG_DIR = 'config'
ITEM_SHELF_MAPPING_JSON = 'item_shelf_mapping.json'
ACTION_STEP_MAPPING_JSON = 'action_step_mapping.json'
ITEM_CONTAINER_MAPPING_JSON = 'item_container_mapping.json'
ITEM_DEPTH_MAPPING_JSON = 'item_depth_mapping.json'

Y_OFFSET = 0 #.1 # in cm
DEPTH_OFFSET = 0.005
# self.move_robot = False


class ActionInterface():
    def __init__(self):
        self.move_robot = rospy.get_param('/move_robot', default=False)
        self.query_feedback_sp = None

        # Item Request service proxy (to perception nodes for inverse kinematics)
        print("Waiting for /chefbot/learning/master_item_requirement")
        rospy.wait_for_service('/chefbot/learning/master_item_requirement')
        self.master_item_requirement_sp = rospy.ServiceProxy('/chefbot/learning/master_item_requirement', MasterItemRequirement)

        # Query Feedback service proxy
        #print("Waiting for /chefbot/feedback/query_feedback")
        #rospy.wait_for_service('/chefbot/feedback/query_feedback')
        self.query_feedback_sp = rospy.ServiceProxy('/chefbot/feedback/query_feedback', QueryFeedback)

        if self.move_robot:
            print("Robot will move!!!!")
            # Action Command service proxy (to forward kinematics nodes)
            print("Waiting for /chefbot/learning/action_command")
            rospy.wait_for_service('/chefbot/learning/action_command')
            self.action_command_sp = rospy.ServiceProxy('/chefbot/learning/action_command', ActionCommand)

            # Pick Place service proxy
            print("Waiting for /chefbot/manipulation/pick_place")
            rospy.wait_for_service("/chefbot/manipulation/pick_place")
            self.pick_place_sp = rospy.ServiceProxy("/chefbot/manipulation/pick_place", PickPlace)

            # Robot position service proxy
            print("Waiting for /ur/control_wrapper/left/get_pose")
            rospy.wait_for_service("/ur/control_wrapper/left/get_pose")
            self.get_robot_pose_sp = rospy.ServiceProxy("/ur/control_wrapper/left/get_pose", GetPose)

            # Obtain the current robot pose
            self.current_pose = self.get_robot_pose_sp()

        print("All wait for services available!!")

        # Service that receives action command from reasoning
        self.abstract_action_srv = rospy.Service("chefbot/reasoner/next_action", AbstractActionCommand, self._abstract_action_srv_cb)

        # Text2Speech publisher
        self.tts_pub = rospy.Publisher('chefbot/tts', String, queue_size=10)

        self.visualize_markers_pub = rospy.Publisher('/chefbot/manipulation/destination_pose', Marker, queue_size=1)

        self.item_shelf_mapping = json.load(open(os.path.join(PKG_PATH, CONFIG_DIR, ITEM_SHELF_MAPPING_JSON)))
        self.action_step_mapping = json.load(open(os.path.join(PKG_PATH, CONFIG_DIR, ACTION_STEP_MAPPING_JSON)))
        self.item_container_mapping = json.load(open(os.path.join(PKG_PATH, CONFIG_DIR, ITEM_CONTAINER_MAPPING_JSON)))
        self.item_depth_mapping = json.load(open(os.path.join(PKG_PATH, CONFIG_DIR, ITEM_DEPTH_MAPPING_JSON)))

        self.item_in_workspace = []

    def _abstract_action_srv_cb(self, req):
        rospy.loginfo("[abstract_action_srv_cb] Recieved cmd: {}".format(req))
        action = req.action.data
        action_type = req.action_type.data
        item = req.item.data
        dest = req.dest.data
        command = None
        speak_command = None

        # action_type = "say"
        print()
        print("ACTION: ", action, "ITEM: ", item, "DEST: ", dest)
        if req.is_feedback.data:
            rospy.loginfo("Recieved feeback, skipping action.")
            return AbstractActionCommandResponse(True)

        if action == "gather":
            item_location = self.item_shelf_mapping[item]
            command = "Get item from row {}, column {}".format(item_location["row"], item_location["column"])
            ## send service request to robot directly
            print(command)
            action_cmd = ActionCommandRequest()
            action_cmd.command.data = command
            if self.move_robot:
               success = self.action_command_sp(action_cmd)
               self.send_robot_home("home position")
            #self.item_in_workspace.append(item)


        elif action == "pour":
            if item == "water":
                if action_type == "do":
                    command = "pour_" + item +"_from_measuring_cup_into_pan"
                    item = None
                    self.command_execution(command, item)
                if action_type == "say":
                    speak_command = "Pour the water from the cup into the pan!"
                    print("Saying: ", speak_command)
                    self.tts_pub.publish(speak_command)
            else:
                if dest == "pan":
                    if action_type == "do":
                        command = "pour_into_pan"
                        self.command_execution(command, item)
                    if action_type == "say":
                        speak_command = "Pour the " + item +" into the pan!"
                        print("Saying: ", speak_command)
                        self.tts_pub.publish(speak_command)
                elif dest == "bowl":
                    if action_type == "do":
                        command = "pour_into_bowl"
                        self.command_execution(command, item)
                    if action_type == "say":
                        speak_command = "Pour the " + item +" into the bowl!"
                        print("Saying: ", speak_command)
                        self.tts_pub.publish(speak_command)

        elif action == "putin":
            if dest == "microwave":
                if action_type == "do":
                    command = "putin_microwave"
                    nl_str = "put {item} in microwave"
                    self.command_execution(command, item)
                if action_type == "say":
                    speak_command = "Put the " + item +" in the microwave!"
                    print("Saying: ", speak_command)
                    self.tts_pub.publish(speak_command)
            elif dest == "sink":
                if action_type == "do":
                    command = "putin_sink"
                    self.command_execution(command, item)
                if action_type == "say":
                    speak_command = "Put the measuring cup under the sink!"
                    print("Saying: ", speak_command)
                    self.tts_pub.publish(speak_command)

        elif action == "collectwater":
            if action_type == "do":
                command = "collect_water"
                self.command_execution(command, item)
            if action_type == "say":
                speak_command = "Get water from the sink!"
                print("Saying: ", speak_command)
                self.tts_pub.publish(speak_command)


        elif action == "turnon":
            if item == "microwave":
                if action_type == "do":
                    command = "turnon_microwave"
                    self.command_execution(command, item)
                if action_type == "say":
                    speak_command = "Turn on the microwave!"
                    print("Saying: ", speak_command)
                    self.tts_pub.publish(speak_command)
            elif item == "stove":
                if action_type == "do":
                    command = "turnon_stove"
                    self.command_execution(command, item)
                if action_type == "say":
                    speak_command = "Turn on the stove!"
                    print("Saying: ", speak_command)
                    self.tts_pub.publish(speak_command)

        elif action == "takeoutmicrowave":
            if action_type == "do":
                command = "take_out_of_microwave"
                self.command_execution(command, item)
            if action_type == "say":
                speak_command = "Take the "+str(item)+" out of the microwave!"
                print("Saying: ", speak_command)
                self.tts_pub.publish(speak_command)

        elif action == "grabspoon":
            if action_type == "do":
                command = "grab_spoon"
                self.command_execution(command, item)

        elif action == "mix":
            if action_type == "do":
                command = "mix"
                self.command_execution(command, item)
            if action_type == "say":
                speak_command = "Mix the ingredients in the pan!"
                print("Saying: ", speak_command)
                self.tts_pub.publish(speak_command)

        elif action == "reduceheat":
            if action_type == "do":
                command = "reduce_heat"
                self.command_execution(command, item)
            if action_type == "say":
                speak_command = "Turn the heat on the stove down!"
                print("Saying: ", speak_command)
                self.tts_pub.publish(speak_command)

        elif action == "serveoatmeal":
            if action_type == "do":
                command = "serve_oatmeal"
                self.command_execution(command, item)
            if action_type == "say":
                speak_command = "Serve yourself some oatmeal from the pan!"
                print("Saying: ", speak_command)
                self.tts_pub.publish(speak_command)

        # print("Items in workspace: {} ".format(.item_in_workspace))
        if self.query_feedback_sp is not None:
            print("Checking for human feedback!!")
            response = self.query_feedback_sp()
        else:
            print("Cannot check for human feedback! Is process_feedback running?")
        return AbstractActionCommandResponse(True)

    def command_execution(self, command, item):

        # Visualize the real world position of the object in RVIZ

        skip_commands = []
        # skip_steps = ["Pour into pan", "Open microwave door", "Put ingredient in microwave",	"Close microwave"]
        skip_steps = []
        if command not in skip_commands:
            steps = self.action_step_mapping[command]["Steps"]
            manipulation = self.action_step_mapping[command]["Manipulation"]
            print()

            # self.send_robot_home("middle position")
            for i, (step, manip) in enumerate(zip(steps, manipulation)):
                if step not in skip_steps:
                    if manip == "FK":
                        # continue
                        print(step, "Command Robot")
                        # import pdb; pdb.set_trace()
                        action_cmd = ActionCommandRequest()
                        action_cmd.command.data = step
                        if self.move_robot:
                            # import pdb; pdb.set_trace()
                            success = self.action_command_sp(action_cmd)

                    if manip == "IK":
                        rospy.loginfo("[Action_interface] Checking if cameras are working...")
                        rospy.wait_for_message("/master/rgb/image_raw/compressed",
                                               CompressedImage)
                        # import pdb; pdb.set_trace()
                        # self.send_robot_home("middle position")
                        print("step: ", step)
                        pick_succ, place_succ = False, False
                        while not pick_succ:
                            response, container_depth = self._look_for_item(item)
                            if response.found:
                                pick_succ, place_succ = self._pick_place_wrapper(response,
                                                                                container_depth)
                            else:
                                print("No item found!!!")
            self.send_robot_home("home position")
        else:
            print("Skipping: ", command)

    def _pick_place_wrapper(self, response, container_depth):
        print("Item found!!")
        item_position = response.pose
        item_position.position.y += Y_OFFSET
        center_position = self.get_destination_pose_for_workspace_center()
        marker = self.get_ros_marker(center_position, 1, 1200000, 1200000)
        self.visualize_markers_pub.publish(marker)

        print("Received item position: ", item_position)

        if self.move_robot:
            self.send_robot_home("middle position")

            rospy.loginfo("Attempting pick and place....")
            pick_place_resp = self._pick_place(item_position,
                                                center_position.pose,
                                                container_depth)
            pick_succ = pick_place_resp.picked_object
            place_succ  = pick_place_resp.placed_object

            if not pick_succ:
                speak_command = "Failed to find action plan, re-attempting..."
                self.tts_pub.publish(speak_command)
        else:
            pick_succ, place_succ = True, True

        return pick_succ, place_succ


    def _pick_place(self, item_position, center_position, container_depth):
        # self.object_depth = item_position.position.z + DEPTH_OFFSET # temporary
        self.object_depth = container_depth - DEPTH_OFFSET
        self.object_pose = item_position
        self.start_pose = self.current_pose.pose
        self.destination_pose = center_position

        rospy.sleep(0.5)
        # import pdb; pdb.set_trace()
        # PickPlaceResponse has picked_object and placed_object bools.
        return self.pick_place_sp(start_pose = self.start_pose,
                                  object_pose = self.object_pose,
                                  destination_pose = self.destination_pose, # temporary
                                  object_depth = self.object_depth)

    def _look_for_item(self, item):
        container = self.item_container_mapping[item]
        container_type = container.split('_')[-1]
        container_depth = self.item_depth_mapping[container_type]
        print("Query Perception", item, container)
        if self.move_robot:
            self.send_robot_home("home position")
        # import pdb; pdb.set_trace()
        item_req = MasterItemRequirementRequest()
        item_req.item.data = container

        item_found = False
        while not item_found:
            rospy.loginfo("Looking for {}...".format(container))
            response = self.master_item_requirement_sp(item_req)
            item_found = response.found
            if not item_found:
                rospy.loginfo("Item not found. Please move it somewhere else...")
                # speak_command = "Item not found. Please move it somewhere else..."
                # self.tts_pub.publish(speak_command)
                rospy.sleep(1.0)
            else:
                return response, container_depth


    def send_robot_home(self, home_type):
        # home_type = middle position/home position
        command = str(home_type)
        print("Going to ", home_type)
        action_cmd = ActionCommandRequest()
        action_cmd.command.data = command
        if self.move_robot:
            success = self.action_command_sp(action_cmd)


    def get_destination_pose_for_workspace_center(self):
        dest_pose = PoseStamped()
        h = Header()
        h.stamp = rospy.Time.now()
        h.frame_id = "base_link"
        dest_pose.header = h
        dest_pose.pose.position.x = -0.38
        dest_pose.pose.position.y = -0.11
        dest_pose.pose.position.z = 0.29
        dest_pose.pose.orientation.x = -0.72
        dest_pose.pose.orientation.y = -0.68
        dest_pose.pose.orientation.z = 0.03
        dest_pose.pose.orientation.w = 0.06


        return dest_pose

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

        self.marker.color.r = 255
        self.marker.color.g = 0
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
        rospy.init_node('action_interface', anonymous=True)

        action_interface = ActionInterface()
        while not rospy.is_shutdown():
            action_interface.run()


    except rospy.ROSInterruptException:
        pass
