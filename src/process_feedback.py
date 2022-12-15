#!/usr/bin/env python

import rospy
import rospkg
import numpy as np

from control_wrapper.srv import SetPose
from control_wrapper.srv import GetPose
from control_wrapper.srv import SetJoints
from control_wrapper.srv import SetTrajectory
from chefbot.srv import QueryFeedback, QueryFeedbackRequest, QueryFeedbackResponse, \
                        AbstractActionCommand, AbstractActionCommandRequest, AbstractActionCommandResponse, \
                        Overlay, OverlayRequest, OverlayResponse, \
                        Reward, RewardRequest, RewardResponse
from chefbot.msg import Speech2Text

from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
from tri_star.msg import TargetPositions
from geometry_msgs.msg import Point, Quaternion, PoseStamped, Pose
from tri_star.robot_util import Robot, pose_matrix_to_msg
from std_msgs.msg import String,Bool, Float32
from chefbot_utils.util import STT, ProcessCommand



class ProcessFeedback(object):
    def __init__(self):

        print("Waiting for chefbot/reasoner/corrective_action")
        rospy.wait_for_service("chefbot/reasoner/corrective_action")
        self._next_action_srv_sp = rospy.ServiceProxy("chefbot/reasoner/corrective_action", AbstractActionCommand)

        print("Waiting for chefbot/reasoner/overlay")
        rospy.wait_for_service("chefbot/reasoner/overlay")
        self._overlay_sp = rospy.ServiceProxy("chefbot/reasoner/overlay", Overlay)

        print("Waiting for chefbot/reasoner/reward")
        rospy.wait_for_service("chefbot/reasoner/reward")
        self._reward_sp = rospy.ServiceProxy("chefbot/reasoner/reward", Reward)

        print("Waiting for /chefbot/pause_sync_detection_srv")
        # rospy.wait_for_service('/chefbot/pause_sync_detection_srv')
        self.pause_stt_sp = rospy.ServiceProxy('/chefbot/pause_sync_detection_srv', SetBool)

        ingredients = rospy.get_param("chefbot/ingredients",
                                      default=['oats', 'milk',
                                               'measuring cup', 'strawberry', 'blueberry',
                                               'banana', 'chocolate chips', 'salt', 'mixing spoon',
                                               'eating spoon', 'coco puffs', 'pie', 'muffin',
                                               'egg', 'croissant', 'jelly pastry', 'bowl'])
        meal = rospy.get_param("chefbot/meal", default=['oatmeal', 'cereal'])
        side = rospy.get_param("chefbot/side", default=['pastry'])
        shelf = rospy.get_param("chefbot/shelf", default=['top', 'bottom'])
        nutritional = rospy.get_param("chefbot/nutritional", default=['dairy', 'nonvegan', "healthy", "sweet", "fruity", "protein",
                                                                      "gluten", "bread"])
        dest = rospy.get_param("chefbot/dest", default=['pan', 'bowl', 'microwave'])


        self.pc = ProcessCommand(ingredients, meal, side, shelf, nutritional, dest)

        self.s2t_sub = rospy.Subscriber('/chefbot/speech2text_sync', Speech2Text, self._speech2text_cb)
        self.command_prob_thresh = rospy.get_param("chefbot/command_prob_thresh",default=75)

        self.query_feedback_srv = rospy.Service('/chefbot/feedback/query_feedback',
                                                QueryFeedback, self._query_feedback_srv_cb)

        self.nl_command = None
        self.reward = None
        self.overlay = []
        print("All wait for services available!")

    def _query_feedback_srv_cb(self, srv):
        qf_resp = QueryFeedbackResponse()
        self.pause_stt_sp(False)
        if self.nl_command == None:
            print("No human feedback! Go for next action")
            qf_resp.success = True

            self.reward = 0
            reward_req = RewardRequest()
            reward_req.reward = self.reward
            self._reward_sp(reward_req)

        else:
            print("Got human feedback! Should have received a corrective action")
            qf_resp.success = False

            self.reward = -0.5
            reward_req = RewardRequest()
            reward_req.reward = self.reward
            self._reward_sp(reward_req)

        return qf_resp

    def _speech2text_cb(self, msg):
        self.nl_command = msg.command
        print("Received command: ", self.nl_command)
        # _, rules, _, act_dict = self.pc.process_command(self.nl_command)
        result_dict = self.pc.process_command(self.nl_command)
        # currently cmd_type is always overlay so hacking it
        # print("Abstract_action_dict: ", act_dict)
        if result_dict["score"] < self.command_prob_thresh:
            rospy.loginfo("Heard: {} but score was: {}".format(self.nl_command,
                                                               result_dict["score"]))
            return 
        if result_dict['type'] == "action":
            act_dict = result_dict["action_param_dict"]
            print("[process_feedback] Corrective Action, sending an AbstractAction service request")
            abs_act_req = AbstractActionCommandRequest()
            abs_act_req.action.data = act_dict['action']
            abs_act_req.item.data = act_dict['item']
            abs_act_req.dest.data = act_dict['dest']
            abs_act_req.action_type.data = act_dict['action_type']
            abs_act_req.is_feedback.data = True
            response = self._next_action_srv_sp(abs_act_req)
        elif result_dict['type'] == "overlay":
            print("Overlay, sending to reasoner")

            self.reward = 0.0
            reward_req = RewardRequest()
            reward_req.reward = self.reward
            self._reward_sp(reward_req)

            overlay_req = OverlayRequest()
            print(result_dict)
            overlay_req.overlay.extend(result_dict["rules"])
            overlay_req.type = result_dict["overlay_type"]
            overlay_req.name = result_dict["key"]
            overlay_req.transcript = result_dict["transcript"]
            if result_dict["overlay_type"] == "forget":
                overlay_req.params = result_dict["params"]
            print("Sending: {}".format(overlay_req))
            response = self._overlay_sp(overlay_req)

        self.nl_command = None
        self.reward = None
        self.overlay = []

    def run(self):
        pass

if __name__ == '__main__':
    try:
            rospy.init_node('process_feedback', anonymous=True)

            process_feedback = ProcessFeedback()
            while not rospy.is_shutdown():
                process_feedback.run()

    except rospy.ROSInterruptException:
        pass
