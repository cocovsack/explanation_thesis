#!/usr/bin/env python
import rospy
from chefbot_utils.util import STT, ProcessCommand
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse, SetBool, SetBoolResponse
from chefbot.msg import Speech2Text
from chefbot_utils.util import TTS
from std_msgs.msg import String

class ConvertText2Speech(object):
    def __init__(self):
        self.pause_detection = False
        self.tts = TTS()
        
        self.tts_sub = rospy.Subscriber('chefbot/tts', String, self.tts_cb)
        
        print("Waiting for /chefbot/pause_sync_detection_srv")
        rospy.wait_for_service('/chefbot/pause_sync_detection_srv')
        self.pause_stt_sp = rospy.ServiceProxy('/chefbot/pause_sync_detection_srv', SetBool)
       
        print("Node has started and speech detection is paused")

    def tts_cb(self, msg):
        text = msg.data
        self.pause_stt_sp(True)
        print("Speaking: " + text)
        self.tts.say(text)
        self.pause_stt_sp(False)
        
    def run(self):
        pass

def main():
    rospy.init_node("text2speech")

    text2speech = ConvertText2Speech()

    while not rospy.is_shutdown():
        text2speech.run()


if __name__ == '__main__':
    main()

