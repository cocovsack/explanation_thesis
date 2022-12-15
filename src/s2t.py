#!/usr/bin/env python
import rospy
from chefbot_utils.util import STT, ProcessCommand
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse, SetBool, SetBoolResponse
from chefbot.msg import Speech2Text

class ConvertSpeech2Text(object):
    def __init__(self):
        self.pause_detection = False
        ingredients = rospy.get_param("chefbot/ingredients",
                                      default=['oats', 'milk',
                                               'measuring cup', 'strawberry', 'blueberry',
                                               'banana', 'chocolate chips', 'salt', 'mixing spoon',
                                               'eating spoon', 'coco puffs', 'pie', 'muffin',
                                               'eggs', 'croissant', 'jelly pastry', 'bowl'])
        meal = rospy.get_param("chefbot/meal", default=['oatmeal', 'cereal'])
        side = rospy.get_param("chefbot/side", default=['pastry'])
        shelf = rospy.get_param("chefbot/shelf", default=['top', 'bottom'])
        nutritional = rospy.get_param("chefbot/nutritional", default=['dairy', 'vegan', "healthy"])
        dest = rospy.get_param("chefbot/dest", default=['pan', 'bowl', 'microwave'])
        self.pc = ProcessCommand(ingredients, meal, side, shelf, nutritional, dest)
        # self.stt = STT(7)
        self.stt = STT()

        # Create a publisher to the keyboard topic
        self.s2t_publisher = rospy.Publisher('/chefbot/speech2text_sync', Speech2Text, queue_size=1)

        # self.speech2text_srv = rospy.Service('/ur_xylophone_wodoto/speech2text_srv', Speech2Text, self.speech2text_srv_cb)

        self.pause_detection_srv = rospy.Service("/chefbot/pause_sync_detection_srv", SetBool, self.pause_detection_srv_cb)
        print("Node has started and speech detection is paused")

    def pause_detection_srv_cb(self, request):
        self.pause_detection = request.data
        print("Pause detection is now: " + str(self.pause_detection))
        return SetBoolResponse(True, "")
        # return True


    def run(self):
        rate = rospy.Rate(10)
        if not self.pause_detection:
            transcript = self.stt.get_transcript()
            rospy.loginfo("Recieved transctipt: " + transcript)
            s2t = Speech2Text()
            s2t.command = str(transcript)
            self.s2t_publisher.publish(s2t)

            if not self.pause_detection:  # we check this again because convert is a blocking function
                rate.sleep()  # This is here to prevent a race condition. Seems to work.
        else:
            rate.sleep()  # This is here to prevent a race condition. Seems to work.
                # self.s2t_publisher.publish(transcript)

def main():
    try:
        rospy.init_node("speech2text_sync")

        speech2text = ConvertSpeech2Text()


        while not rospy.is_shutdown():
            speech2text.run()
    except rospy.ROSInterruptException:
        pass
    


if __name__ == '__main__':
    main()

