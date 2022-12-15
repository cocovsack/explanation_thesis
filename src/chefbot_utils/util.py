#!/usr/bin/env python

# from __future__ import division

import os
import rospy
import re
import sys
import subprocess
import time
import rospkg
import pyaudio

import sys, contextlib

from fuzzywuzzy import fuzz
from itertools import product, combinations


from chefbot.srv import AbstractActionCommand, AbstractActionCommandRequest, AbstractActionCommandResponse

import numpy as np

from google.cloud import texttospeech

from google.cloud import speech
# from google.cloud.speech import enums
from playsound import playsound


from six.moves import queue


PKG_NAME = "chefbot"
RATE = 48000 #16000
CHUNK = int(RATE / 10)  # 100ms

PKG_PATH = rospkg.RosPack().get_path(PKG_NAME)
GOOGLE_KEY_PATH = os.path.join(PKG_PATH, "chefbot-google-key.json")

@contextlib.contextmanager
def ignoreStderr():
    """
    Code for supressing annoying (but harmless) clang ALSA warning/error printouts
    reference: https://stackoverflow.com/questions/7088672/pyaudio-working-but-spits-out-error-messages-each-time
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

def get_Trobot_camera(frame_id):
    robot_usbcam_world_matrix = rospy.get_param("{}_cam_matrix".format(frame_id))
    Trobot_camera = np.array([robot_usbcam_world_matrix]).reshape((4, 4))
    return Trobot_camera


def transform_points_camera_to_robot(T, point):
    #T = get_Trobot_camera(frame_id)
    if point.ndim == 1: # A single point
        # T matrix is 4x4 so add 1 padding
        new_point = np.ones(4)
        new_point[:3] = point

        point = new_point
    else: # We have a matrix of points
        new_point = np.ones((point.shape[0], 4))
        new_point[:, :3] = point

        point = new_point

    return point.dot(T)


class TTS(object):
    """A basic TTS class using Google's TTS API."""
    def __init__(self):
        # Load google credentials
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_KEY_PATH

        self.client = texttospeech.TextToSpeechClient()
        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
        # Select the type of audio file you want returned
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3)

    def say(self, txt):
        synthesis_input = texttospeech.SynthesisInput(text=txt)
        response = self.client.synthesize_speech(input=synthesis_input,
                                                  voice=self.voice,
                                                  audio_config=self.audio_config)

        tmp_path = os.path.join(PKG_PATH)
        tmp_file  = os.path.join(tmp_path, "output.mp3")

        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)

        with open(tmp_file, "wb") as out:
            # Write the response to the output file.
            out.write(response.audio_content)
            # print('Audio content written to file "output.mp3"')

        playsound(tmp_file) # play the audio file
        os.remove(tmp_file) # delete it as we dont actually need it.




"""Google Cloud Speech API sample application using the streaming API.

NOTE: This module requires the additional dependency `pyaudio`. To install
using pip:

    pip install pyaudio

Example usage:
    python transcribe_streaming_mic.py
"""

# [START speech_transcribe_streaming_mic]

# Audio recording parameters
MIC_NAME = "Scarlett"

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk, audio_device_id=None):
        self._rate = rate
        self._chunk = chunk
        if audio_device_id is None:
            self._audio_device_id = self._get_audio_device_id()
        else:
            self._audio_device_id = audio_device_id

        # Create a thread-safe buffer of audio data

        self._buff = queue.Queue()
        self.closed = True

    def _get_audio_device_id(self):

        with ignoreStderr():
            self._audio_interface = pyaudio.PyAudio()
            info = self._audio_interface.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            ret_idx = 0

            for i in range(0, numdevices):
                if (self._audio_interface.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    name = self._audio_interface.get_device_info_by_host_api_device_index(0, i).get('name')

                    name = name.split(" ")[0]
                    print("Input Device id ", i, " - ", name)
                    if MIC_NAME in name:
                        ret_idx = i
                        return ret_idx
                    else:
                        print("Cant find: ", MIC_NAME)
            return ret_idx

    def __enter__(self):
        with ignoreStderr(): # Supress ALSA error printouts.
            self._audio_stream = self._audio_interface.open(
                format=pyaudio.paInt16,
                # The API currently only supports 1-channel (mono) audio
                # https://goo.gl/z757pE
                channels=1, rate=self._rate,
                input=True, frames_per_buffer=self._chunk,
                input_device_index=self._audio_device_id,
                # Run the audio stream asynchronously to fill the buffer object.
                # This is necessary so that the input device's buffer doesn't
                # overflow while the calling thread makes network requests, etc.
                stream_callback=self._fill_buffer,
            )

            self.closed = False

            return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            #print(chunk, "chunk")
            if chunk is None:
                return
            data = [chunk]
            #print(data)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            #print(b''.join(data))
            yield b''.join(data)


class STT(object):
    def __init__(self, device_id=None):
        "docstring"

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_KEY_PATH

        self._device_id = device_id
        # if device_id is None:
        #     self._device_id = self._get_device_id()
        # else:
        #     self._device_id = device_id


    def _get_device_id(self):

        for i in range(0, numdevices):
            if (self._audio_interface.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ",
                    self._audio_interface.get_device_info_by_host_api_device_index(0, i).get('name'))
    def _get_device_id_bak(self):
        """
        Automatically gets the device id.
        """
        DEV_NAME = "alsa_input.usb-Focusrite_Scarlett_Solo_USB"
        o = subprocess.check_output(["pactl", "list", "short", "sources"], text=True)
        for dev in o.split("\n"):
            dev_id, name =  dev.split("\t")[:2]
            if DEV_NAME in name:
                print("Found device id: ", dev_id)
                return int(dev_id)


    def _listen_print_loop(self, responses):
        """Iterates through server responses and prints them.

        The responses passed is a generator that will block until a response
        is provided by the server.

        Each response may contain multiple results, and each result may contain
        multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
        print only the transcription for the top alternative of the top result.

        In this case, responses are provided for interim results as well. If the
        response is an interim one, print a line feed at the end of it, to allow
        the next result to overwrite it, until the response is a final one. For the
        final one, print a newline to preserve the finalized transcription.
        """
        print("inside listen print loop")
        # import pdb; pdb.set_trace()
        num_chars_printed = 0
        for response in responses:
            if not response.results:
                continue

            # The `results` list is consecutive. For streaming, we only care about
            # the first result being considered, since once it's `is_final`, it
            # moves on to considering the next utterance.
            result = response.results[0]
            if not result.alternatives:
                continue

            # Display the transcription of the top alternative.
            transcript = result.alternatives[0].transcript

            # Display interim results, but with a carriage return at the end of the
            # line, so subsequent lines will overwrite them.
            #
            # If the previous result was longer than this one, we need to print
            # some extra spaces to overwrite the previous result
            overwrite_chars = ' ' * (num_chars_printed - len(transcript))

            if not result.is_final:

                #sys.stdout.write(transcript + overwrite_chars + '\r') ## uncomment for command line output
                #sys.stdout.flush()

                # string_transcript = transcript.encode("utf-8") ## get_transcript unicode string to string
                #print(type(string_transcript))
                # print(string_transcript)

                num_chars_printed = len(transcript)

                # return string_transcript

            else:
                string_transcript = transcript ## get_transcript unicode string to string
                #print(type(string_transcript))
                print(string_transcript)
                print(overwrite_chars)
                num_chars_printed = len(transcript)
                #print(transcript + overwrite_chars)

                # Exit recognition if any of the transcribed phrases could be
                # one of our keywords.
                if re.search(r'\b(exit|quit)\b', transcript, re.I):
                    print('Exiting..')
                    break

                num_chars_printed = 0

                return (str(string_transcript) + str(overwrite_chars))


    def get_transcript(self):
        # See http://g.co/cloud/speech/docs/languages
        # for a list of supported languages.

        transcript_array = []

        language_code = 'en-US'  # a BCP-47 language tag
        speech_contexts=speech.SpeechContext(phrases=['$OPERAND', "oatmeal", "pastry",
                                                            "fruity", "healty", "make"])

        speech.SpeechContext()
        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code=language_code,
            # enable_automatic_punctuation=True,
            speech_contexts=[speech_contexts]
            )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True)
        #while time.time() - begin_time <= timeout:
        print("Opening microphone stream...")
        with MicrophoneStream(RATE, CHUNK, self._device_id) as stream:
            #print(time.time() - begin_time)
            audio_generator = stream.generator()
            #for content in audio_generator:
            #    requests = (speech.StreamingRecognizeRequest(audio_content=content))
            requests = (speech.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)
            responses = client.streaming_recognize(streaming_config, requests)
            # Now, put the transcription responses to use.
            transcript = self._listen_print_loop(responses)
            return transcript


# [END speech_transcribe_streaming_mic]




class ProcessCommand(object):
    """
    Possible commands:
    - "The next <k> pieces should be <color>"
    - "Don't get any <color> pieces
    - Let's build a <color> chair
    - The next [<k> chairs| chair] should be <color>

    """



    PERMIT_ACTION   = "action"
    PROHIBIT_ACTION = '~action'
    PASS_THROUGH    = 'all'
    OVERLAY_TYPE = "overlay"
    COMMAND_TYPE = "command"
    PERMIT   = "permit"
    PROHIBIT = "prohibit"
    TRANSFER = "transfer"
    FORGET = "forget"
    STOP_WORDS = ["the", "with", "high", "rest"]


    def __init__(self, ingredients, meal, side, shelf, nutritional, dest, thresh = 0.5):
        self.ingredients = ingredients
        self.meal = meal
        self.side = side
        self.shelf = shelf
        self.nutritional = nutritional
        self.meal_side = self.meal + self.side
        self.dest = dest
        self.appliance = ["stove", "microwave"]

        self.num2word = {1:'one',2:'two',3:"three",4:'four',
                         5:'five',6:'six',7:'seven', 8:'eight',
                         9:'nine',10:'ten',11:'eleven',12:'twelve',
                         13:'thirteen'}

        self.word2num = {v:k for k,v in self.num2word.items()}

        self.overlay_template_dict = {"Ov_0": ("first let's make {meal_side_1} and then let's make {meal_side_2}", self.PERMIT),
                                      "Ov_1": ("let's make something {nutritional}", self.PERMIT),
                                      "Ov_2": ("don't use any ingredients from the {shelf} shelf", self.PROHIBIT),
                                      "Ov_3": ("you make the {meal_side}",self.TRANSFER),
                                      "Ov_4": ("i'll make the {meal_side}",self.TRANSFER),
                                      "Ov_5": ("you make the {meal_side_1} and i'll make the {meal_side_2}", self.TRANSFER),
                                      # "Ov_5": "bring me a {ingredients}",
                                      "Ov_6": ("dont help me", self.TRANSFER),
                                      "Ov_7": ("forget rule {rule}", self.FORGET),
                                      "Ov_8": ("forget the last rule", self.FORGET),
                                      "Ov_9": ("don't use anything {nutritional}", self.PROHIBIT),
                                      }

        """
            std_msgs/String action
            std_msgs/String action_type
            std_msgs/String item
            std_msgs/String dest
        """
        self.action_template_dict = {"gather": "gather the {item}",
                                  "pourwater": "{pronoun} pour water in {dest}",
                                  # "pour": "{pronoun} pour {item} in {dest}",
                                  "pour": "{pronoun} add the {item} to the {dest}",
                                  "putin": "{pronoun} insert the {item} in {dest}",
                                  "turnon": "{pronoun} turn on {appliance}",
                                  "collectwater": "{pronoun} collect water",
                                  "takeoutmicrowave": "{pronoun} remove {item} from microwave",
                                  "grabspoon": "{pronoun} grab the mixing spoon",
                                  "mix": "{pronoun} mix the oatmeal",
                                  "reduceheat": "{pronoun} reduce the heat",
                                     "serveoatmeal": "{pronoun} serve the oatmeal",
                                     "complete": "the {meal_side} is complete"
                                  }
        self.template_type_dict = {
            self.OVERLAY_TYPE: ["Ov_0", "Ov_1", "Ov_2", "Ov_5", "Ov_6", "Ov_7"],
            self.COMMAND_TYPE: ["Ov_3", "Ov_5"]
        }

        self.thresh = thresh


    def _gen_corrective_commands_from_template(self, action, template):
        action_type_dict = {"do": "you", "say": "I"}
        ret_dict = {"action_type":"", "dest":"", "item":"", "action":""}
        if action in ["gather"]:
            for i in self.ingredients:
                ret_dict = {"action_type":"do", "dest":"", "item":"", "action":""}
                ret_dict["action"] = action
                ret_dict["item"] = i
                yield template.format(item=i), ret_dict
        elif action in ["turnon"]:
            for action_type, pronoun in action_type_dict.items():
                for a in self.appliance:
                    ret_dict = {"action_type":"", "dest":"", "item":"", "action":""}
                    ret_dict["action"] = action
                    ret_dict["item"] = a
                    ret_dict["action_type"] = action_type
                    yield (template.format(appliance=a,pronoun=pronoun), ret_dict)

        elif action in ["mix", "reduceheat", "serveoatmeal", "grabspoon", "collectwater"]:
            for action_type, pronoun in action_type_dict.items():
                ret_dict = {"action_type":"", "dest":"", "item":"", "action":""}
                ret_dict["action"] = action
                ret_dict["action_type"] = action_type
                yield template.format(pronoun=pronoun), ret_dict
        elif action in ["takeoutmicrowave"]:
            for i in self.ingredients:
                    for action_type, pronoun in action_type_dict.items():
                        ret_dict = {"action_type":"", "dest":"", "item":"", "action":""}
                        ret_dict["action"] = action
                        ret_dict["item"] = i
                        ret_dict["action_type"] = action_type
                        yield (template.format(item=i,pronoun=pronoun),
                               ret_dict)

        elif action in ["complete"]:
            for m in self.meal_side:
                    ret_dict = {"action_type":"", "dest":"", "item":"", "action":""}
                    ret_dict["item"] = m
                    ret_dict["action"] = action
                    yield (template.format(meal_side=m), ret_dict)

        elif action in ["pourwater"]:
            for dest in self.dest:
                for action_type, pronoun in action_type_dict.items():
                    ret_dict = {"action_type":"", "dest":"", "item":"", "action":""}
                    ret_dict["action"] = action
                    ret_dict["dest"] = dest
                    ret_dict["action_type"] = action_type
                    yield (template.format(pronoun=pronoun,dest=dest),
                            ret_dict)

        else:
            for i in self.ingredients:
                for dest in self.dest:
                    for action_type, pronoun in action_type_dict.items():
                        ret_dict = {"action_type":"", "dest":"", "item":"", "action":""}
                        ret_dict["action"] = action
                        ret_dict["dest"] = dest
                        ret_dict["item"] = i
                        ret_dict["action_type"] = action_type
                        yield (template.format(item=i,pronoun=pronoun,dest=dest),
                               ret_dict)

    def _gen_overlay_commands_from_template(self, name, template):

        max_num = max(list(self.num2word.keys()))
        # print(name)
        if name in ["Ov_1", "Ov_9"]:
            for n in self.nutritional:
                yield template.format(nutritional=n).lower(), [n]

        if name in ["Ov_7"]:
            for k in range(1, len(self.overlay_template_dict) + 1):
                yield template.format(rule=self.num2word[k]).lower(), k
        elif name in ["Ov_3", "Ov_4"]:
            for m1 in self.meal_side:
                    yield template.format(meal_side=m1).lower(), [m1]
        elif name in ["Ov_0", "Ov_5"]:
            for m1 in self.meal_side:
                for m2 in self.meal_side:
                    yield template.format(meal_side_1=m1, meal_side_2=m2).lower(), [m1, m2]
        elif name in ["Ov_8"]:
            yield template, ["last"]
        else: #
            for p in product(self.meal, self.side, self.shelf, self.nutritional):
                m, s, sh, n = p
                yield template.format(meal=m, side=s, shelf=sh, nutritional=n).lower(), p


    def _score_template(self, command, name, template, template_type="overlay"):
        results = []
        if template_type == "overlay":
            cmd_gen = self._gen_overlay_commands_from_template
        elif template_type == "action":
            cmd_gen = self._gen_corrective_commands_from_template

        for t, var in cmd_gen(name, template):
            # score = fuzz.partial_ratio(command, t)
            score = fuzz.ratio(command, t)
            # print("t: {} score: {}".format(t, score))
            results.append((t, var,  score))

        return max(results, key=lambda x: x[2])

    def _command_to_rule(self, template_key, var, command):

        # ROBOT_DO_TEMP = "(do(A_out) or say_only(A_out)) and making_{dish}(A_out)"
        ROBOT_DO_TEMP = "do(A_out) and making_{dish}(A_out)"
        # ROBOT_SAY_TEMP = "(say(A_out) or do_only(A_out)) and making_{dish}(A_out)"
        ROBOT_SAY_TEMP = "say(A_out)  and making_{dish}(A_out)"
        for num, word in self.num2word.items():
            if word in command:
                k = num

        if template_key == 'Ov_0':
            m1, m2 = var
            first_pred = "making_{dish1}(A_out) and state(no_completed_dish)".format(dish1=m1)
            last_pred = "making_{dish2}(A_out) and state(one_completed_dish)".format(dish2=m2)
            rule  = "not state(two_completed_dish) then ({first} or {last})".format(first=first_pred,
                                                                               last=last_pred)
            # rules = ["is_making_{meal}(A_out) or is_making_{side}(A_out) -> action(A_out)".format(
            #     meal=meal, side=side)]
            rules = [rule]
        elif template_key == 'Ov_1':
            nutr = var[0]
            # rules = ["{nutr}(A_out) -> action(A_out)".format(nutr=nutr)]
            rules = ["{nutr}(A_out) or {nutr}_precursor(A_out)".format(nutr=nutr)]
        elif template_key == 'Ov_2':
            _, _, shelf, _ = var
            shelf = shelf.split() + ["shelf"]
            shelf = "_".join(shelf)
            rules = ["not {}(A_out)".format(shelf)]
        elif template_key == "Ov_3":
            dish = var[0]
            robot_do_pred = ROBOT_DO_TEMP.format(dish=dish)
            rule = "not state(two_completed_dish) then equiv_action(A_in, A_out) and {pred}".format(pred=robot_do_pred)
            rules = [rule]
        elif template_key == "Ov_4":
            dish = var[0]
            robot_say_pred = ROBOT_SAY_TEMP.format(dish=dish)
            rule = "not state(two_completed_dish) then equiv_action(A_in, A_out) and {pred}".format(pred=robot_say_pred)
            rules = [rule]
        elif template_key == 'Ov_5':
            m1, m2 = var
            robot_do_preds = ROBOT_DO_TEMP.format(dish=m1)
            robot_say_preds = ROBOT_SAY_TEMP.format(dish=m2)
            rule1 = "not state(two_completed_dish) then (equiv_action(A_in, A_out) and (({}) or ({}) ))".format(robot_do_preds, robot_say_preds)
            rules = [rule1]

        elif template_key == 'Ov_8':
            rules = ["forget the last rule"]
        elif template_key == 'Ov_9':
            nutr = var[0]
            rules = ["not state(two_completed_dish) then not {nutr}(A_out)".format(nutr=nutr)]

        else:
            rules = ["foo"]


        return rules

    def _rule_from_command(self, command):
        overlay_score_dict = {}
        action_score_dict = {}

        for name, temp_vars in self.overlay_template_dict.items():
            temp, overlay_type = temp_vars
            est_command, var ,score = self._score_template(command, name, temp,
                                                           template_type="overlay")
            overlay_score_dict[name] = (est_command,  var, score)
        for name, temp in self.action_template_dict.items():
            est_command, var ,score = self._score_template(command, name, temp,
                                                           template_type="action")
            action_score_dict[name] = (est_command,  var, score)

        overlay_score_list = [(cmd, score) for cmd, _, score in overlay_score_dict.values()]
        action_score_list = [(cmd, score) for cmd, _, score in action_score_dict.values()]
        overlay_score_list = sorted(overlay_score_list, reverse=True, key=lambda x: x[1])
        action_score_list = sorted(action_score_list, reverse=True, key=lambda x: x[1])

        print("Top 3 actions:")
        for i in action_score_list[:3]:
            cmd, score = i
            print("\t{}: {}".format(cmd, score))

        print("Top 3 overlays:")
        for i in overlay_score_list[:3]:
            cmd, score = i
            print("\t{}: {}".format(cmd, score))

        # print([k[1][2] for k in overlay_score_dict.items()])
        best_ov_key = max(overlay_score_dict.items(), key=lambda k: k[1][2])[0]
        best_act_key = max(action_score_dict.items(), key=lambda k: k[1][2])[0]
        best_est_ov_command, ov_var, ov_score = overlay_score_dict[best_ov_key]
        best_est_act_command, act_var, act_score = action_score_dict[best_act_key]
        # Remove potential whitespace from item name.
        act_var["item"] = act_var["item"].replace(" ", "")
        # TODO: Need to do the above for overlay ov_var as well.
        print("best ov key: {} and cmd: {} score: {}".format(best_ov_key, best_est_ov_command, ov_score))
        print("best action key: {} and cmd: {} score: {}".format(best_act_key, best_est_act_command,
              act_score))
        # print(action_score_dict)
        # print(req)
        # print("ov var: ", ov_var)
        # print("act var: ", act_var)
        ov_rules = self._command_to_rule(best_ov_key, ov_var, best_est_ov_command)
        # Get overlay type (eg.g PERMIT, PROHIBIT, TRANSFER, REMOVE) associated with the best
        # scoring overlay
        ov_type = self.overlay_template_dict[best_ov_key][1]

        ov_res_dict = {"key": best_ov_key, "rules": ov_rules, "score": ov_score,
                       "type":"overlay", "overlay_type":ov_type, "params": ov_var,}
        act_res_dict = {"key": best_act_key, "action_param_dict":act_var, "score":act_score,
                        "type":"action"}
        # return the result dictionary with the highest score.
        return max([ov_res_dict, act_res_dict], key=lambda res: res["score"])




    def process_command(self, command):
        # Remove stop words
        pattern = re.compile(r'\b(' + r'|'.join(self.STOP_WORDS )+ r')\b\s*')
        processed_command = pattern.sub('', command)
        # Get overlay or corrective action from command
        res_dict = self._rule_from_command(processed_command)
        res_dict["transcript"] = command

        return res_dict

    @staticmethod
    def ok_baxter(command):
        regex =  \
            r"^(\w+\b\s){0,2}(baxter|boxer|braxter|back store| back sir|baxar|dexter|pastor|faster)"

        # Remove 'ok baxter' from command, if present.
        subbed_command = re.sub(regex, '', command.lower())
        has_ok_baxter = not command == subbed_command

        return subbed_command, has_ok_baxter



if __name__ == '__main__':
    # stt = TTS()
    # stt.say("Yale chicken tenders are the best!")
    # device_id = 11
    stt = STT(0)
    ingredients = ['oats', 'milk', 'measuring cup', 'strawberry', 'blueberry', 'banana',
                   'chocolate chips', 'salt', 'mixing spoon', 'eating spoon', 'coco puffs',
                   'pie', 'muffin', 'eggs', 'croissant', 'jelly pastry', 'bowl']
    meal = ['oatmeal', 'cereal']
    side = ['pastry']
    nutritional = ['dairy', 'vegan', 'healthy']
    shelf = ['top', 'bottom']
    dest = ['pan', 'bowl', 'microwave']
    # stt.get_transcript()
    pc = ProcessCommand(ingredients, meal, side, shelf, nutritional, dest)
    while True:
        in_cmd = input("input command: ").strip()
        res = pc.process_command(in_cmd)
        print(res)
    # transcript = get_transcript(device_id)
    # print(transcript, "after exit")
