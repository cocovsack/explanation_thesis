#!/usr/bin/env python

import os
import json
import numpy as np
import rospy
import rospkg
import torch
from chefbot_utils.learning import DQNAgent, load_action_space

from chefbot_utils.overlay import PrologOverlay, RosOverlayList
from chefbot_utils.learning_util import (ROLL, MUFFIN, JELLYPASTRY, PIE, EGG,
                                         CEREAL, MILK, WATER, PLAINOATMEAL, FRUITYOATMEAL,
                                         CHOCOLATEOATMEAL, PBBANANAOATMEAL,
                                         FRUITYCHOCOLATEOATMEAL, PBCHOCOLATEOATMEAL
                                         )

from chefbot_utils.htn import CookingHTNFactory
from chefbot_utils.pddl_util import load_environment, update_goal


from chefbot.srv import AbstractActionCommand, AbstractActionCommandRequest, AbstractActionCommandResponse, \
                        Overlay, OverlayRequest, OverlayResponse, \
                        Reward, RewardRequest, RewardResponse

from chefbot_utils.pddl_util import FeatureDomainGenerator
PKG_PATH = rospkg.RosPack().get_path("chefbot")


pddl_dir = os.path.join(PKG_PATH, "pddl")
_env = load_environment(pddl_dir, False)

FeatureDomainGenerator(_env, pddl_dir).save_domain_to_file()

class Reasoner(DQNAgent):
    def __init__(self, **kwargs):
        "docstring"
        pddl_dir = os.path.join(PKG_PATH, "pddl")
        self._env, f_env = load_environment(pddl_dir, True)

        # Generate pddl file used for representing state as a feature vec.
        FeatureDomainGenerator(self._env, pddl_dir).save_domain_to_file()
        lr = rospy.get_param("chefbot/learning_params/lr", default=.0025)
        gamma = rospy.get_param("chefbot/learning_params/gamma", default=.90)
        fc2_size = rospy.get_param("chefbot/learning_params/fc2_size", default=32)
        batch_size = rospy.get_param("chefbot/learning_params/batch_size", default=64)

        # action_space = load_action_space(os.path.join(PKG_PATH,
        #                                               "models/robot_trained_models_seed_888888/ACTION_SPACE.npy"))
        action_space = load_action_space(os.path.join(PKG_PATH, "models", kwargs["model_path"],
                                                      "ACTION_SPACE.npy"))


        print("reasoner action space: ", action_space[:10])
        super(Reasoner, self).__init__(f_env, action_space=action_space,
                                       batch_size=batch_size,
                                       gamma=gamma, lr=lr, fc2_size=fc2_size,
                                       **kwargs)
        self._corrective_action_list = []
        self._run_name = rospy.get_param("chefbot/learning/run_name", default="jake_run_1")
        self._results_save_path = rospy.get_param("chefbot/learning/save_path", default="robot_trial_results")
        self._results_save_path = os.path.join(PKG_PATH, "models", self._results_save_path)

        self._negative_reward = rospy.get_param("chefbot/learning/negative_rewards", default=-.5)
        # These enable you to test the reasoner on ground truth sequences rather than
        # from agent predictions
        self._testing = rospy.get_param("chefbot/learning/testing", default=False)
        # The main course of the task. options are CEREAL, PLAINOATMEAL,  FRUITYOATMEAL,
        # CHOCOLATEOATMEAL, PBBANANAOATMEAL, FRUITYCHOCOLATEOATMEAL, PBCHOCOLATEOATMEAL
        self.main_course = rospy.get_param("chefbot/learning/main_course", default=FRUITYOATMEAL)
        # The side course of the task. options are ROLL, MUFFIN, JELLYPASTRY, PIE, EGG,
        self.side_course = rospy.get_param("chefbot/learning/side_course", default=MUFFIN)
        # These control how the robot intervenes for each meal. Choices are "do" or "say"
        main_action_type = rospy.get_param("chefbot/learning/main_action_type",default="do")
        side_action_type = rospy.get_param("chefbot/learning/side_action_type",default="do")
        # The liquid base for the oatmeal. Choices are MILK and WATER. NOTE: does not apply when main_course == CEREAL
        oatmeal_base = rospy.get_param("chefbot/learning/oatmeal_base", default=MILK)
        # Controls preference for how robot manipulates ingredients. Options are "use immediately" or "ingredients first"
        order_preference = rospy.get_param("chefbot/learning/order_preference",
                                                 default="use immediately")

        self._htn_factory = CookingHTNFactory(main=self.main_course, side=self.side_course,
                                              liquid=oatmeal_base,
                                              action_types={"main":main_action_type,
                                                            "side":side_action_type},
                                              order= "main first")


        if self._testing:
            if order_preference == "use immediately":
                self._gt_htn = self._htn_factory.use_immediately()
            elif order_preference == "ingredients first":
                self._gt_htn = self._htn_factory.ingredients()

        self._state, _ = self._env.reset()

        # rospy.loginfo("[reasoner] waiting for chefbot/reasoner/corrective_action")
        # rospy.wait_for_service("chefbot/reasoner/corrective_action")
        rospy.Service("chefbot/reasoner/corrective_action", AbstractActionCommand, self._corrective_action_cb)
        rospy.loginfo("[reasoner] waiting for chefbot/reasoner/next_action")
        rospy.wait_for_service("chefbot/reasoner/next_action")
        self._next_action_srv = rospy.ServiceProxy("chefbot/reasoner/next_action", AbstractActionCommand)

        # reward service callback
        self._reward_srv = rospy.Service('chefbot/reasoner/reward', Reward, self._reward_cb)

        # overlay service callback
        self.overlay_srv = rospy.Service('chefbot/reasoner/overlay', Overlay, self._overlay_cb)


        self._check_for_rewards = False
        self.action = None
        self.reward = None
        self.overlay = None
        self.overlay_count = 0
        self.result_dict  = {"predicted_actions":[],
                             "overlays": {}, "corrections": {},
                             'sequence': [], 'reward':[]}


        self.goal_main = "oatmeal"
        self.goal_side = "pastry"

    # TODO This is just a place holder for now, we are going to need
    # to incorporate state info from perception module

    def _reward_cb(self, srv):
        self.reward = srv.reward
        print("Received reward = ", self.reward)
        return RewardResponse(True)

    def _overlay_cb(self, srv):

        # self.overlay = srv.overlay
        print("recieved: {}".format(srv))
        name = srv.name
        expr = srv.overlay[0]
        transcript = srv.transcript
        overlay_type = srv.type

        if name in "Ov_0":
            if "oatmeal" in transcript:
                self.goal_main = "oatmeal"
            elif "cereal" in transcript:
                self.goal_main = "cereal"

        overlay_dict = {"rule": expr, "type": overlay_type,
                        "transcript": transcript}

        name += "_{}".format(self.overlay_count)
        self.result_dict["overlays"][name] = overlay_dict
        self.result_dict["sequence"].append(name)
        self.overlay_count += 1

        if overlay_type == "forget":
            rospy.loginfo("Removing overlay...")
            overlay_key = srv.params[0]
            self.remove_overlay(overlay_key)
        else:
            print("Adding overlay...")
            rospy.sleep(1.)
            overlay = PrologOverlay(name, expr, overlay_type)
            self.add_overlay(overlay)
        return OverlayResponse(True)
        # print("Received overlay = ", self.overlay)

        # return OverlayResponse(True)

    def _get_state(self):
        return self._state

    def _corrective_action_cb(self, req):
        # import pdb; pdb.set_trace()
        rospy.loginfo("[corrective_action_cb] Recieved cmd: {}".format(req))
        if req.action.data == "complete":
            self._check_for_rewards = True
            return AbstractActionCommandResponse(True)

        if req.is_feedback.data:
            corrective_action = self._abstract_action_to_action_literal(req)
            self._corrective_action_list.append(corrective_action)
            rospy.loginfo("[corrective_action_cb] Added corrective action to list")
            # state, reward, done, info = self._env.step(corrective_action)
            # rospy.loginfo("Updating state with corrective action")
            # self._state = state
            # TODO NEED TO SAVE/REMEBER this 
            # self._corrective_action_list.append(corrective_action)
        return AbstractActionCommandResponse(True)


    def _abstract_action_to_action_literal(self, req):
        name = req.action.data
        at = req.action_type.data
        item = req.item.data
        dest = req.dest.data
        # The action literals for putin actions are putinmicrowave and putinsink
        # not 'putin' alone
        if name == "putin":
            name += dest
            dest = ""
        var_list = [i for i in [at,item, dest] if not i == ""]
        cmd_action_var_set = set(var_list)

        found_action = False
        for action in self.action_space:
            if name in action.predicate.name:
                action_var_set = set([v.name for v in \
                                      action.variables])
                # import pdb; pdb.set_trace()
                if cmd_action_var_set.issubset(action_var_set):
                    rospy.loginfo("Found corrective action: {}".format(action))
                    found_action == True
                    return action

        if not found_action:
            raise RuntimeError("action not found!!!!")



    def _send_action(self, action):
        """
        @input action, Pddlgym.structs.TypedEntity: The action to be sent.
        """
        if action is not None:
            name = action.predicate.name
            req = AbstractActionCommandRequest()
            req.is_feedback.data = False

            if "gather" in name:
                item, action_type = action.variables
                req.action.data = 'gather'
                req.action_type.data = action_type.name
                req.item.data = item.name

            elif "pourwater" in name:
                dest, action_type = action.variables
                req.action.data = 'pour'
                req.item.data = "water"
                req.dest.data = dest
                req.action_type.data = action_type.name
            elif "pour" in name:
                item, dest, action_type = action.variables
                req.action.data = 'pour'
                req.item.data = item.name
                req.dest.data = dest
                req.action_type.data = action_type.name
            elif "putin" in name:
                req.action.data = "putin"
                if "sink" in name:
                    item, _, action_type = action.variables
                    req.dest.data = "sink"
                elif "microwave" in name:
                    item, action_type = action.variables
                    req.item.data = item.name
                    req.dest.data = "microwave"

                req.item.data = item.name
                req.action_type.data = action_type.name
            elif "turnon" in name:
                item, action_type = action.variables
                req.action.data = 'turnon'
                req.action_type.data = action_type.name
                req.item.data = item.name
            elif "collectwater" in name:
                _,_, action_type = action.variables
                req.action.data = 'collectwater'
                req.action_type.data = action_type.name
            elif "takeoutmicrowave" in name:
                item, action_type = action.variables
                req.action.data = 'takeoutmicrowave'
                req.action_type.data = action_type.name
                req.item.data = item.name
            elif "grabspoon" in name:
                action_type = action.variables[-1]
                req.action.data = 'grabspoon'
                req.action_type.data = action_type.name
            elif "mix" in name:
                action_type = action.variables[-1]
                req.action.data = 'mix'
                req.action_type.data = action_type.name
            elif "reduceheat" in name:
                action_type = action.variables[-1]
                req.action.data = 'reduceheat'
                req.action_type.data = action_type.name
            elif "serveoatmeal" in name:
                action_type = action.variables[-1]
                req.action.data = 'serveoatmeal'
                req.action_type.data = action_type.name

            #TODO Need to figure out to do with actions like checkoatmeal, checkcereal,
            # boilliquid, etc. as these sorts of actions are mostly useful in pddl context
            req.is_feedback.data = False
            print("send_action: ", action)
            resp = self._next_action_srv(req)
            return resp.exec_success


    def save_result(self):
        if not os.path.isdir(self._results_save_path):
            os.makedirs(self._results_save_path)

        save_file = os.path.join(self._results_save_path, self._run_name + ".json")
        print("[save_results] saving...")
        with open(save_file, 'w') as outfile:
            json.dump(self.result_dict, outfile)



    def run(self):
        # TODO: Need to handle loading/saving data 
        # TODO: Need to incorporate overlays

        reward = 0
        self._state = self._get_state()
        self._state = update_goal(self._env, self._state, main=self.goal_main, side=self.goal_side)
        rospy.loginfo("Curr state: {}".format( self._state))
        self._env.set_state(self._state)
        rospy.loginfo("[corrective actions] Corrective actions: {}".format(self._corrective_action_list))
        # Get the possible actions in the current state
        possible_actions = list(self._env.action_space.all_ground_literals(self._state))

        # print("[reasoner] feasable actions: ", possible_actions)
        if self._corrective_action_list:
            rospy.loginfo("Applying correctivea action")
            action = self._corrective_action_list.pop(0)
            reward += self._negative_reward

            correction_label = "c{}".format(len(self.result_dict['corrections']))
            self.result_dict["corrections"][correction_label] = str(action)
            self.result_dict["sequence"].append(correction_label)
        elif self._testing:
            action = self._gt_htn.get_next_action(self.action_space)
            print(action)
            prompt = input("Press:\n\t \'s\' to skip self.action.\n\t\'Enter\' to perform next self.action\n")
            # Then get the next self.action from the ground truth sequence defined by HTN
            if  prompt == "s":
                action = None
        else:
            # Otherwise get action from the DQN agent
            action, pred_action = self.act(self._state, possible_actions, ret_original_pred=True)
            self.result_dict["sequence"].append(str(action))
            self.result_dict["predicted_actions"].append(str(pred_action))

        print(action)
        prompt = input("Press:\n\t \'s\' to skip self.action.\n\t\'Enter\' to perform next self.action\n")
        # Then get the next self.action from the ground truth sequence defined by HTN
        if  prompt == "s":
            # action = None
            send_action = True
        else:
            send_action = self._send_action(action)
        # if self._send_action(action):
        # if self._send_action(action):
        if send_action:
            # Update world state with self.action
            next_state, _, done, info = self._env.step(action)
            next_state = update_goal(env=self._env, state=next_state,
                                     main=self.goal_main, side=self.goal_side)

            # Check for rewards (byseeing if any check- actions are present in possible_actions)
            if self._check_for_rewards:
                rospy.loginfo("[reasoner] Checking for rewards")
                possible_actions = list(self._env.action_space.all_ground_literals(next_state))
                next_state, new_reward, done = self.check_rewards(self._env, possible_actions,
                                                                # goal=[self.main_course, self.side_course]
                                                                # goal=None,
                                                                # goal=[self.goal_main,
                                                                #       self.goal_side],
                                                                )
                reward += new_reward
                self._check_for_rewards = False


            self.result_dict["reward"].append(reward)

            rospy.loginfo("Current reward: {}".format(reward))
            rospy.loginfo("\nCurrent results:\n{}\n".format(self.result_dict))
            self.save_result()
            # Remember SARSA and perform experience replay
            s_bin = self.get_transformed_state(self._state)
            s_next_bin = self.get_transformed_state(next_state)
            self.remember(s_bin, action,reward, s_next_bin, done)
            self.experience_replay()
            self._state = next_state
        
        input("Press 'Enter' once you have provided all feedback!")



if __name__ == '__main__':
    # model_dir = os.path.join(PKG_PATH, "models")
    # pddl_dir = os.path.join(PKG_PATH, "pddl")
    # generate_action_space(pddl_dir, model_dir)
    try:
        rng = np.random.default_rng(0)
        rospy.init_node('reasoner', anonymous=True)
        # model_path = "robot_trained_models_seed_888888/DQN_ep_240"
        model_path = "trained_models/goal_conditioned_True/num_training_50/seed_888888/DQN_ep_9"
        r = Reasoner(model_path=model_path, pretrained=True,
                     exp_rate=0.0,rng=rng, load_memories=False, max_memory_size=10000,
                     goal_condition=True)
        rospy.sleep(2.0)
        while not rospy.is_shutdown():
            with torch.no_grad():
                r.run()
    except rospy.ROSInterruptException:
        pass
