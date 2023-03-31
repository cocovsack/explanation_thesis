#!/usr/bin/env python

import os
# import rospy
import json
import pdb
import re
import sys
import subprocess
import time
import numpy as np
# import rospkg
# import pyaudio


import sys, contextlib
import learning_util

from collections import defaultdict
from fuzzywuzzy import fuzz
from itertools import product, combinations, permutations
from copy import deepcopy
from learning import DQNAgent, load_action_space
from htn import CookingHTNFactory
from pddl_util import load_environment, dist_to_goal, PDDL_DIR
from overlay import Overlay, OverlayList, PrologOverlay, PrologOverlayList, OverlayFactory
from sim_eval import InteractionGenerator, undo_incorrect_pastry_action, same_action_diff_intervention_type
from state_transformer import add_shelf_classifaction

SEED = 1000
N_MEALS = 1
SPLIT = 0
FAILURE_REWARD = -1.0
DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../models")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../config")

WRONG_PRED_REWARD = -.5
WRONG_ACTION_TYPE_REWARD = WRONG_PRED_REWARD * .5
DEFAULT_MODEL_ARGS = {
    "batch_size":64,
    "gamma":.99,
    "lr":0.002,
    "max_memory_size":50000,
    "exp_min":0.0,
    "exp_decay":0.99,
    "exp_rate":0.0,
    "pretrained":False,
    "load_memories": False,
    "fc2_size":32,
}

INGREDIENTS = ['oats', 'milk', 'measuring cup', 'strawberry', 'blueberry', 'banana',
                   'chocolatechips', 'salt', 'mixingspoon', 'eatingspoon', 'cocopuffs',
                   'pie', 'muffin', 'peanutbutter', 'egg', 'roll', 'jellypastry', 'bowl']
MEAL = ['oatmeal', 'cereal']
SIDE = ['pastry']
DISHES = ['oatmeal', 'cereal', 'pastry']
NUTRITIONAL = ["fruity", "quick", "sweet", "gluten", "sodium", "protein", "nonvegan", "dairy", "healthy", "bread"]
ACTIONS = {
    "gat": "gathering the {}",
    "pou": "pouring the {}",
    "pourw": "pouring the water in the {}",
    "put": "putting the {} in the microwave",
    "tur": "turning on the {}",
    "coo": "cooking the oatmeal on the {}",
    "col": "collecting the water in a {}",
    "tak": "taking the {} out of the microwave",
    # "gra": "grabbing",
    "mix": "mixing in a {}",
    "mic": "microwaving the {}",
    "re": "reducing the heat of the {}",
    "se": "serving the oatmeal in a {}",
    # "co": "completing the meal",
    "boi": "boiling the {} in liquid"
}

SHELF = ['top', 'bottom']
DEST = ['pan', 'bowl', 'microwave']

class Query(object):
    STOP_WORDS = ["the", "with", "high", "rest"]

    def __init__(self):
        self.query_template_dict = {"Qv_0": "why did you use {ingredient}?", 
                                    "Qv_1": "can i use {ingredient} instead?", 
                                    "Qv_2": "why did you {action} the {ingredient}?",
                                    "Qv_3": "why did you make the {meal} {order}?",
                                    "Qv_4": "why didn't you use {ingredient}?",
                                    }
    
    def process_query(self, query):
        # Remove stop words
        pattern = re.compile(r'\b(' + r'|'.join(self.STOP_WORDS )+ r')\b\s*')
        processed_query = pattern.sub('', query)
        res_dict = self.query_generator(processed_query)
        res_dict["transcript"] = query
        return res_dict
    
    def query_generator(self, processed_query):
        query_score_dict = {}

        for name, temp_vars in self.query_template_dict.items():
            est_command, var ,score = self._query_score_template(processed_query, name, temp_vars,
                                                            template_type="query")
            query_score_dict[name] = (est_command,  var, score)

        query_score_list = [(cmd, score) for cmd, _, score in query_score_dict.values()]
        query_score_list = sorted(query_score_list, reverse=True, key=lambda x: x[1])

        best_key = max(query_score_dict.items(), key=lambda k: k[1][2])[0]
        best_est_ov_command, var, score = query_score_dict[best_key]
            
        query_res_dict = {"key": best_key, "score": score,
                        "type": "query", "params": var,}
        # return the result dictionary with the highest score.
        return max([query_res_dict], key=lambda res: res["score"])



    def _query_score_template(self, command, name, template, template_type="query"):
        results = []
        if template_type == "query":
            cmd_gen = self._gen_query_commands_from_template

        for t, var in cmd_gen(name, template):
            score = fuzz.ratio(command, t)
            results.append((t, var,  score))

        return max(results, key=lambda x: x[2])

    def _gen_query_commands_from_template(self, name, template):
        # PERMIT queries
        if name in ["Qv_0", "Qv_1", "Qv_4"]:
            # why did you use _____? or # can i use ___ instead? or why didn't you use
            for i in INGREDIENTS:
                yield template.format(ingredient=i).lower(), [i]
        elif name in ["Qv_2"]:
            # why did you use _____?
            for action_type in learning_util.ACTION_DICT.keys():
                for i in INGREDIENTS:
                    yield template.format(action=action_type, ingredient=i).lower(), [action_type, i]
        elif name in ["Qv_3"]:
            # why did you make ___ first?
            for m in (learning_util.MAIN_LIST + learning_util.PASTRY_LIST + learning_util.OATMEAL_LIST):
                for o in ["first", "second"]:
                    yield template.format(meal=m, order = o).lower(), [m, o]
                            
class ProcessCommand(object):
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
                                      "Ov_10": ("it isn't safe to use {ingredient} because I am allergic", self.PROHIBIT),
                                      "Ov_11": ("i want to use {ingredient}", self.PERMIT),
                                      "Ov_12": ("user isn't allowed to eat {ingredient}", self.PROHIBIT)
                                      }

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

    def process_command(self, command):
            # Remove stop words
            pattern = re.compile(r'\b(' + r'|'.join(self.STOP_WORDS )+ r')\b\s*')
            processed_command = pattern.sub('', command)   # TKTK WhAT DOES COMPILE/SUB DO?
            res_dict = self._rule_from_command(processed_command)
            res_dict["transcript"] = command

            return res_dict

    def _get_action_space(self, constraints, template_key, type):
        action_space = {}
        potential_ingredients = set()
        if template_key == 'Ov_0' or template_key =='Ov_5':
            main, side = constraints
            if main == "oatmeal" or side == 'oatmeal':
                potential_ingredients.update(learning_util.ALL_OATMEAL_INGREDIENTS)
            elif main == "cereal" or side == 'oatmeal':
                potential_ingredients.update(learning_util.CEREAL_INGREDIENTS)
            elif main == 'pastry' or side == "pastry":
                potential_ingredients.update(learning_util.PASTRY_LIST)
            for v in potential_ingredients:
                action_space[v] = learning_util.INGREDIENT_DICT[v]
        elif template_key == 'Ov_1':
            nutr = constraints
            for v in learning_util.INGREDIENT_DICT.keys():
                if nutr in learning_util.INGREDIENT_DICT[v] or "{}_precursor".format(nutr) in learning_util.INGREDIENT_DICT[v]:
                    action_space[v] = learning_util.INGREDIENT_DICT[v]
        # elif template_key == 'Ov_2':
        #     shelf = constraints
        #     add_shelf_classifaction(learning_util.INGREDIENT_DICT)
        #     for v in learning_util.INGREDIENT_DICT.keys():
        #         if shelf in learning_util.INGREDIENT_DICT[v] or "{}_precursor".format(nutr) in learning_util.INGREDIENT_DICT[v]:
        #             # action_space[v] = learning_util.INGREDIENT_DICT[v] 
        #             if v in list(action_space):
        #                 del action_space[v]
        elif template_key == 'Ov_3' or template_key == 'Ov_4':
            dish = constraints
            if dish == "oatmeal":
                potential_ingredients.update(learning_util.ALL_OATMEAL_INGREDIENTS)
            elif dish == "cereal":
                potential_ingredients.update(learning_util.CEREAL_INGREDIENTS)
            elif dish == "pastry":
                potential_ingredients.update(learning_util.PASTRY_LIST)
            for v in potential_ingredients:
                action_space[v] = learning_util.INGREDIENT_DICT[v]
        elif template_key == 'Ov_9':
            no_nutr = constraints
            for v in learning_util.INGREDIENT_DICT.keys():
                if no_nutr not in learning_util.INGREDIENT_DICT[v] and "{}_precursor".format(no_nutr) not in learning_util.INGREDIENT_DICT[v]:
                    action_space[v] = learning_util.INGREDIENT_DICT[v]
        
        print(action_space)
        return action_space


    def _command_to_rule(self, template_key, var, command):

        # ROBOT_DO_TEMP = "(do(A_out) or say_only(A_out)) and making_{dish}(A_out)"
        ROBOT_DO_TEMP = "do(A_out) and making_{dish}(A_out)"
        # ROBOT_SAY_TEMP = "(say(A_out) or do_only(A_out)) and making_{dish}(A_out)"
        ROBOT_SAY_TEMP = "say(A_out)  and making_{dish}(A_out)"
        for num, word in self.num2word.items():
            if word in command:
                k = num

        #  PERMIT
        if template_key == 'Ov_0':
            m1, m2 = var
            first_pred = "making_{dish1}(A_out) and state(no_completed_dish)".format(dish1=m1)
            last_pred = "making_{dish2}(A_out) and state(one_completed_dish)".format(dish2=m2)
            rule  = "not state(two_completed_dish) then ({first} or {last})".format(first=first_pred,
                                                                               last=last_pred)
            # rules = ["is_making_{meal}(A_out) or is_making_{side}(A_out) -> action(A_out)".format(
            #     meal=meal, side=side)]
            rules = [rule]
            type = "permit"
            constraints = [m1, m2]
            action_space = self._get_action_space(constraints, template_key, type)
        elif template_key == 'Ov_1':
            nutr = var[0]
            constraints = var[0]
            type = "permit"
            # rules = ["{nutr}(A_out) -> action(A_out)".format(nutr=nutr)]
            rules = ["{nutr}(A_out) or {nutr}_precursor(A_out)".format(nutr=nutr)]
            action_space = self._get_action_space(constraints, template_key, type)
        elif template_key == 'Ov_2':
            _, _, shelf, _ = var
            shelf = shelf.split() + ["shelf"]
            shelf = "_".join(shelf)
            constraints = shelf
            rules = ["not {}(A_out)".format(shelf)]
            type = "prohibit"
            action_space = self._get_action_space(constraints, template_key, type)
        elif template_key == "Ov_3":
            dish = var[0]
            constraints = dish
            robot_do_pred = ROBOT_DO_TEMP.format(dish=dish)
            rule = "not state(two_completed_dish) then equiv_action(A_in, A_out) and {pred}".format(pred=robot_do_pred)
            rules = [rule]
            type = "robot_transfer"
            action_space = self._get_action_space(constraints, template_key, type)
        elif template_key == "Ov_4":
            dish = var[0]
            constraints = dish
            robot_say_pred = ROBOT_SAY_TEMP.format(dish=dish)
            rule = "not state(two_completed_dish) then equiv_action(A_in, A_out) and {pred}".format(pred=robot_say_pred)
            rules = [rule]
            type = "person_transfer"
            action_space = self._get_action_space(constraints, template_key, type)
        elif template_key == 'Ov_5':
            m1, m2 = var
            constraints = [m1, m2]
            type = ["robot_person_transfer", m1, m2]
            robot_do_preds = ROBOT_DO_TEMP.format(dish=m1)
            robot_say_preds = ROBOT_SAY_TEMP.format(dish=m2)
            rule1 = "not state(two_completed_dish) then (equiv_action(A_in, A_out) and (({}) or ({}) ))".format(robot_do_preds, robot_say_preds)
            rules = [rule1]
            action_space = self._get_action_space(constraints, template_key, type)
            ## tktk both sides????
        elif template_key == 'Ov_8':
            rules = ["forget the last rule"]
            type = "forget"
            action_space = None
            constraints = None
            # TKTK??? action space here?
        elif template_key == 'Ov_9':
            nutr = var[0]
            constraints = nutr
            rules = ["not state(two_completed_dish) then not {nutr}(A_out)".format(nutr=nutr)]
            type = 'prohibit'
            action_space = self._get_action_space(constraints, template_key, type)
        elif template_key == 'Ov_10':
            ingredient = var
            constraints = ingredient
            rules = ["not state(two_completed_dish) then not has_{ingred}(A_out)".format(ingred=ingredient)]
            type = 'prohibit'
            #TK TK check the action space and rules
            action_space = self._get_action_space(constraints, template_key, type)
        elif template_key == 'Ov_11':
            ingredient = var
            constraints = ingredient
            rules = ["has_{ingred}(A_out)".format(ingred=ingredient)]
            type = 'permit'
            #TK TK check the action space and rules
            action_space = self._get_action_space(constraints, template_key, type)
        elif template_key == 'Ov_12':
            ingredient = var
            constraints = [ingredient, "authority"]
            rules = ["not state(two_completed_dish) then not has_{ingred}(A_out)".format(ingred=ingredient)]
            type = 'prohibit'
            #TK TK check the action space and rules
            action_space = self._get_action_space(constraints, template_key, type)
        else:
            rules = ["foo"]
        
        return rules, action_space, constraints

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
            
        ov_rules, action_space, params = self._command_to_rule(best_ov_key, ov_var, best_est_ov_command)
        # Get overlay type (eg.g PERMIT, PROHIBIT, TRANSFER, REMOVE) associated with the best
        # scoring overlay
        ov_type = self.overlay_template_dict[best_ov_key][1]

        ov_res_dict = {"key": best_ov_key, "rules": ov_rules, "score": ov_score, "params": params,
                        "type": "overlay", "overlay_type": ov_type, "params": ov_var, "overlay_action_space": action_space}
        act_res_dict = {"key": best_act_key, "action_param_dict":act_var, "score":act_score,
                            "type":"action"}
        # return the result dictionary with the highest score.
        return max([ov_res_dict, act_res_dict], key=lambda res: res["score"])


    def _score_template(self, command, name, template, template_type="overlay"):
        results = []
        if template_type == "overlay":
            cmd_gen = self._gen_overlay_commands_from_template
        elif template_type == "action":
            cmd_gen = self._gen_corrective_commands_from_template

        for t, var in cmd_gen(name, template):
            score = fuzz.ratio(command, t)
            results.append((t, var,  score))
        return max(results, key=lambda x: x[2])

    def _gen_overlay_commands_from_template(self, name, template):
        max_num = max(list(self.num2word.keys()))
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
        elif name in ["Ov_10", "Ov_11", "Ov_12"]:
            for i in self.ingredients:
                yield template.format(ingredient=i).lower(), i
        else: 
            for p in product(self.meal, self.side, self.shelf, self.nutritional):
                m, s, sh, n = p
                yield template.format(meal=m, side=s, shelf=sh, nutritional=n).lower(), p

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


def model_run(overlay_input, rng, model_args=None, agent_name="overlay",
                       step=5, start= 5, end=30, max_n_steps=50, save=True,
                       save_dir="explainable_trial", model_path="trained"):
    # get the test meal
    # TKTK, use ingredients first... the way gt_htn is constructed? should i allow for more user input
    training_htns, testing_htns = learning_util.generate_train_test_meals(rng, N_MEALS, SPLIT)
    gt_htn = CookingHTNFactory(testing_htns[0]["main"], testing_htns[0]["side"], testing_htns[0]["liquid"], testing_htns[0]["action_types"], testing_htns[0]["order"])._generate_meal("use immediately")

    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../../model_files/")
    action_space = load_action_space(os.path.join(model_dir, "ACTION_SPACE.npy"))



    #set up default model
    _, dummy_feature_env =  load_environment(None, True,)
    load_model_path = os.path.join(model_dir, "DQN.pt")

    default_model_args = deepcopy(DEFAULT_MODEL_ARGS)
    default_model_args["feature_env"] = dummy_feature_env
    default_model_args["rng"] = rng
    default_model_args["action_space"] = action_space
    default_model_args["model_path"] = load_model_path

    if not model_args is None:
        default_model_args.update(model_args)

    # agent/environment initalization
    agent = DQNAgent(agent_name="overlay", **default_model_args)
    

    env, feature_env =  load_environment(None, True, goal_main=gt_htn.main, goal_side=gt_htn.side)
    feature_env.goal_condition = model_args["goal_condition"]
    agent.feature_env = feature_env


    overlay_permutations = []
    
    for i in range(1, len(overlay_input)+1):
        ele = list(combinations(overlay_input, i))
        for el in ele:
            overlay_permutations.append(list(el))

    for user_overlay in [overlay_permutations[-1]]:

        gt_htn.reset()
        agent.reset()
        agent.overlays = None

        trace = []
        performed_actions = []
        corrective_action = None
        corrective_actions = []
        correct_preds = []
        predicted_actions  = []
        done, failed = False, False
        reward, step, n_corrections, goal_dist = 0, 0, 0, 0
        loss_ep = 0

        for i in range(0, len(user_overlay)):
            overlay = PrologOverlay("testing_meal", user_overlay[i]["rules"][0], user_overlay[i]["overlay_type"])

            # agent.overlays = PrologOverlayList([overlay])
            if agent.overlays is None:
                agent.overlays = PrologOverlayList([overlay])
            else:
                agent.overlays.append(overlay)
    
        # initialize trace
        int_reward_dict = {"reward": 0 , "corrections": 0, "completed":False, "steps":0,
                            "overlays": {}, "shields": {}}

        print("Overlays: ", agent.overlays)

        state, _ = env.reset()
        possible_actions = list(env.action_space.all_ground_literals(state))

        testing = []
        meal = testing_htns[0]["main"], testing_htns[0]["side"]
        while not done and step < max_n_steps:

            print("\n\n#########STEP {}#########".format(step))
            # Get action predictions
            print("Remaining corrective actions action: ", corrective_actions)
            pred_action, pred_action_no_ov, relevant_values = agent.act(state, possible_actions, ret_original_pred=True)
            print("[compare_agents] pred action: ", pred_action, pred_action_no_ov)

            # Get queued corrective actions if they exist
            if len(corrective_actions) > 0:
                corrective_action = corrective_actions.pop(0)
            else:
                plan_action, already_performed = gt_htn.get_next_action(action_space,
                                                                            performed_actions)
                testing.append(plan_action)
                print("[compare_agents] gt_htn action: ", plan_action)
                print("[compare_agents] already_performed: ", already_performed)
                predicted_actions.append(str(plan_action))

            ### Decide what the actual action will be
            if plan_action is None and corrective_action is None:
                break
            elif step == -1:
                print("[compare_agents] Using gt_htn action: ", plan_action)
                action = plan_action
            elif corrective_action:
                action = corrective_action
                # corrective_action = None
                print("[compare_agents] Appying corrective action: ",action)
            else:
                print("[compare_agents] Predicting action: ", pred_action)
                action = pred_action

            trace += already_performed

            # if not str(action) in trace:
            if len(corrective_actions) == 0:
                print("[Compare] appending to trace: ", action)
                trace.append(str(action))

            performed_actions.append(str(action))
            already_performed = [] # TKTK why reset?

            next_state, new_reward, done, info = env.step(action)
            print("{} ACTION: {} goal dist: {}\n".format(agent.agent_name, action,
                                                            goal_dist))

            failed = agent.check_errors(next_state)
            if not failed:
                possible_actions = list(env.action_space.all_ground_literals(next_state))
                next_state, new_reward, done = agent.check_rewards(env, possible_actions,
                                                                [gt_htn.main, gt_htn.side])
            else:
                print("[compare_agents] Failed!")
                done = True
                new_reward = FAILURE_REWARD

            reward += new_reward

            print("[compare_agents] corrective action:  ", corrective_action)
            
            # Check if needs corrections
            wrong_action = not gt_htn.check_action_in_seq(gt_htn.root, action) and  corrective_action is None
            print("[compare_agents] wrong action:  ", wrong_action)
            if wrong_action:
                try:
                    last_action = trace.pop(-1)
                    need_to_undo, undo_action, n_backtracks = undo_incorrect_pastry_action(last_action,
                                                                                                  action_space)
                    if same_action_diff_intervention_type(pred_action, plan_action):
                        n_corrections += .5
                        reward += WRONG_ACTION_TYPE_REWARD
                        corrective_actions.append(plan_action)
                    elif need_to_undo:
                        print("[compare_agents]  undoing ", last_action)
                        # trace.pop(-1)
                        prev_actions = gt_htn.get_k_previous_actions(agent.action_space,
                                                                           n_backtracks)

                        if n_backtracks == 0:
                            corrective_actions = [undo_action, plan_action]
                        elif undo_action.predicate.name in [a.predicate.name for a in prev_actions]:
                            corrective_actions  = prev_actions
                        else:
                            corrective_actions  = [undo_action] + prev_actions
                            print("[compare_agents] Adding corrective actions: ", corrective_actions)
                            n_corrections += len(corrective_actions)
                            reward += WRONG_PRED_REWARD * len(corrective_actions)
                    else:
                        reward += WRONG_PRED_REWARD
                        corrective_actions.append(plan_action)
                        n_corrections += 1
                except IndexError as e:
                        print(e)
                        corrective_action  = None
            else:
                # n_errors.append(False)
                correct_preds.append(True)
                corrective_action = None

                    # agent.update_action_hist(action)
                    # print(next_state)
                    # assert not next_state == state

            state = next_state
            if done:
                    break
            step +=1
            # reward = 0 ##TTKTKTK do we want this? 

                # if learn:
                #     agent.lr_scheduler.step()
                # writer.add_scalar("Mean Corrections ", n_corrections / step, num)
                # writer.add_scalar("Total Corrections ", n_corrections, num)

            int_reward_dict["reward"] = reward
            int_reward_dict["corrections"] = n_corrections
            int_reward_dict["steps"] = step
                # int_reward_dict["dist_to_goal"].append(dist_to_goal)
            int_reward_dict["completed"] = done       
            
            int_reward_dict["predicted_actions"] = predicted_actions
            int_reward_dict["actual_actions"] = performed_actions


            print("Ep rewards: ", int_reward_dict)  
            if True:
                print("action: ", action)
                step_cmd = input("query the robot: ").strip()
                if step_cmd !="":
                    query = Query()
                    res = query.process_query(step_cmd)
                    
                    #do something about explanation
                    print("Generating explanations:")
                    generate_explanations(user_overlay, relevant_values, int_reward_dict["actual_actions"])
                    input("press enter to continue")



        if save:
            save_model_path = "src/chefbot_utils/explainability_trials"
        
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)

            # track overlays
            overlay_list = []
            overlay_combo = []

            for ov in agent.overlays:
                overlay_list.append(ov.info()["testing_meal"]["rule"])
            int_reward_dict["overlays"] = overlay_list

            for o in user_overlay:
                overlay_combo.append(str(o["id"]))

            int_reward_dict["id"] = overlay_combo

            # save
            save_file = os.path.join(save_model_path,
                                    "{}.json".format('_'.join(overlay_combo)))
            print("Saving {}...".format(save_file))
            with open(save_file, 'w') as outfile:
                json.dump(int_reward_dict, outfile)

    return int_reward_dict["actual_actions"], user_overlay
    

def generate_explanations (overlays, relevant_values, actions):
    explanations = {}
    config_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "../../config/")
    exp_json = json.load(open(os.path.join(config_dir, "explanation_template.json")))

    '''
    Construct "no" explanation
    '''
    template_none = exp_json['none']
    
    flag = False
    noun = ""
    for char in actions[-1]:
        if char == "(":
            flag = True
            continue
        elif char == ":":
            flag = False
            break
        if flag == True:
            noun += char

    print(noun)
    
    
    if actions[-1][0:5] == "pourw":
        verb = ACTIONS[actions[-1][0:5]]
    else:
        verb = ACTIONS[actions[-1][0:3]]

    
    none_exp = template_none["beginning"] + verb.format(noun) + "."
    print(none_exp)
    

   
    '''
    Construct "statement of fact" explanation
    '''

    '''
    Construct "propose alternative" explanation
    '''
    template_alt = exp_json['alternative']
    alternative_exp = template_alt["beginning"]

    for overlay in overlays:
        print("action space keys:", list(overlay['overlay_action_space']))



    ''' 
    Construct "most important overlay" explanation
    '''

    '''
    Construct "all overlays" explanation
    '''
    template_all = exp_json['all']
    all_exp = template_all["beginning"]

    permissive_adjectives, permissive_dishes, prohibitive_adjectives, prohibitive_dishes, permissive_ingredients, prohibitive_ingredients = get_clauses(overlays)
    print(permissive_adjectives, permissive_dishes, prohibitive_adjectives, prohibitive_dishes, permissive_ingredients, prohibitive_ingredients)
    
    something_permissive = False
    # only permissive adjectives
    if permissive_adjectives != [] and permissive_dishes == [] and permissive_ingredients == []:
        something_permissive = True
        all_exp += "something "
        for i in range(0, len(permissive_adjectives) - 1): 
            all_exp += permissive_adjectives[i] + " and "   
        all_exp += permissive_adjectives[-1]
    # only permissive dishes
    if permissive_adjectives == [] and permissive_dishes != [] and permissive_ingredients == []:
        something_permissive = True
        for i in range(0, len(permissive_dishes) - 1): 
            all_exp += permissive_dishes[i] + " and "   
        all_exp += permissive_dishes[-1]
    # only permissive ingedients
    if permissive_adjectives == [] and permissive_dishes == [] and permissive_ingredients != []:
        something_permissive = True
        all_exp += "something using "
        for i in range(0, len(permissive_ingredients) - 1): 
            all_exp += permissive_ingredients[i] + " and "   
        all_exp += permissive_ingredients[-1]
    #permissive adjectives and permissive ingredients
    elif permissive_adjectives != [] and permissive_dishes != [] and permissive_ingredients == []:
        something_permissive = True
        if len(permissive_adjectives) > 2: 
            all_exp = all_exp + permissive_adjectives[0] + ", " + permissive_adjectives[1] + ", and " + permissive_adjectives[2]
        if len(permissive_adjectives) == 2:
            all_exp = all_exp + permissive_adjectives[0]  + ' and ' + permissive_adjectives[1] + " "
        if len(permissive_adjectives) == 1:
            all_exp = all_exp + permissive_adjectives[0] + " "
        
        for i in range(0, len(permissive_dishes) - 1): 
            all_exp += permissive_dishes[i] + " and "   
        all_exp += permissive_dishes[-1]
    #permissive adjectives and permissive dishes
    elif permissive_adjectives != [] and permissive_dishes == [] and permissive_ingredients != []:
        something_permissive = True
        all_exp += "something "
        for i in range(0, len(permissive_adjectives) - 1): 
            all_exp += permissive_adjectives[i] + " and "   
        all_exp += permissive_adjectives[-1] + " using "
        for i in range(0, len(permissive_ingredients) - 1): 
            all_exp += permissive_ingredients[i] + " and " 
        all_exp += permissive_ingredients[-1]
    #permissive dishes and permissive ingredients
    elif permissive_adjectives == [] and permissive_dishes != [] and permissive_ingredients != []:
        something_permissive = True
        for i in range(0, len(permissive_dishes) - 1): 
            all_exp += permissive_dishes[i] + " and "   
        all_exp += permissive_dishes[-1] + " using "
        for i in range(0, len(permissive_ingredients) - 1): 
            all_exp += permissive_ingredients[i] + " and " 
        all_exp += permissive_ingredients[-1]
    #permissive dishes, adjectives, and ingredients
    elif permissive_adjectives != [] and permissive_dishes != [] and permissive_ingredients != []:
        something_permissive = True
        if len(permissive_adjectives) > 2: 
            all_exp = all_exp + permissive_adjectives[0] + ", " + permissive_adjectives[1] + ", and " + permissive_adjectives[2]
        if len(permissive_adjectives) == 2:
            all_exp = all_exp + permissive_adjectives[0]  + ' and ' + permissive_adjectives[1] + " "
        if len(permissive_adjectives) == 1:
            all_exp = all_exp + permissive_adjectives[0] + " "
        for i in range(0, len(permissive_dishes) - 1): 
            all_exp += permissive_dishes[i] + " and "   
        all_exp += permissive_dishes[-1] + ' using '
        for i in range(0, len(permissive_ingredients) - 1): 
            all_exp += permissive_ingredients[i] + " and " 
        all_exp += permissive_ingredients[-1]

    #only prohibitive adjectives
    if prohibitive_adjectives != [] and prohibitive_dishes == [] and prohibitive_ingredients == []:
        if something_permissive:
            all_exp += " but also to make "
        all_exp += "something not "
        for i in range(0, len(prohibitive_adjectives) - 1): 
            all_exp += prohibitive_adjectives[i] + " or "   
        all_exp += prohibitive_adjectives[-1]
    #only prohibitive dishes - not possible given current overlays
    #only prohibitive ingredients
    if prohibitive_adjectives == [] and prohibitive_dishes == [] and prohibitive_ingredients != []:
        if something_permissive:
            all_exp += " but also to not use "
        else:
            all_exp = all_exp[:-5] + "not use "
        for i in range(0, len(prohibitive_ingredients) - 1): 
            all_exp += prohibitive_ingredients[i] + " or "   
        all_exp += prohibitive_ingredients[-1]
    #prohibitive ingredients and prohibitive dishes - not possible given current overlays
    #prohibitive adjectives and prohibitive dishes - not possible given current overlays
    #prohibitve adjectives and prohibitive ingredients
    if prohibitive_adjectives != [] and prohibitive_dishes == [] and prohibitive_ingredients != []:
        if something_permissive:
            all_exp += " but also to make "
        all_exp += "something not "
        for i in range(0, len(prohibitive_adjectives) - 1): 
            all_exp += prohibitive_adjectives[i] + " or "   
        all_exp += prohibitive_adjectives[-1]
        all_exp += " while not using "
        for i in range(0, len(prohibitive_ingredients) - 1): 
            all_exp += prohibitive_ingredients[i] + " or "   
        all_exp += prohibitive_ingredients[-1]
    #prohibitive adjectives and prohibitive dishes and prohibitive ingredient - not possible given current overlays
    
    all_exp += "."
    print("All overlays explanation:", all_exp)
    explanations["all"] = all_exp

    '''
    Construct "goal requirement" explanation
    '''

    '''
    Construct "mathematical justification" explanation
    '''

    template_math = exp_json['math']
    math_exp = template_math["beginning"] + str(relevant_values[0]) + " which is " + str(relevant_values[1]) + " higher than the next best action."
    print("Mathematical explation: ", math_exp)
  
    return True


def get_clauses(overlays):
    permissive_adjectives = []
    permissive_dishes = []
    permissive_ingredients = []
    prohibitive_adjectives = []
    prohibitive_dishes = []
    prohibitive_ingredients = []
    for overlay in overlays:
        o_type = overlay["overlay_type"]
        if type(overlay["params"]) != list:
            overlay["params"] = [overlay["params"]]
        if o_type == "permit" or o_type == "transfer":
            for param in overlay["params"]:
                if param in NUTRITIONAL:
                    permissive_adjectives.append(param)
                elif param in DISHES:
                    print("here")
                    permissive_dishes.append(param)
                elif param in INGREDIENTS:
                    permissive_ingredients.append(param)
        elif o_type == "prohibit":
            for param in overlay["params"]:
                if param in NUTRITIONAL:
                    prohibitive_adjectives.append(param)
                elif param in DISHES:
                    prohibitive_dishes.append(param)
                elif param in INGREDIENTS:
                    prohibitive_ingredients.append(param)
    return permissive_adjectives, permissive_dishes, prohibitive_adjectives, prohibitive_dishes, permissive_ingredients, prohibitive_ingredients
        

#     explanations["none"]


#     # query 0: "why did you use _____?"
#     # answer: "You asked me to make [nutr][dish], and [ingred] is required to make [dish][nutr]"
#     # Handling: Ov_0, Ov_3, Ov_5 -- to determine what type of dish to make
#     # Handling: Ov_1 and Ov_9-- to determine what nutritions are allowed
#     if res["key"] == "Qv_0":
#         template = exp_json["Qv_0"]
#         ingredient = res["params"][0]
#         flag = False
#         for action in action_seq:
#             if "gather({i}:".format(i =ingredient) in str(action):
#                 flag = True
#         # if ingredient isn't gathered, respond with error
#         if flag == False:
#             explanation = template["ingred_not_used"].format(ingredient)
#         # otherwise, identify which overlay's action space includes that ingredient
#         else:
#             for overlay in overlays:
#                 if ingredient in overlay["overlay_action_space"]:
#                     relevant_overlays[overlay["key"]].append(overlay)
            
#             # Beginning of explanation
#             # If a dish was specificed: "You asked me to make"
#             dish, dish_clause = get_dish_exp(relevant_overlays, template, ingredient)
#             # no dish was specified
#             if not dish:
#                 begin = template["begin"]["no_dish_ov"]
#                 # param = "breakfast"
#                 # dish_clause = template["dish"].format(param)
#             else:
#                 begin = template["begin"]["dish_ov"]

#             # Nutrtional details
#             # If nutr info was permitted add those qualities: " healthy, fruity"
#             nutr, nutr_clause = get_nutritional_exp(relevant_overlays, template)
            
#             # Closing explanation
#             if nutr_clause and dish_clause:
#                 end_clause = template["end"]["nutr_dish_ovs"].format(dish, nutr) #TKTK what if multiple params?
#             elif dish_clause:
#                 end_clause = template["end"]["dish_ov"].format(dish) #TKTK what if multiple params?
#             elif nutr_clause:
#                 end_clause = template["end"]["nutr_ov"].format(nutr) #TKTK what if multiple params?
#             # no clear reason based on preferences
#             else:
#                 explanation = "This was not based on any directives you gave me. I randomly chose to use {}.".format(ingredient)
#                 print(explanation)
#                 return True

#             ingredient_clause = template["ingred"].format(ingredient)
#             explanation = begin + nutr_clause + dish_clause + ingredient_clause + end_clause

#     # query 1: "can I use [ingredient] instead?"
#     # answer: "Yes, we could have used {} since it is a valid ingredient of {} that is also {}"
#     # Handling: Ov_0, Ov_3, Ov_5 -- to determine what type of dish to make
#     # Handling: Ov_1 and Ov_9-- to determine what nutritions are allowed
#     elif res["key"] == "Qv_1":
#         template = exp_json["Qv_1"]
#         ingredient = res["params"][0]
#         base_action_space = set(overlays[0]["overlay_action_space"])
#         for i in range(1, len(overlays)):
#             base_action_space.intersection_update(set(overlays[i]["overlay_action_space"]))
#         # if it is not in the intersection, find which action space prohibited the use of the ingredient
#         if ingredient not in base_action_space:
#             for overlay in overlays:
#                 if ingredient not in overlay["overlay_action_space"]:
#                     relevant_overlays[overlay["key"]].append(overlay)
            
#             # Beginning of explanation
#             # "No, we couldn't use [ingred]" becuase it is not"
#             begin = template["begin"]["no"].format(ingredient)
            
#             # Nutrtional details
#             nutr, nutr_clause = get_nutritional_exp(relevant_overlays, template)

#             # Dish details
#             # "is not an ingredient of [dish]"
#             dish, dish_clause = get_non_dish_exp(relevant_overlays, template, ingredient)

#             # Add a transition if there is both a nutritional and a dish reason 
#             # " and also not"
#             if nutr_clause and dish_clause:
#                 nutr_clause += template["trans"]["no"]
#             # if no information precluded it
#             # if not nutr_clause and not dish_clause:
#             #     explanation = ""

#             period = template["end"]
        
#             explanation = begin + nutr_clause + dish_clause + period  

#         # Otherwise, report that is a valid ingredient for __ reasons
#         else:
#             for overlay in overlays:
#                 if ingredient in overlay["overlay_action_space"]:
#                     relevant_overlays[overlay["key"]].append(overlay)

#             # "Yes, we could have used {} since it is a valid ingredient of"
#             begin = template["begin"]["yes"].format(ingredient)
#             # Dish from all overlays 
#             dish, dish_clause = get_dish_exp(relevant_overlays, template, ingredient)
#             # Nutrition from all overlays 
#             nutr, nutr_clause = get_nutritional_exp(relevant_overlays, template)
#             # Transition "that is also"
#             if nutr_clause and dish_clause:
#                 dish_clause += template["trans"]["yes"]
#             period = template["end"]

#             explanation = begin + dish_clause + nutr_clause + period

    
#     # query 2: "why did you ____ with the ___?"
#     # answer: "I {} the {} because you asked me to responsible for the {}"
#     # Handling: Ov_3, Ov_5, and Ov_9 - to determine do transfers AND Ov_4 and Ov_5 - to determine say transfers
#     # TKTKTK what should we do with appliances?
#     elif res["key"] == "Qv_2":
#         action, ingredient = res["params"]
#         template = exp_json["Qv_2"]

#         # find all the transfer overlays with the ingredient
#         for overlay in overlays:
#             type = overlay["key"]
#             if (type == 'Ov_3' or type == 'Ov_5') and ingredient in overlay["overlay_action_space"]:
#                 do_transfer_overlays[overlay["key"]].append(overlay)
#             elif (type == 'Ov_4' or type == 'Ov_5') and ingredient in overlay["overlay_action_space"]:
#                 say_transfer_overlays[overlay["key"]].append(overlay)
        
#         # if no transfer overlays:
#         if not do_transfer_overlays and not say_transfer_overlays:
#             explanation = "You didn't specify who should be in charge of that. You are welcome to {} with {} if you want!".format(action, ingredient)
            
#         else:
#             # get DO dish
#             assigned_dish, do_dish_clause  = get_dish_exp(do_transfer_overlays, template, ingredient)
#             if assigned_dish:
#                 begin = template["do_transfered"].format(action, ingredient)
#                 dish_clause = do_dish_clause
#             # otherwise get SAY
#             else:
#                 say_dish_clause = ""
#                 for i in range(0, len(say_transfer_overlays["Ov_4"])):
#                     said_dish = str(say_transfer_overlays["Ov_4"][i]["params"][0])
#                     say_dish_clause += template["dish"].format(said_dish)
#                 for i in range(0, len(say_transfer_overlays["Ov_5"])):
#                     said_dish = str(say_transfer_overlays["Ov_5"][i]["params"][1])
#                     say_dish_clause += template["dish"].format(said_dish)
#                 if said_dish:
#                     begin = template["say_transfered"].format(action, ingredient)
#                     dish_clause = say_dish_clause
#             explanation = begin + dish_clause + template["end"]
#     # q3: why did you make [] first?
#     # Handling: Ov_0 -- to determine ordering
#     elif res["key"] == "Qv_3":
#         meal, order= res["params"]
#         for overlay in overlays:
#             if overlay["key"] == "Ov_0":
#                 m1, m2 = overlay["params"]
#                 if m1 == meal and order == "first":
#                     explanation = "You asked me to make {} first".format(meal)
#                 elif m2 == meal and order == "second":
#                     explanation = "You asked me to make {} second".format(meal)
#                 else:
#                     explanation = "I actually didn't make {} {}".format(meal, order)
#             else:
#                 explanation = "You didn't give me any instructions on what order to prepare things in. I can make the {} {} if you want!".format(meal, order)
#     # query 4: "why didn't you use _____?"
#     # answer: "You asked me to make [nutr][dish], and [ingred] is not an ingredient of {} that is {}.
#     # Handling: Ov_0, Ov_3, Ov_5 -- to determine what type of dish to make
#     # Handling: Ov_1 and Ov_9-- to determine what nutritions are allowed
#     elif res["key"] == "Qv_4":
#         template = exp_json["Qv_4"]
#         ingredient = res["params"][0]
#         flag = False

#         for action in action_seq:
#             if "gather({i}:".format(i =ingredient) not in str(action):
#                 flag = True
#         # if ingredient was actually indeed gathered, respond with error
#         if flag == True:
#             explanation = template["ingred_used"].format(ingredient)
#         # otherwise, identify which overlay's action space excluded that ingredient
#         else:
#             for overlay in overlays:
#                 if ingredient not in overlay["overlay_action_space"]:
#                     relevant_overlays[overlay["key"]].append(overlay)
            
#             # Beginning of explanation
#             # If a dish was specificed: "You asked me to make"
#             dish, dish_clause = get_non_dish_exp(relevant_overlays, template, ingredient)
#             # no dish was specified
#             if not dish:
#                 begin = template["begin"]["no_dish_ov"]
#             else:
#                 begin = template["begin"]["dish_ov"]

#             # Nutrtional details
#             # If nutr info was permitted add those qualities: " healthy, fruity"
#             nutr, nutr_clause = get_nutritional_exp(relevant_overlays, template)
            
#             # Closing explanation
#             if nutr_clause and dish_clause:
#                 end_clause = template["end"]["nutr_dish_ovs"].format(dish, nutr) #TKTK what if multiple params?
#             elif dish_clause:
#                 end_clause = template["end"]["dish_ov"].format(dish) #TKTK what if multiple params?
#             elif nutr_clause:
#                 end_clause = template["end"]["nutr_ov"].format(nutr) #TKTK what if multiple params?

#             ingredient_clause = template["ingred"].format(ingredient)
            
            
#             explanation = begin + nutr_clause + dish_clause + ingredient_clause + end_clause
#     else:
#         explanation = "That query was invalid. Please ask again!"
#     print(explanation)
#     return True


# def check_order(meal, order):
#     ### TKTK how to do this
#     # check if putinmicrowave (pastry) is before/after pourwater/other
#     return True

# def get_nutritional_exp(relevant_overlays, template):
#         nutr = ""   
#         # add all the permissive nutritional qualities
#         for i in range(0, len(relevant_overlays["Ov_1"])):
#             if i == 0:
#                 count = "al_one"
#             else:
#                 count = "extras"
#             param = str(relevant_overlays["Ov_1"][i]["params"][0])
#             nutr += template["nutr"][count].format(param)
#         # add all the prohibited nutritional qualities
#         for i in range(0, len(relevant_overlays["Ov_9"])):
#             if i == 0:
#                 count = "not_al_one"
#             else:
#                 count = "not_extras"
#             param = str(relevant_overlays["Ov_9"][i]["params"][0])
#             nutr += template["nutr"][count].format(param)
#         # If no dish was assigned in overlays
#         if not nutr:
#             # param = "breakfast"
#             # dish_clause = template["dish"].format(param)
#             param = ""
#             nutr = ""
#         return param, nutr



# def get_dish_exp(relevant_overlays, template, ingredient):
#     Ov0_len = len(relevant_overlays["Ov_0"])
#     Ov3_len = len(relevant_overlays["Ov_3"])
#     Ov5_len = len(relevant_overlays["Ov_5"])
#     dish_clause = ""
#     # make sure its a valid ingredient
#     if ingredient not in INGREDIENTS and ingredient not in MEAL and ingredient not in SIDE:
#         param = ""
#         dish_clause = ""
#         return param, dish_clause
#     for i in range(0, Ov0_len):
#         # determine which dish is relevant:
#         # TKTK what if multiple types of this overlay???
#         if ingredient not in relevant_overlays["Ov_0"][i]["overlay_action_space"]:
#             continue
#         ingred_in_aspace = relevant_overlays["Ov_0"][i]["overlay_action_space"][ingredient]
#         if "making_oatmeal" in ingred_in_aspace:
#             param = "oatmeal"
#             if param not in dish_clause:
#                 dish_clause += template["dish"].format("oatmeal")
#         elif "making_pastry" in ingred_in_aspace:
#             param = "a pastry"
#             if param not in dish_clause:    
#                 dish_clause += template["dish"].format("a pastry")
#         elif "making_cereal" in ingred_in_aspace:
#             param = "cereal"
#             if param  not in dish_clause:
#                 dish_clause += template["dish"].format("cereal")
#     # If one dish was assigned: " oatmeal"
#     for i in range(0, Ov3_len):
#         param = str(relevant_overlays["Ov_3"][i]["params"][0])
#         if param not in dish_clause:
#             dish_clause += template["dish"].format(param)
#     for i in range(0, Ov5_len):
#         param = str(relevant_overlays["Ov_5"][i]["params"][0])
#         if param not in dish_clause:
#             dish_clause += template["dish"].format(param)

#     # If no dish was assigned in overlays
#     if not dish_clause:
#         param = ""
#         dish_clause = ""
      
    
#     return param, dish_clause

# def get_non_dish_exp(relevant_overlays, template, ingredient):
#     Ov0_len = len(relevant_overlays["Ov_0"])
#     Ov3_len = len(relevant_overlays["Ov_3"])
#     Ov5_len = len(relevant_overlays["Ov_5"])
#     dish_clause = ""
#     # make sure its a valid ingredient
#     if ingredient not in INGREDIENTS and ingredient not in MEAL and ingredient not in SIDE:
#         param = ""
#         dish_clause = ""
#         return param, dish_clause
#     for i in range(0, Ov0_len):
#         # determine which dish is relevant:
#         # TKTK what if multiple types of this overlay???
#         if ingredient not in relevant_overlays["Ov_0"][i]["overlay_action_space"]:
#             param1 = str(relevant_overlays["Ov_0"][i]["params"][0])
#             param2 = str(relevant_overlays["Ov_0"][i]["params"][1])
#             if (param1 == "oatmeal" or param2 == "oatmeal") and ingredient not in learning_util.ALL_OATMEAL_INGREDIENTS:
#                 param = "oatmeal"
#                 if param not in dish_clause:
#                     dish_clause += template["dish"].format("oatmeal")
#             elif (param1 == "cereal" or param2 == "cereal") and ingredient not in learning_util.CEREAL_INGREDIENTS:
#                 param = "cereal"
#                 if param not in dish_clause:
#                     dish_clause += template["dish"].format("cereal")
#             elif (param1 == "pastry" or param2 == "pastry") and ingredient not in learning_util.PASTRY_LIST:
#                 param = "pastry"
#                 if param not in dish_clause:
#                     dish_clause += template["dish"].format("pastry")
#     # If one dish was assigned: " oatmeal"
#     for i in range(0, Ov3_len):
#         param = str(relevant_overlays["Ov_3"][i]["params"][0])
#         if param not in dish_clause:
#             dish_clause += template["dish"].format(param)
#     for i in range(0, Ov5_len):
#         param = str(relevant_overlays["Ov_5"][i]["params"][0])
#         if param not in dish_clause:
#             dish_clause += template["dish"].format(param)

#     # If no dish was assigned in overlays
#     if not dish_clause:
#         param = ""
#         dish_clause = ""
      
    
#     return param, dish_clause


if __name__ == '__main__':
    pc = ProcessCommand(INGREDIENTS, MEAL, SIDE, SHELF, NUTRITIONAL, DEST)
    overlay_input = []
    count = 1
    while True:
        in_cmd = input("input command: ").strip()
        if not in_cmd:
            break
        else:
            res = pc.process_command(in_cmd)
            res["id"] = count
            overlay_input.append(res)
            count += 1
    
    rng = np.random.default_rng(SEED)
    model_args = {"goal_condition": True} ## TKTK anything I need to add here?
    action_seq, overlays = model_run(overlay_input, rng, model_args, save=True)
    # print("BEGGINNING EXPLANATIONS of :", action_seq)
    # while True:
    #     generate_explanations(action_seq, overlays)  
