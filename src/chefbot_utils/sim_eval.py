#!/usr/bin/env python

import os
import json
import pickle
import pdb
import torch
import pddlgym
from matplotlib import pyplot as plt
from collections import defaultdict, Counter, OrderedDict
from itertools import product 
from copy import deepcopy

import random
import pandas as pd
import numpy as np
try:
    # from pddlgym_planners.ff import FF
# from lnn import (Predicate, Variable,
#                  Exists, Implies, ForAll, Model, Fact, World)
    from state_transformer import StateTransformer, PrologClassifier
    from overlay import Overlay, OverlayList, PrologOverlay, PrologOverlayList, OverlayFactory
    from shield import AlternateShield, RefineShield, ShieldFactory
    from learning import DQN, DQNAgent, ShieldAgent, load_action_space
    from learning_util import (ROLL, MUFFIN, JELLYPASTRY, PIE, EGG, CEREAL,
                               MILK, WATER, PLAINOATMEAL, FRUITYOATMEAL, CHOCOLATEOATMEAL,
                               PBBANANAOATMEAL, FRUITYCHOCOLATEOATMEAL, PBCHOCOLATEOATMEAL,
                               PASTRY_LIST, generate_train_test_meals)
    from pddl_util import load_environment, dist_to_goal, PDDL_DIR, FeatureDomainGenerator
    from htn import CookingHTNFactory, generate_action_space
except ModuleNotFoundError as e:
    from chefbot_utils.state_transformer import StateTransformer, PrologClassifier
    from chefbot_utils.overlay import Overlay, OverlayList, PrologOverlay, PrologOverlayList, OverlayFactory
    from chefbot_utils.shield import AlternateShield, RefineShield, ShieldFactory
    from chefbot_utils.learning import (DQN, DQNAgent, ShieldAgent, load_action_space)
    from chefbot_utils.pddl_util import load_environment, dist_to_goal, PDDL_DIR
    from chefbot_utils.learning_util import (ROLL, MUFFIN, JELLYPASTRY, PIE, EGG, CEREAL,
                                             MILK, WATER, PLAINOATMEAL, FRUITYOATMEAL, CHOCOLATEOATMEAL,
                                             PBBANANAOATMEAL, FRUITYCHOCOLATEOATMEAL, PBCHOCOLATEOATMEAL,
                                             PASTRY_LIST, generate_train_test_meals)

    from chefbot_utils.htn import CookingHTNFactory, generate_action_space, FeatureDomainGenerator


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs')

# Where all the saving occurs
DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../models")
SAVE_MODEL_PATH = os.path.join(DATA_PATH, "trained_models")
COMPARE_AGENTS_PATH = os.path.join(DATA_PATH, "compare_agents_results")
RETRAIN_AGENTS_PATH = os.path.join(DATA_PATH, "retrain_agents_results")
WRONG_PRED_REWARD = -.5
WRONG_ACTION_TYPE_REWARD = WRONG_PRED_REWARD * .5
FAILIURE_REWARD = -1.0
SEED1 = 0
SEED2 = 1000
SEED3 = 888888
ROBOT_TRAINING_SEED = SEED3
RNG = np.random.default_rng(ROBOT_TRAINING_SEED)


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


### Code for handling errors during simulated colaborations
def undo_incorrect_pastry_action(wrong_action, possible_actions):
    """
    Some incorrect actions need to be undone with an additional action(s)
    before the task can be completed properly. This function determines
    this sequence of corrective actions based on the input.

    @input wrong_action, str: The action to correct
    @input possible_actions, np.ndarray of PDDLLiterals: The grounded action
                                                         literals the robot can perform.
    """
    need_to_undo = False
    ret_action = None
    n_backtrack = 1
    # If the robot puts the wrong thing in the microwave,
    # then we need to remove it first
    if "putinmicrowave" in wrong_action:
        corrective_temp = "takeoutmicrowave({side}:pastry,do:action_type)"
        need_to_undo = True
        # n_backtrack = 1
        n_backtrack = 0
    # If the robot removes something too early,
    # first we need to put it back in then we need
    # to turn the microwave back on.
    elif "takeoutmicrowave" in wrong_action:
        corrective_temp = "putinmicrowave({side}:pastry,do:action_type)"
        need_to_undo = True
        n_backtrack = 2 

    if need_to_undo:
        for p in PASTRY_LIST:
            if p in wrong_action:
                ret_action = corrective_temp.format(side=p)

        for a in possible_actions:
            if ret_action == str(a):
                ret_action = a

    return need_to_undo, ret_action, n_backtrack

def same_action_diff_intervention_type(pred_action, gt_action):
    """
    Checks if two actions are the same except for action type (say/do)
    """
    if pred_action.predicate.name == gt_action.predicate.name:
        pred_vars = set([v.name for v in pred_action.variables])
        gt_vars = set([v.name for v in gt_action.variables])
        diff = gt_vars - pred_vars
        return len(diff) == 1 and ("say" in diff or "do" in diff)

    return False


def initialize_simulation():
    """
    Generate the necessary files and dirs for running the simulations below.
    """
    env = load_environment(goal_side="jellypastry", goal_main="chocolateoatmeal")
    gen = FeatureDomainGenerator(env)
    gen.save_domain_to_file()

    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)
    if not os.path.isfile(os.path.join(DATA_PATH, "ACTION_SPACE.npy")):
        print("[initialize_simulation] Generating action space (may take a bit)...")
        generate_action_space(PDDL_DIR)



class InteractionGenerator(object):
    def __init__(self, testing_plans, training_plans, action_space, rng):
        """
        @input testing_plans, list of dicts: each dict paramterizes an CookingHTN.
                                             Used for testing the base model
        @input training_plans, list of dicts: each dict paramterizes an CookingHTN.
                                             Used for training the base model
        @input action_space: numpy array of PDDLLiterals: The complete action space of hte agent
        @input, rng: np.random.default_rng obj

        """
        self.rng = rng
        self._orig_action_types = {"main":"do", "side": "say"}
        self._orig_meal_order = {"main": "first", "side": "last"}
        self._testing_plans = testing_plans
        self._training_plans = training_plans
        # self._meal_data = self._plan_list_to_dict()
        self.sf = ShieldFactory(action_space)
        self.of  = OverlayFactory()
        self.htnf = CookingHTNFactory(main=PLAINOATMEAL, side=PIE,
                                      liquid=MILK, action_types={"main":"do",
                                                                 "side":"say"},
                                      order="main first")

        # self._training_list = [self._training_set_interactions_gen]
        self.n_retrain_meals = len(training_plans)

    def _plan_list_to_dict(self, plan_list):
        ret_dict = {"main":[], "side":[], "liquid":[],
                    "main_at":[], "side_at":[], "main_order":[],
                    "side_order":[]}
        for p in plan_list:
            ret_dict["main"].append(p["main"])
            ret_dict["side"].append(p["side"])
            ret_dict["liquid"].append(p["liquid"])
            ret_dict["main_at"].append(p["action_types"]["main"])
            ret_dict["side_at"].append(p["action_types"]["side"])
            ret_dict["main_order"].append(p["main_order"])
            ret_dict["side_order"].append(p["side_order"])

        print("[simInteraction] ", ret_dict)
        return ret_dict

    def _get_side_order(self, main_order):
        if main_order == "first":
            side_order = "last"
        else:
            side_order = "first"

        return side_order

    def _get_dish_types(self, main):
        if CEREAL in main:
            main_dish = ["cereal"]
        else:
            main_dish = ["oatmeal"]

        return main_dish, ["pastry"]

    def _generate_overlay_list(self, main, side, liquid, main_order):
        """
        Generate list of overlays based on ground truth HTN parameters
        """
        ret_list = []
        main_preds = []
        side_preds = []

        main_dish, side_dish = self._get_dish_types(main)

        if "oatmeal" in main_dish:
            if liquid == WATER:
                main_preds.append("not dairy")

        side_order = self._get_side_order(main_order)

        if side in [EGG]:
            side_preds = ["protein", "healthy"]
        elif side in [MUFFIN]:
            side_preds = ["gluten", "healthy"]
        elif side in [ROLL]:
            side_preds = ["gluten"]
        elif side in [PIE]:
            side_preds = ["fruity"]
        elif side in [JELLYPASTRY]:
            side_preds = ["sweet"]

        ret_list.append(self.of.make_meal_overlay(dish=main_dish, order=[main_order]))
        ret_list.append(self.of.make_meal_overlay(dish=side_dish, order=[side_order],
                                                  preds=side_preds))

        main_preds = []
        if main in [FRUITYOATMEAL]:
            main_preds.append("fruity")
        elif main in [PBCHOCOLATEOATMEAL]:
            main_preds.extend(["protein", "sweet"])
            # main_preds.extend(["protein"])
        elif main in [FRUITYCHOCOLATEOATMEAL]:
            main_preds.extend(["sweet"])
        elif main in [CHOCOLATEOATMEAL]:
            main_preds.extend(["sweet"])
        elif main in [PBBANANAOATMEAL]:
            main_preds.extend(["protein", "fruity"])



        if main_preds:

            ret_list.append(self.of.make_meal_overlay(order=[main_order], preds=main_preds,
                                                      logical_op="or"))

            # ret_list.append(self.of.make_meal_overlay(dish=main_dish, order=[main_order],
            #                                           preds=main_preds,
            #                                           logical_op="or"))


        return ret_list

    def _generate_shield_list(self, main, side, liquid, m_at, s_at, main_order, side_order):
        """
        Generate a list of shields based on the paramteters of the Ground truth HTN.
        """
        ret_list = []

        # Generate alternate shields for all undesired pastries
        # ret_list.append(self.sf.pastry_forbid_shield(desired_pastry=EGG))

        main_dish, side_dish = self._get_dish_types(main)
        undesired_pastries = [p for p in PASTRY_LIST if not p == side]
        for p in undesired_pastries:
            name="{}_to_{}_alternate".format(p, side)
            ret_list.append(self.sf.gather_pastry_alternate(orig_item=p, alt_item=side))

        # Creates a shield based on desired liquid (milk or water)
        ret_list.append(self.sf.alternate_liquid(liquid))

        ret_list += self.sf.alternate_action_type_by_meal(main_dish[0], m_at)
        ret_list += self.sf.alternate_action_type_by_meal(side_dish[0], s_at)

        if main_order == "first":
            ret_list += self.sf.forbid_meal_from_order(main, side)
        else:
            ret_list += self.sf.forbid_meal_from_order(side, main)
        # ret_list += self.sf.alternate_action_type_pastry(s_at, side)

        return ret_list


    def _testing_set_interactions_gen(self):

        for p in self._testing_plans:

            m,s, l = p["main"], p["side"], p["liquid"]
            m_at, s_at = p["action_types"]["main"], p["action_types"]["side"]
            if p["order"] == "main first":
                m_order, s_order = "first", "last"
            else:
                m_order, s_order = "last", "first"

            overlays = self._generate_overlay_list(m,s,l,m_order)
            shields = self._generate_shield_list(m,s,l,m_at, s_at, m_order, s_order)
            main_dish, side_dish = self._get_dish_types(m)

            # Change action types
            overlays += self.of.action_type_assignment_overlays(p["action_types"], main_dish[0],
                                                                side_dish[0])

            self.htnf.set_meal_type(**p)


            name = "{}_{}_with_{}_{}_and_{}_{}".format(m_at, m, l, m_order, s_at, s)
            yield (self.htnf.use_immediately(), overlays, shields, name)


    def _get_gt_htn_order(self, main_order):
        if main_order == "first":
            gt_htn = self.htnf.ingredients_first_main_first()
        elif main_order == "last":
            gt_htn = self.htnf.ingredients_first_side_first()

        return gt_htn

    def generate_training_htns(self):
        for p in self._training_plans:
            self.htnf.set_meal_type(**p)
            yield self.htnf.use_immediately()

    def generate_interactions(self):
        """
        Returns a generator object used to simulate a cooking interactions based
        on the htn parameters from self.testing_plans

        Retuns: name, str: The name of this set of interaction.
                htn, HTN: An htn paramterized by each testing_plan dict.
                overlays, list of Overlays.
                shields, list of Shields.
                meal_name, the name of the meal corresponding to htn.
        """
        name = "testing_meal"
        for interactions in self._testing_set_interactions_gen():
            yield name, interactions



    def __iter__(self):
        for name, task in self._task_dict.items():
            f, arg = task
            print("{}\n".format(name))
            for interactions in f(arg):
                yield name, interactions






def DQN_trainer_from_htn(training_htns, rng, model_args=None, exploration_max=1, epochs=1, save_dir="trained_models"):
    """
    Trains and stores a DQN.

    @input training_htns, list of dicts: Parameters for ground truth HTNs used to train the model. 
    @input rng, np.random.default_rng obj
    @input model_args: dict of strs. Additional model parameters for the DQNAgent
    @input exploration_max, int: exploration rate
    @input epochs, int: number of times to loop through training_htns
                        (N training episodes == len(training_htns) * epochs)

    @input save_dir, str: The name of the directory where the trained models will be stored.
    """

    _, feature_env = load_environment(pddl_dir=None, load_feature_env=True)
    action_space = load_action_space(os.path.join(DATA_PATH, "ACTION_SPACE.npy"))

    default_model_args = deepcopy(DEFAULT_MODEL_ARGS)
    default_model_args["feature_env"] = feature_env
    default_model_args["rng"] = rng
    default_model_args["action_space"] = action_space

    if not model_args is None:
        default_model_args.update(model_args)

    # instantiate DQNAgent
    agent = DQNAgent(**default_model_args)




    interaction_generator = InteractionGenerator(training_plans=training_htns,
                                                 testing_plans=[],
                                                 action_space=action_space,
                                                 rng=rng)
    total_rewards = []
    for epoch in range(epochs):
        print("\nSEpoch: {}\n".format(epoch))
        for ep, htn in enumerate(interaction_generator.generate_training_htns(), start=1):
            ep += epoch * len(training_htns)
            print("\nStarting Episode: {}\n".format(ep))
            trace = []
            performed_actions = []
            htn.reset()

            env =  load_environment(None, False,
                                    goal_main=htn.main,
                                    goal_side=htn.side)
            state,_ = env.reset()
            total_reward = 0 # How much reward in this epoch?
            correct_preds = []
            n_errors = []
            loss_ep, n_corrections = 0, 0
            goal_dist = 0

            corrective_action = None
            corrective_actions = []

            # htn = ff_planner(env.domain, state)
            # plan_copy = htn.copy()
            # plan_action = []
            step = 0
            done = False
            failed = False
            possible_actions = list(env.action_space.all_ground_literals(state))
            while not done and step < 45:

                print("\n\n#########STEP {}#########".format(step))
                # print("[trainer] Remaining corrective actions action: ",
                      # corrective_actions)
                s_bin = agent.get_transformed_state(state)
                pred_action = agent.act(state, possible_actions)

                # plan_action = htn.pop(0)

                # print("Corrective actions: ", corrective_actions)
                if len(corrective_actions) > 0:
                    corrective_action = corrective_actions.pop(0)
                else:
                    plan_action, already_performed = htn.get_next_action(agent.action_space, performed_actions)

                if plan_action is None and corrective_action is None:
                    break
                # if ep < num_episodes + 1:

                # elif ep < num_episodes * .1 or step == 0:
                elif  step == 0:
                    # print("remaining htn: ", htn)

                    # print("[DQN_trainer_from_htn] Using htn action: ", plan_action)
                    action = plan_action

                elif corrective_action:

                    action = corrective_action
                    # corrective_action = None
                    # print("[DQN_trainer_from_htn] Appying corrective action: ",action)
                else:
                    # print("[DQN_trainer_from_htn] Predicting action: ", pred_action)
                    action = pred_action

                # print("[DQN_trainer_from_htn] Plan action: ", plan_action)
                # print("[DQN_trainer_from_htn] Pred action: ", pred_action)
                # print("[DQN_trainer_from_htn] Chosen action: ", action)

                # HTN currently only manipulates string reps of actions
                # First, add any actions to trace that the HTN suggests but that
                # have already been performed
                trace+= already_performed
                # if not str(action) in trace:
                if len(corrective_actions) == 0:
                    trace.append(str(action))
                # Then Add the selected action (may be removed later)
                performed_actions.append(str(action))
                already_performed = []

                # pred_state_next, _, pred_done, _ = env.sample_transition(pred_action)
                next_state, reward, pddl_done, info = env.step(action)
                # goal_dist  = dist_to_goal(env, next_state)
                print("{} ACTION: {} goal dist: {}\n".format(agent.agent_name, action,
                                                            goal_dist))


                # new_reward, done = agent.check_rewards(next_state)
                failed = agent.check_errors(next_state)
                if not failed:
                    possible_actions = list(env.action_space.all_ground_literals(next_state))
                    next_state, new_reward, done = agent.check_rewards(env, possible_actions,
                                                                    [htn.main, htn.side])
                else:
                    done = True
                    new_reward = FAILIURE_REWARD
                # next_state, new_reward, done = agent.check_rewards(env)
                reward += new_reward
                # reward *= 100
                s_next_bin = agent.get_transformed_state(next_state)
                # assert not state.literals == next_state.literals, print(possible_actions)
                # check if two np arrays are the same
                # assert not np.array_equal(s_bin, s_next_bin)

                # correct_preds.append(pred_action == plan_action)
                # correct += pred_action == plan_action

                # if not action == plan_action:
                #     reward += WRONG_PRED_REWARD
                wrong_action = not htn.check_action_in_seq(htn.root, action) and \
                    corrective_action is None
                # print("[DQN_trainer_from_htn] trace:  ", trace)
                # if not htn.check_seq_in_tree(htn.root, trace) and not corrective_action:
                if wrong_action:
                    correct_preds.append(False)
                    n_errors.append(True)
                    try:
                        last_action = trace.pop(-1)
                        need_to_undo, undo_action, n_backtracks = undo_incorrect_pastry_action(last_action,
                                                                                              action_space)

                        # print("[DQN_trainer_from_htn] trace:  ", trace)
                        if same_action_diff_intervention_type(pred_action, plan_action):
                            n_corrections += .5
                            reward += WRONG_ACTION_TYPE_REWARD
                            corrective_actions.append(plan_action)
                        elif need_to_undo:
                            # print("[DQN_trainer_from_htn]  undoing ", last_action)
                            # trace.pop(-1)
                            prev_actions = htn.get_k_previous_actions(agent.action_space,
                                                                       n_backtracks)
                            if n_backtracks == 0:
                                corrective_actions = [undo_action, plan_action]

                            elif undo_action in prev_actions:
                                corrective_actions  = prev_actions 
                            else:
                                corrective_actions  = [undo_action] + prev_actions

                            # Remove duplicates
                            # print("[compare_agents] trace:  ", trace)
                            # corrective_actions.append(prev_action)
                            # corrective_actions.append(plan_action)
                            # print("[compare_agents] Adding corrective actions: ", corrective_actions)
                            n_corrections += len(corrective_actions)
                            reward += WRONG_PRED_REWARD * len(corrective_actions)

                        else:
                            n_corrections += 1
                            reward += WRONG_PRED_REWARD
                            corrective_actions.append(plan_action)

                        # corrective_action = plan_action

                        # for corrective_action in corrective_actions:
                        #     agent.remember(s_bin, corrective_action, 0.0, s_next_bin, done)
                    except IndexError as e:
                        print(e)
                        corrective_action  = None

                else:
                    correct_preds.append(True)
                    # n_errors.append(False)
                    corrective_action = None


                # print("[DQN_trainer_from_htn] Done? {} PDDL Done?: {} Reward: {}".format(done,
                #                                                                 pddl_done,
                #                                                                 reward))
                # assert not np.array_equal(s_bin, s_next_bin)
                # print("[DQN_trainer_from_htn] remembering:{} reward: {} ".format(action, reward))
                agent.remember(s_bin, action, reward, s_next_bin, done)
                # if not pred_action == plan_action:
                #     print("[DQN_trainer_from_htn] Adding incorrect pred to memory")
                #     pred_s_next_bin = agent.get_transformed_state(pred_state_next)
                #     agent.remember(s_bin, pred_action, WRONG_PRED_REWARD,
                #                     pred_s_next_bin, pred_done)

                loss = agent.experience_replay()
                if loss is not None:
                    loss_ep += loss


                state = next_state
                total_reward += reward
                step += 1
                reward = 0
            total_rewards.append(total_reward)
            if loss_ep > 0:
                print("[DQN_trainer_from_htn] writing {} to ep: {}".format(loss_ep, ep))
                writer.add_scalar("Loss/train", loss_ep, ep)

            agent.reset()
            print("\nEpisode {} ID:{} N steps: {} N Errors: {} Mean Errors: {}".format(ep,
                                                                                    htn.name,
                                                                                    htn.n_steps,
                                                                                    n_corrections,
                                                                                    n_corrections / htn.n_steps),
                )
            writer.add_scalar("Total Epoch Reward", total_reward, ep)
            writer.add_scalar("Total Corrections ", n_corrections, ep)
            writer.add_scalar("Mean Corrections ", n_corrections / htn.n_steps, ep)
            # Save model params to dict along with accuracy score


            # only save model if accuracy is higher than previous best
            # First get highest accuracy from model_dict
        name = "DQN_ep_{}".format(epoch)
        print("[DQN_trainer_from_htn] Saving model!!!")
        new_save_dir = os.path.join(DATA_PATH, save_dir, name)
        # Make the save directory if it doesnt exist

        if not os.path.exists(new_save_dir):
            os.makedirs(new_save_dir)

        torch.save(agent.dqn.state_dict(), os.path.join(new_save_dir, "DQN.pt"))
        torch.save(agent.STATE_MEM,  os.path.join(new_save_dir, "STATE_MEM.pt"))
        torch.save(agent.ACTION_MEM, os.path.join(new_save_dir, "ACTION_MEM.pt"))
        torch.save(agent.REWARD_MEM, os.path.join(new_save_dir, "REWARD_MEM.pt"))
        torch.save(agent.STATE2_MEM, os.path.join(new_save_dir, "STATE2_MEM.pt"))
        torch.save(agent.DONE_MEM,  os.path.join(new_save_dir, "DONE_MEM.pt"))
        np.save(os.path.join(new_save_dir, "ACTION_SPACE.npy"), agent.action_space,
                allow_pickle=True)
        with open(os.path.join(new_save_dir, "ending_position.pkl"), "wb") as f:
            pickle.dump(agent.ending_position, f)

        with open(os.path.join(new_save_dir, "num_in_queue.pkl"), "wb") as f:
            pickle.dump(agent.num_in_queue, f)





def compare_agents_htn(testing_plans, training_plans, rng, model_args=None, agent_name="overlay",
                       step=5, start= 5, end=30, max_n_steps=50, save=False, learn=False,
                       save_dir="first_run", model_path="trained"):



    action_space = load_action_space(os.path.join(DATA_PATH, "ACTION_SPACE.npy"))
    # Instantiate interaction generator
    interaction_generator = InteractionGenerator(testing_plans,training_plans,
                                            action_space, rng)

    for ep in range(start, end, step):

        #NOTE:  Temporary feature env. Need to pass it to DQNAgent
        # in order to properly instantiate it. The correct feature
        # 
        _, dummy_feature_env =  load_environment(None, True,)
        print("Starting episode {}".format(ep))
        load_model_path = "{}/DQN_ep_{}".format(model_path, ep)

        default_model_args = deepcopy(DEFAULT_MODEL_ARGS)

        default_model_args["feature_env"] = dummy_feature_env
        default_model_args["rng"] = rng
        default_model_args["action_space"] = action_space
        default_model_args["model_path"] = load_model_path

        # default_model_args  = {"feature_env":dummy_feature_env,
        #             "rng":rng,
        #             "batch_size":64,
        #             "gamma":.99,
        #             "lr":0.002,
        #             "max_memory_size":50000,
        #             "exp_min":0.0,
        #             "exp_decay":0.99,
        #             "exp_rate":0.0,
        #             "pretrained":True,
        #             "load_memories": False,
        #             "fc2_size":32,
        #             "action_space": action_space,
        #             "model_path":load_model_path}

        if not model_args is None:
            default_model_args.update(model_args)

        if agent_name == "overlay":
            agent = DQNAgent(agent_name="overlay", **default_model_args)
        elif agent_name == "shield":
            agent = ShieldAgent(**default_model_args)
        elif agent_name == "control":
            agent = DQNAgent(agent_name="control", overlays=None, **default_model_args)

        for num, name_interactions in enumerate(interaction_generator.generate_interactions()):
            name, interactions = name_interactions

            int_reward_dict = {"reward": 0 , "corrections": 0, "completed":False, "steps":0,
                              "overlays": {}, "shields": {}}
            print("TESTING INTERACTION: ", name)
            gt_htn, overlay_list, shield_list, meal_name = interactions

            env, feature_env =  load_environment(None, True,
                                                 goal_main=gt_htn.main,
                                                 goal_side=gt_htn.side)
            feature_env.goal_condition = model_args["goal_condition"]
            # import pdb; pdb.set_trace()
            agent.feature_env = feature_env
            if agent_name == "shield":
                agent._shield_list = shield_list
            elif agent_name == "overlay":
                overlays = PrologOverlayList(overlay_list)
                agent.overlays = overlays
            # agents = [control_agent, overlay_agent, shield_agent]
            # agents = [shield_agent]
            # agents = [overlay_agent, shield_agent]
            # plan = gt_htn.get_plan(action_space)

            # for i, plan_action in enumerate(plan):
            gt_htn.reset()
            agent.reset()
            trace = []
            performed_actions = []
            corrective_action = None
            corrective_actions = []
            done, failed = False, False
            reward, step, n_corrections, goal_dist = 0, 0, 0, 0
            loss_ep = 0


            print("Agent: {} Starting episode {} meal: {}".format(ep, agent.agent_name, meal_name))
            print("Overlays: ", agent.overlays)

            state, _ = env.reset()
            possible_actions = list(env.action_space.all_ground_literals(state))

            while not done and step < max_n_steps:

                print("\n\n#########STEP {}#########".format(step))
                # Get action predictions
                print("[compare_agents] Remaining corrective actions action: ",
                      corrective_actions)
                if agent.agent_name == "shield":
                    pred_action = agent.act(env, state, possible_actions)
                else:
                    pred_action = agent.act(state, possible_actions)

                print("[compare_agents] pred action: ", pred_action)

                # Get queued corrective actions if they exist
                if len(corrective_actions) > 0:
                    corrective_action = corrective_actions.pop(0)

                # if corrective_action is None:
                else:
                    plan_action, already_performed = gt_htn.get_next_action(action_space,
                                                                        performed_actions)
                    print("[compare_agents] gt_htn action: ", plan_action)
                    print("[compare_agents] already_performed: ", already_performed)

                ### Decide what the actual action will be
                if plan_action is None and corrective_action is None:
                    break
                # if ep < num_episodes + 1:

                # elif ep < num_episodes * .1 or step == 0:
                # elif step == 0:
                elif step == -1:
                    # print("remaining gt_htn: ", gt_htn)

                    print("[compare_agents] Using gt_htn action: ", plan_action)
                    action = plan_action

                elif corrective_action:

                    action = corrective_action
                    # corrective_action = None
                    print("[compare_agents] Appying corrective action: ",action)
                else:
                    print("[compare_agents] Predicting action: ", pred_action)
                    action = pred_action

                # if goal_dist == 5:
                #     import pdb; pdb.set_trace()
                trace += already_performed
                # if not str(action) in trace:
                if not str(action) in trace:
                    print("[Compare] appending to trace: ", action)
                    trace.append(str(action))

                performed_actions.append(str(action))
                already_performed = []

                next_state, new_reward, done, info = env.step(action)
                # goal_dist  = dist_to_goal(env, next_state)
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
                    new_reward = FAILIURE_REWARD
                # new_reward, done = agent.check_rewards(next_state)
                reward += new_reward


                # print("[compare_agents] initial trace:  ", trace)
                print("[compare_agents] corrective action:  ", corrective_action)

                # wrong_action = not gt_htn.check_seq_in_tree(gt_htn.root, trace) and  corrective_action is None
                wrong_action = not gt_htn.check_action_in_seq(gt_htn.root, action) and  corrective_action is None
                print("[compare_agents] wrong action:  ", wrong_action)
                if wrong_action:

                    # n_errors.append(True)
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

                            # Remove duplicates
                            # print("[compare_agents] trace:  ", trace)
                            # corrective_actions.append(prev_action)
                            # corrective_actions.append(plan_action)
                            print("[compare_agents] Adding corrective actions: ", corrective_actions)
                            n_corrections += len(corrective_actions)
                            reward += WRONG_PRED_REWARD * len(corrective_actions)
                        else:
                            reward += WRONG_PRED_REWARD
                            corrective_actions.append(plan_action)
                            # corrective_action = plan_action
                            n_corrections += 1
                    except IndexError as e:
                        print(e)
                        raise e
                        corrective_action  = None

                else:
                    # n_errors.append(False)
                    corrective_action = None

                # agent.update_action_hist(action)
                # print(next_state)
                # assert not next_state == state

                if learn:
                    s_bin = agent.get_transformed_state(state)
                    s_next_bin = agent.get_transformed_state(next_state)
                    agent.remember(s_bin, action, reward, s_next_bin, done)
                    loss = agent.experience_replay()

                    if loss is not None:
                        loss_ep += loss
                        writer.add_scalar("Loss/train", loss_ep, ep)

                state = next_state
                if done:
                    break
                step +=1
                reward = 0

            if learn:
                agent.lr_scheduler.step()
            writer.add_scalar("Mean Corrections ", n_corrections / step, num)
            writer.add_scalar("Total Corrections ", n_corrections, num)

            int_reward_dict["reward"] = reward
            int_reward_dict["corrections"] = n_corrections
            int_reward_dict["steps"] = step
            # int_reward_dict["dist_to_goal"].append(dist_to_goal)
            int_reward_dict["completed"] = done
            if agent_name == "overlay":
                overlay_dict = {}
                [overlay_dict.update(ov.info()) for ov in overlay_list]
                int_reward_dict["overlays"] = overlay_dict
            elif agent_name == "shield":
                shield_dict = {}
                [shield_dict.update(s.info()) for s in shield_list]
                int_reward_dict["shields"] = shield_dict

            print("Ep rewards: ", int_reward_dict)

            if save:

                epoch_name = "epoch_{}".format(ep)
                save_epoch_dir = os.path.join(save_dir,epoch_name)
                print(save_dir)
                print(save_epoch_dir)
                if not os.path.exists(save_epoch_dir):
                    os.makedirs(save_epoch_dir)

                save_file = os.path.join(save_epoch_dir,
                                       "{}_{}.json".format(num, meal_name))
                print("Saving {}...".format(save_file))
                with open(save_file, 'w') as outfile:
                    json.dump(int_reward_dict, outfile)



def generate_all_meal_combos():
    sides = PASTRY_LIST
    oatmeal_mains = [PLAINOATMEAL, FRUITYOATMEAL, CHOCOLATEOATMEAL,
                PBBANANAOATMEAL, FRUITYCHOCOLATEOATMEAL, PBCHOCOLATEOATMEAL]
    cereal_mains = [CEREAL]
    orders = ["main_first", "side_first"]
    liquids = [WATER, MILK]
    oatmeal_count = 0
    for p in product(sides, oatmeal_mains, liquids, orders):
        print(p)
        oatmeal_count += 1
    cereal_count = 0
    for p in product(sides, cereal_mains, orders):
        cereal_count += 1
    print("FINAL MEAL COUNT: ", cereal_count + oatmeal_count)

def train_test_models(seeds, splits, agents, n_comparison_meals=50, total_training_episodes=500, goal_condition=True):
    count = 0
    N_MEALS = 500
    model_args = {"goal_condition": goal_condition}
    for split in splits:
        for run, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            training_htns, testing_htns = generate_train_test_meals(rng, N_MEALS,  split)
            n_retrain_meals = len(training_htns)
            training_epochs = max(int(total_training_episodes / n_retrain_meals), 1 )
            save_model_path = "trained_models/goal_conditioned_{}/num_training_{}/seed_{}".format(goal_condition,
                                                                                                  n_retrain_meals, seed)
            print("training: split: {} seed: {} n_meals: {} n epochs: {} total episodes: {}".format(split, seed, n_retrain_meals,
                                                                                                    training_epochs, total_training_episodes))
            DQN_trainer_from_htn(training_htns, rng, model_args,
                                 epochs=training_epochs, save_dir=save_model_path)
            pdb.set_trace()
            for agent in agents:
                print("Agent: {} count: {}".format(agent, count))
                save_dir = os.path.join(COMPARE_AGENTS_PATH, agent, "goal_conditioned_{}".format(goal_condition),
                                        "num_training_{}/run_{}_seed_{}".format(n_retrain_meals,
                                                                                run, seed))
                with torch.no_grad():
                    compare_agents_htn(testing_htns[:n_comparison_meals], training_htns, rng=rng,
                                       model_args=model_args, agent_name=agent, start=training_epochs-1,
                                       end=training_epochs, step=1, save=True, save_dir=save_dir,
                                       model_path=save_model_path)
                count += 1

def retrain_models_per_meal(seeds, agents, num_training=52, n_epochs=10, n_retrain_meals=50, model_load_epoch=0, goal_condition=True):
    count = 0
    # model_args = {"batch_size":32, "lr":.0002, "load_memories":True}
    model_args = {"batch_size":32, "lr":.002, "load_memories":False, "lr_decay":.9, "goal_condition":goal_condition}
    for run, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        training_htns, _ = generate_train_test_meals(rng, n_retrain_meals,  1.)
        # n_retrain_meals = len(training_htns)
        save_model_path = "trained_models/goal_conditioned_{}/num_training_{}/seed_{}".format(goal_condition,
                                                                                              num_training, seed)
        # print("training: split: {} seed: {}".format(split, seed))
        for agent in agents:
            print("Agent: {} count: {}".format(agent, count))
            for i, meal in enumerate(training_htns):
                save_dir = os.path.join(RETRAIN_AGENTS_PATH, agent, "goal_conditioned_{}".format(goal_condition),
                                        "run_{}_seed_{}/meal_{}".format(run, seed, i))
                meal_training_set = [meal] * n_epochs
                compare_agents_htn(meal_training_set, [], rng=rng, model_args=model_args,
                                   agent_name=agent, start=model_load_epoch,
                                   end=model_load_epoch+1, step=1, save=True,
                                    learn=True, save_dir=save_dir, model_path=save_model_path)
                count += 1

if __name__ == '__main__':
    # random.seed(0)
    import pdb
    initialize_simulation()
    model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../../models/")
    training_results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "../../models/training_by_agent_type/")
    action_space = load_action_space(os.path.join(model_dir, "ACTION_SPACE.npy"))


    # Should result in 26 training meals
    n_new_meals = 50
    training_episodes = 500
    # n_new_meals = 1
    seeds = [SEED1, SEED2, SEED3]
    splits = [.1, .05, .02]
    agents = ["overlay", "control", "shield"]
    # agents = ["shield"]
    num_training = [50, 25, 10 ]
    goal_condition = [True, False]
    for g_c in goal_condition:
        train_test_models(seeds, splits, agents, n_new_meals, total_training_episodes=training_episodes,
                          goal_condition=g_c)
        retrain_models_per_meal(seeds, agents, n_retrain_meals=n_new_meals,num_training=50, n_epochs=10, model_load_epoch=9,
                                goal_condition=g_c)
