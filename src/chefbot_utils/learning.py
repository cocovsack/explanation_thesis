#!/usr/bin/env python

import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from scipy.special import softmax

from collections import defaultdict
from copy import deepcopy

# from pddlgym_planners.ff import FF
try:
        from state_transformer import StateClassifier
        from overlay import OverlayList, PrologOverlayList, PrologOverlay
        from pddl_util import CHECK_TEMPLATE_DICT
except ModuleNotFoundError:
        from chefbot_utils.state_transformer import StateClassifier
        from chefbot_utils.overlay import OverlayList, PrologOverlayList, PrologOverlay, RosOverlayList
        from chefbot_utils.pddl_util import CHECK_TEMPLATE_DICT


# TensorBoard
# try:
from torch.utils.tensorboard import SummaryWriter
# except ImportError:
# print("Cant import tensorboard!")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# writer = SummaryWriter()
torch.manual_seed(0)

# REFS: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


# def SARSADataset():

def load_action_space(path):
        return np.load(path, allow_pickle=True).tolist()

class DQN(nn.Module):

    def __init__(self, state_space, action_space, fc2_size=64):
        super(DQN, self).__init__()
        # state, _ = env.reset()
        # env.action_space._update_objects_from_state(state)
        # self.action_space = list(env.action_space._compute_all_ground_literals(state))
        self.state_space = state_space
        self.action_space = action_space
        fc1_size = 32
        self.linear_relu_stack = nn.Sequential(
                    nn.Linear(self.state_space,fc2_size),
                    nn.ReLU(),
                    nn.Linear(fc2_size,fc1_size),
                    nn.ReLU(),
                    nn.Linear(fc1_size, self.action_space),
                )
    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits




class DQNAgent(object):
    """
    Q learning agent where Q function is estimated using a DQN

    @input feature_env, StateTransformer: Transforms the PDDL state rep into binary vec, among other things.
    @input rng, np.random.default_rng obj
    @input agent_name, str
    @input action_space, np.ndarray of PDDLLiterals: The grounded action literals the robot can perform.
    @input batch_size, int: size of SARSA data to train on.
    @input gamma, int: reward decay for Q learning
    @input fc2_size, int: size of hidden layer for DQN
    @input max_memory_sizem, int: How much SARSA data to store?
    @input pretrained, bool: If true, load model from model_path.
    @input overlays, OverlayList: Ovrlays to be used by the agent.
    @goal_condition, bool: Is the state conditioned on the meal goal? 


    """
    # Reference: https://towardsdatascience.com/deep-q-network-with-pytorch-146bfa939dfe
    def __init__(self, feature_env, rng, agent_name="dqn", action_space=None, batch_size=15,
                 gamma=.90, lr=.00025,exp_decay=.99, lr_decay=.9, exp_rate=.2, exp_min=.02,
                 fc2_size=32, max_memory_size=1000, pretrained=False, load_memories=True,
                 model_path="test", overlays=None, goal_condition=True):

        self.model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            "../../models")
        self.model_path = os.path.join(self.model_dir,
                                       model_path)
        # PDDL env for generating fixed length binary vector observation of a pddl state
        self.feature_env = feature_env
        self.feature_env.goal_condition = goal_condition
        self._rng = rng
        # self.state_space = len(feature_env.state_features) # size of state_space, int
        self.state_space = self.feature_env.n_features()
        print("[DQNAgent] state_space: ", self.state_space)
        # NOTE: Action space derivd from pddlgym is not generated in a consistent order
        # so we need to save the one we trained to a file.
        if action_space is None:
            self.action_space = load_action_space(os.path.join(self.model_path,
                                                           "ACTION_SPACE.npy"))
        else:
            self.action_space = action_space # list of actions
        self.n_actions = len(self.action_space)

        self.pretrained = pretrained
        self.load_memories = load_memories
        self.agent_name = agent_name

        # Learning params
        self.gamma = gamma
        self.lr = lr
        self.l1 = nn.SmoothL1Loss().to(device) # Also known as Huber loss

        # Replay memory params
        self.max_memory_size = max_memory_size
        # How many data points do we sample when re
        self.memory_sample_size = batch_size

        # Exploration params
        self.exp_max = 1.0
        self.exp_min = exp_min
        self.exp_rate = exp_rate
        self.exp_decay = exp_decay

        self.dqn = DQN(self.state_space, self.n_actions, fc2_size=fc2_size)
        self.dqn.to(device)

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr)
        # For LR decay
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=lr_decay)

        self._action_hist = []
        # Used for administering rewards
        self._checks_performed = []
        self._finished_oatmeal = False
        self._finished_toast = False
        self._finished_cereal = False
        # self._intermediate_reward_val = .33
        self._intermediate_reward_val = .5
        #self._intermediate_reward_dict = 0.0
        self._intermediate_reward_dict = {"oatmeal_done": self._intermediate_reward_val,
                                          "pastry_done": self._intermediate_reward_val,
                                          "cereal_done": self._intermediate_reward_val
                                          }
        self._goal_states = {"oatmeal_goal": ("oatmeal_done", "pastry_done",
                                              "oatmeal_in_bowl"),
                             "cereal_goal": ("pastry_done", "cereal_done")
                             }
        self._goal_reward = 1.0
        self._failiure_reward = -1.0
        self._done = False


        self.overlays = overlays

        self.STATE_MEM = torch.zeros(max_memory_size, self.state_space)
        self.ACTION_MEM = torch.zeros(max_memory_size, 1)
        # self.ACTION_MEM = torch.zeros(max_memory_size, self.n_actions)
        self.REWARD_MEM = torch.zeros(max_memory_size, 1)
        self.STATE2_MEM = torch.zeros(max_memory_size, self.state_space)
        self.DONE_MEM = torch.zeros(max_memory_size, 1)
        self.sample_weight = torch.zeros(max_memory_size, 1)
        self._new_sample_weight = .5


        self.ending_position = 0
        self.num_in_queue = 0

        if self.pretrained:
            print("[DQNAgent] Loading model from {}".format(self.model_path))
            # self.dqn.load_state_dict(torch.load(self.pretrained))
            self.dqn.load_state_dict(torch.load(os.path.join(self.model_path,"DQN.pt"),
                                                map_location=torch.device(device)))
            self.dqn.eval()
            if self.load_memories:


                STATE_MEM = torch.load(os.path.join(self.model_path, "STATE_MEM.pt"))
                ACTION_MEM = torch.load(os.path.join(self.model_path, "ACTION_MEM.pt"))
                REWARD_MEM = torch.load(os.path.join(self.model_path, "REWARD_MEM.pt"))
                STATE2_MEM = torch.load(os.path.join(self.model_path, "STATE2_MEM.pt"))
                DONE_MEM = torch.load(os.path.join(self.model_path, "DONE_MEM.pt"))
                shape = STATE_MEM.shape

                self.STATE_MEM[:shape[0], :shape[1]]  = STATE_MEM
                self.STATE2_MEM[:shape[0], :shape[1]]  = STATE2_MEM
                self.ACTION_MEM[:shape[0], :]  = ACTION_MEM
                self.REWARD_MEM[:shape[0], :]  = REWARD_MEM
                self.DONE_MEM[:shape[0], :]  = DONE_MEM
                try:
                        with open(os.path.join(self.model_path, "ending_position.pkl"), "rb") as f:
                                self.ending_position = pickle.load(f)
                        with open(os.path.join(self.model_path, "num_in_queue.pkl"), "rb") as f:
                                self.num_in_queue = pickle.load(f)
                except FileNotFoundError:
                        self.ending_position = 0
                        self.num_in_queue = 0
            else:
                self.ending_position = 0
                self.num_in_queue = 0

        else:
            print("[DQNAgent] Not loading weights from memory! Starting from scratch...")

        # self.sarsa_dataset = torch.utils.data.ConcatDataset((self.STATE_MEM, self.ACTION_MEM, self.REWARD_MEM,
        #                                                     self.STATE2_MEM, self.DONE_MEM))

    def _data_to_tensors(self, state, action, reward, state2, done):
        """Transform training data to tensor representation"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state2 = torch.from_numpy(state2).float().unsqueeze(0).to(device)
        action = torch.tensor([self.action_space.index(action)]).unsqueeze(0)
        reward = torch.tensor([reward]).unsqueeze(0)
        done = torch.tensor([int(done)]).unsqueeze(0)

        return state, action, reward, state2, done

    def _perform_check_actions(self, possible_actions):
        """
        In order to determine if meals are complete in PDDL domain, we
        define a number of "check" actions (e.g. check_plain_oatmeal) to see
        if the world state contains the correct combo of ingredients/to satisfy the
        meal. Rather than have this be a part of of the agent's policy explicitly, we
        will simply have it check after each action if the meal is done.
        """
        # checks = [a for a in possible_actions if a.predicate.name in CHECK_TEMPLATE_DICT]
        for a in possible_actions:
            print("[_perform_check_actions] trying: {}".format(a))
            # if a.predicate.name in CHECK_TEMPLATE_DICT:
            if a.predicate.name in CHECK_TEMPLATE_DICT:
                print("[_perform_check_actions] performing: {}".format(a))
                return a

        return None

    def _get_rewards(self, state):
        """
        Receive intermediate rewards for completing a single course
        """
        state_vars = self.feature_env.eval_pddl_vars(state)
        ret_reward = 0
        for k in self._intermediate_reward_dict:
            if state_vars[k]:
                ret_reward += self._intermediate_reward_dict[k]
                if self._intermediate_reward_dict[k] > 0:
                    self._intermediate_reward_dict[k] = 0

        for k in self._goal_states:
            if all(state_vars[v] for v in self._goal_states[k]):
                ret_reward = self._goal_reward
                self._done = True

        # print("[check_rewards] ", completed_dict)
        # if state_vars['oatmeal_done']:
        #     import pdb
        #     pdb.set_trace()

        return ret_reward, self._done


    def check_errors(self, state):
        """ Check if any of the predifined error state occured """
        ret = False
        features = self.feature_env.eval_pddl_vars(state)
        for k in features:
            if "error" in k:
                ret = features[k]
        return ret


    def check_rewards(self, env, possible_actions, goal=None):
        # check_action = self._perform_check_actions(possible_actions)
        # if not check_action == None:
        #     state,_,_,_ = env.step(check_action)

        # reward, done = self._get_rewards(state)
        # check_actions = [a for a in env.action_space.all_ground_literals(state) \
        state = env.get_state()

        if not goal is None:

                check_actions = [a for g in goal for a in possible_actions \
                                # if a.predicate.name in CHECK_TEMPLATE_DICT and  \
                                if g in a.predicate.name \
                                #and not a.predicate.name in self._checks_performed
                                ]
        else:
                check_actions = [a for a in possible_actions \
                                # if a.predicate.name in CHECK_TEMPLATE_DICT and  \
                                if "check" in a.predicate.name
                                #and not a.predicate.name in self._checks_performed
                                ]


        print("[check_rewards] check actions: ", check_actions)
        reward = 0.0
        done = False
        for a in check_actions:
            print("[check_rewards] performing check: ", a)
            self._checks_performed.append(a)
            state,r,done,_ = env.step(a)
            if done:
                reward = r
                break
            else:
                reward += self._intermediate_reward_val

        return state, reward, done

    def reset(self):
        self._done = False
        self._action_hist = []
        self._finished_oatmeal = False
        self._finished_cereal = False
        self._finished_toast = False
        self._intermediate_reward_dict = {"oatmeal_done": self._intermediate_reward_val,
                                          "pastry_done": self._intermediate_reward_val,
                                          "cereal_done": self._intermediate_reward_val
                                          }

    def remember(self, state, action, reward, state2, done, uniform_weight=True):
        """Store the experiences in a buffer to use later"""

        # Transform data into tensor representations
        state, action, reward, state2, done = self._data_to_tensors(state, action,
                                                                     reward, state2, done)
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)
        self.sample_weight[self.ending_position] = self._new_sample_weight * (1 - uniform_weight) + uniform_weight


    def batch_experiences(self, sample_type="uniform"):
        """Randomly sample 'batch size' experiences"""
        if sample_type == "uniform":
            # idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
            idx = self._rng.choice(range(self.num_in_queue), size=self.memory_sample_size)
        elif sample_type == "weighted":
            # sample a contiguous block of memory at random idx, but wrap around if necessary
            # idx = np.random.randint(0, self.num_in_queue)
            idx = self._rng(0, self.num_in_queue)
            idx = np.arange(idx, idx + self.memory_sample_size) % self.num_in_queue

        
        try:
            idx = idx[:self.memory_sample_size]
            STATE = self.STATE_MEM[idx]
            ACTION = self.ACTION_MEM[idx]
            REWARD = self.REWARD_MEM[idx]
            STATE2 = self.STATE2_MEM[idx]
            DONE = self.DONE_MEM[idx]
        except IndexError:
            import pdb; pdb.set_trace()
        return STATE, ACTION, REWARD, STATE2, DONE


    def get_transformed_state(self, state):
        """
        Transfroms PDDL state to binary vec represenation
        @input stat, Frozenset of PDDLLiterals: the current state

        @return np.darray
        """
        self.feature_env.set_state(state)
        return self.feature_env.transform(state)


    def q_vals(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        return self.dqn(state.to(device)).detach().cpu().numpy()

    def update_action_hist(self, action):
        self._action_hist.append(action)

    def _get_policy_dist(self, state, feasible_actions=None):
        """
        Choose an action based on the current state and eps-greedy policy
        @state: binary np.array representation of the current state
        @feasible_actions: optional list of  actions [a_1, ... , a_k] that the
                            agent can perform in current state

        @returns action, PDDL literal based on the max Q value.
        """

        # state = torch.from_numpy(state.astype(np.float32).reshape(1,-1))
        ACTION_MASK_VAL = -1e4
        s_bin = self.get_transformed_state(state)

        # if np.random.uniform() < self.exp_rate:
        if self._rng.uniform() < self.exp_rate:
            # rand_idx = np.random.randint(len(self.action_space ))
            rand_idx = self._rng.intergers(len(self.action_space ))
            action_prob = np.zeros(len(self.action_space))
            action_prob[rand_idx] = 1.0

            # print("[act] Choosing planned action")

        else:
            # Transform numpy bin to torch tensor
            s_bin = torch.from_numpy(s_bin).float().unsqueeze(0).to(device)
            # Choose an action from the model
            q_vals = self.dqn(s_bin.to(device))

            # If feasable actions are provided then zero out all actions in q_val vector
            # save for the feasable actions
            if not feasible_actions is None:
                print("Action hist: ", set(self._action_hist))
                feasible_actions = [a for a in feasible_actions if \
                                    not a in self._action_hist]
                # print("feasible_actions: ", feasible_actions)

                action_mask  = torch.zeros(size=q_vals.shape).to(device)
                # pdb.set_trace()
                # all_a_idx = np.where(self.action_space[:, None] == feasible_actions[:])[0]
                all_a_idx = np.where(np.in1d(self.action_space, feasible_actions))# [0]
                # print("possible actions no.: {} q val no.: {}".format(len(feasible_actions),
                #                                                       q_vals.shape))
                action_mask[:, all_a_idx] = 1
                q_vals *= action_mask
                q_vals[q_vals == 0.0] = ACTION_MASK_VAL

            # Convert from tensor to numpy
            q_vals = q_vals.cpu().detach().numpy()

            # Normalize
            action_prob = softmax(q_vals, axis=1)

        return action_prob


    def act(self, state, feasible_actions=None, ret_original_pred=False):
        import pdb
        # Action not feasible if not in action space. This can
        # happen if an unusual PDDL action grounding is used.
        # feasible_actions = [a for a in feasible_actions if a in self.action_space]
        feasible_actions = [a for a in feasible_actions if a in self.action_space]
        try:
                action_prob = self._get_policy_dist(state, feasible_actions).squeeze()
                orig_action_prob = deepcopy(action_prob)
        except AttributeError as e:
                print("[act] ", e)
                print("[act] Error!! Cannot get action probabilities!")
                import pdb; pdb.set_trace()

        if not self.overlays is None:
            print("[act] Applying overlays!")
            print(self.overlays)
            action_prob, _ = self.apply_overlays(state,
                                                 feasible_actions,
                                                 action_prob)

        # Normalize again
        action_prob = softmax(action_prob)

        # Get the best action
        a_idx = np.argmax(action_prob)
        action = self.action_space[a_idx.item()]
        # action = np.random.choice(list(self.action_space),
        #                           p=np.squeeze(action_prob))
        # self.softmax_action_selection(q_vals, self.action_space)

        # print("[act] Choosing Q action!")
        sorted_actions = zip(self.action_space,
                             action_prob[:])
        sorted_actions = sorted(sorted_actions, reverse=True, key=lambda x: x[1])
        print("[act] TOP 3 actions: ")
        for a, p in sorted_actions[:10]:
                print("\t{}: {}".format(a, np.round(p, 7)))

        if ret_original_pred:
            # assert not self.overlays is None
            orig_a_idx = np.argmax(orig_action_prob)
            orig_action = self.action_space[orig_a_idx]
            ret = action, orig_action
        else:
            ret = action
        # action = self.softmax_action_selection(q_vals, self.action_space)

        # self._action_hist.append(action)

        return ret


    def weighted_action_selection(self, action_probs):
        sorted_actions = sorted(zip(self.action_space, action_probs),
                                key=lambda x: x[1], reverse=True)
        # rand_prob = np.random.uniform()
        rand_prob = self._rng.uniform()
        for a in sorted_actions[:3]:
            print("[softmax] Action: {}, prob: {}".format(a[0], a[1]))

        for a, prob in sorted_actions:
            rand_prob -= prob
            if rand_prob <= 0:
                return a


    def experience_replay(self):
        """
        Samples SARSA data randomly from memory and updates Q values simultaneously.
        """
        if self.memory_sample_size > self.num_in_queue:
            print("[experience_replay] queue too small!: {} / {}".format(self.num_in_queue,
                                                                         self.memory_sample_size) )
            return

        # Sample a batch of experiences
        STATE, ACTION, REWARD, STATE2, DONE = self.batch_experiences("uniform")
        # STATE, ACTION, REWARD, STATE2, DONE = self.batch_experiences("sequential")
        STATE = STATE.to(device)
        ACTION = ACTION.to(device)
        REWARD = REWARD.to(device)
        STATE2 = STATE2.to(device)
        DONE = DONE.to(device)

        self.optimizer.zero_grad()
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
        target = REWARD + torch.mul((self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1)),
                                    1 - DONE)
        # import pdb; pdb.set_trace()
        # print("[experience_replay] pred shape:", self.dqn(STATE).shape)
        # print("[experience_replay] action space size:", self.n_actions)
        current = self.dqn(STATE).gather(1, ACTION.long())

        loss = self.l1(current, target)
        loss.backward() # Compute gradients
        self.optimizer.step() # Backpropagate error
        # print("[experience_replay] Current: {} Target: {}".format(current, target) )
        print("[experience_replay] ", loss)

        self.exp_rate *= self.exp_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exp_rate = max(self.exp_rate, self.exp_min)
        return loss



    def retrain(self, iterations=20):
        """
        Train the model from data stored to files.
        """
        print(os.path.join(self.model_path, "STATE_MEM.pt" ))
        self.STATE_MEM = torch.load(os.path.join(self.model_path, "STATE_MEM.pt" ))
        self.ACTION_MEM = torch.load(os.path.join(self.model_path, "ACTION_MEM.pt"))
        self.REWARD_MEM = torch.load(os.path.join(self.model_path, "REWARD_MEM.pt"))
        self.STATE2_MEM = torch.load(os.path.join(self.model_path, "STATE2_MEM.pt"))
        self.DONE_MEM = torch.load(os.path.join(self.model_path, "DONE_MEM.pt"))

        print(self.STATE2_MEM.shape)
        self.num_in_queue = self.memory_sample_size

        for i in range(iterations):
            self.experience_replay()



    def add_overlay(self, overlay, ov_type="ros"):
        """Add overlay to overlay list"""

        # import pdb; pdb.set_trace()
        if self.overlays is None:
            print("About to add overlay")
            if ov_type == "ros":
                self.overlays = RosOverlayList([overlay])
            else:
                self.overlays = PrologOverlayList([overlay])
            print("Added overlay")
        else:
            self.overlays.append(overlay)


    def remove_overlay(self, overlay_key):
        print("[remove_overlay] overlay_name: ", overlay_key)
        new_overlay_list = []
        if overlay_key == "last":
            # Deactivate the last overlay in the last.
            new_overlay_list = self.overlays[:-1]
            # self.overlays[-1].is_active = False
            print("[remove_overlay] Deactivating overlay: ", self.overlays[-1])
        else:
            for overlay in self.overlays:
                if not overlay.name == overlay_key:
                    new_overlay_list.append(overlay)
                    # overlay.is_active = False

        self.overlays = RosOverlayList(new_overlay_list)



    def apply_overlays(self, state, possible_actions, action_probs):
        """
        Implements overlay transformations on a q_val vector of the form Tq = Aq + b
        where q is the prob vector, A is a matrix of coefficients that amplifies/diminishes
        a learned action prob for a given action and b increases the prob in the event the learned
        value is zero.
        """

        # Add 1 at the end of the action_probs vector
        action_probs = np.append(action_probs, 1)
        overlay_mat_list = [np.identity(len(action_probs))]
        overlay_mat_dict = {
                # "permit":[np.zeros((len(action_probs), len(action_probs)))],
                "permit":[np.identity(len(action_probs))],
                "prohibit":[np.identity(len(action_probs))],
                "transfer":[np.zeros((len(action_probs), len(action_probs)))]
        }


        # state dict of the form {predicate_name: bool} where bool denotes
        # if the predicate is true in the current state
        state_dict = self.feature_env.eval_pddl_vars(state)

        possible_actions = [a for a in possible_actions if not a in self._action_hist]
        # Evaluating overlays updates the "res" param of each individual overlay
        self.overlays.eval(state_dict, possible_actions)

        for o in self.overlays:
            # overlay.res is a list of dicts of the form [{A_in: a1, A_out: a2}] where
            # A_in corresponds to the action to be changed by the overlay and A_out corresponds
            # to the alternative to A_out
            # Sometimes the overlay is applied to the same action twice, so we only want to
            # apply it once
            if o.overlay_type == "permit" and len(o.res) > 0:
                overlay_mat = np.zeros((len(action_probs), len(action_probs)))
            else:
                overlay_mat = np.identity(len(action_probs))

            overlayed_actions = []
            # import pdb
            # pdb.set_trace()
            print("[apply_overlays] Overlay {} type: {}".format(o.name, o.overlay_type))
            for res in o.res:
                # If true than the overlay not applicable in current state
                if len(res) == 0:
                    continue
                # Not all rules are aimed at changing enforcing a particular
                # alternative actions e.g. a rule like "if <state> then not <action>"
                elif len(res) == 1:
                    _, action = res.popitem()
                    assert not action in overlayed_actions
                    a_out_idx = self.action_space.index(action)
                    a_in_idx = a_out_idx
                    overlayed_actions.append(action)


                elif len(res) == 2:
                    a_in_idx = self.action_space.index(res["A_in"])
                    a_out_idx = self.action_space.index(res["A_out"])
                    # this zeroes out the weight of the action we are transferring to
                    # so that the prob density is only the prob of thing we are transferring from.
                    overlay_mat[a_out_idx, a_out_idx] = 0
                    overlay_mat[a_in_idx, a_in_idx] = 0

                # If there is nonzero padding then the subsequent matrix multiplication
                # Adds the padding value to corresponding q value.
                # print(res)
                # import pdb; pdb.set_trace()
                overlay_mat[a_out_idx, a_in_idx] = o.coeff
                overlay_mat[a_out_idx, -1] += o.padding

            overlay_mat_list.append(overlay_mat)
            overlay_mat_dict[o.overlay_type].append(overlay_mat)

        permit_M   = np.identity(len(action_probs))
        forbid_M   = np.identity(len(action_probs))
        transfer_M = np.identity(len(action_probs))

        if len(overlay_mat_dict["permit"]) > 1:
                # permit_M = np.logical_or.reduce(overlay_mat_dict["permit"])
                permit_M = np.logical_and.reduce(overlay_mat_dict["permit"])
        if len(overlay_mat_dict["prohibit"]) > 1:
                forbid_M = np.logical_and.reduce(overlay_mat_dict["prohibit"])
        if len(overlay_mat_dict["transfer"]) > 1:
                transfer_M = np.logical_or.reduce(overlay_mat_dict["transfer"])

        overlay_mat = np.linalg.multi_dot([permit_M, forbid_M, transfer_M])
        transformed_probs = overlay_mat.dot(action_probs)

        # print("action probs: ", action_probs)
        # print("transformed probs: ", transformed_probs)

        for act in possible_actions:
            try:
                a_idx = self.action_space.index(act)
                # if not action_probs[a_idx] == transformed_probs[a_idx]:
                # if action_probs[a_idx] == transformed_probs[a_idx]:
                if not transformed_probs[a_idx] == action_probs[a_idx]:
                        print("[apply_overlays] action: {}\n\t old q: {} new q: {}"
                              .format(act, np.round(action_probs[a_idx], 4),
                                      np.round(transformed_probs[a_idx], 4))
                              )
            # print error

            except ValueError as e:
                print("Error: ", e)


        # return transformed q value without the added padding
        return transformed_probs[:-1], action_probs[:-1]




    def add_overlay_from_dict(self, rule_dict):
        """
        @input: rule_dict, {"name":str, "expr":str} a rule to be transformed into an overlay
        and added to the list of active overlays.
        """
        assert "name" in rule_dict
        assert "expr" in rule_dict

        ov = PrologOverlay(name=rule_dict["name"],
                           in_expr=rule_dict["expr"])

        self.overlays.append(ov)
        print(self.overlays)




class ShieldAgent(DQNAgent):
    """
    DQNAgent that can utilize RL shields.

    """
    def __init__(self, shield_future_state=True, shield_list=None, **kwargs):
        """
        @input shield_future_state, bool: are we evaluating the shield on th current state or the next state?
                                          For HRI23 this always True
        @input shield_list, ShieldList: The shields to be used by the agent
        """
        super(ShieldAgent, self).__init__(**kwargs)
        self._shield_list = shield_list
        self._shield_actions = []
        self._action_space_str = [str(a) for a in self.action_space]
        self._shield_future_state = shield_future_state
        self.agent_name = "shield"


    def add_shields(self, shield_list):
        self._shield_list = shield_list


    def _apply_forbid_shield(self, env, action_prob, feasible_actions):
        """
        Forbid shields zero out the prob of actions that lead to bad states.
        @input env, pddlgym.Env: The cooking RL environment.
        @input action_prob, ndarray: pi(A|s)
        @input action_space, np.ndarray of PDDLLiterals: The grounded

        """
        for shield in self._shield_list:
            if shield._shield_type == "forbid":
                curr_state = env.get_state()
                curr_state_dict = self.feature_env.eval_pddl_vars(curr_state)
                for action in feasible_actions:
                    # if "cocopuffs" in str(action):
                    state_next,_,_,_ = env.sample_transition(action)
                    state_next_dict = self.feature_env.eval_pddl_vars(state_next)
                    state_diff_dict = dict(set(state_next_dict.items()) - set(curr_state_dict.items()))
                    # import pdb; pdb.set_trace()
                    # a_idx, new_prob = shield.shield(state_next_dict, action,
                    #                                 self.action_space)
                    a_idx, new_prob = shield.shield(state_diff_dict, action,
                                                    self.action_space)
                    action_prob[:, a_idx] *=new_prob

        return action_prob


    def act(self, env, state, feasible_actions):

        # Action not feasible if not in action space. This can
        # happen if an unusual PDDL action grounding is used.
        hypo_env = deepcopy(env)
        feasible_actions = [a for a in feasible_actions if a in self.action_space]
        action_prob = self._get_policy_dist(state, feasible_actions)
        # action_prob = self.softmax_temp(action_prob)
        # action_prob = softmax(action_prob)
        action_prob = self._apply_forbid_shield(hypo_env, action_prob, feasible_actions)
        # action = self.weighted_action_selection(action_prob)
        action = self.action_space[np.argmax(action_prob)]
        print("[act] Curr best action: ", action)

        if self._shield_actions:
            print("[act] Getting action from shield")
            action = self._shield_actions.pop(0)

        for shield in self._shield_list:
            print("[act] Attempting to apply shield: {}".format(shield.name))
            if shield._shield_type == "forbid":
                continue

            if self._shield_future_state:
                state_next,_,_,_ = hypo_env.sample_transition(action)
                state_next_dict = self.feature_env.eval_pddl_vars(state_next)
                action, action_horizon = shield.shield(state_next_dict, action)
            else:
                state_dict = self.feature_env.eval_pddl_vars(state)
                action, action_horizon = shield.shield(state_dict, action)
            # print("[act] horizon: {} action: {}".format(action_horizon, action))
            self._shield_actions.extend(action_horizon)
            print("[act] New shield action: ", action)

            # Return the action as a PDDL literal
            # a_idx = self._action_space_str.index(action_str)
            # action = self.action_space[a_idx]
        print("Returning action: ", action)
        return action


if __name__ == "__main__":
    pass

    # model = DQN(env, 30).to(device)
    # train_model_monte_carlo()