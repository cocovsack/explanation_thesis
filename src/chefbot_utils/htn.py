#!/usr/bin/env python

# coding: utf-8


"""
Tools for task representation.

These make no sense if states are not discrete (although they may be
represented as continuous vectors). To enforce this states are required
to be hashable.

The first part of this module implements Graph based task representations
that encode valid transitions between states, and conjugate task graphs
as introduced by [Hayes2016]_. It also provides the algorithmic basis to
extract hierarchical task representations from such conjugate graphs (see
[Hayes2016])_.

The second part of the module implements classes to represent hierarchical
task models based on the *simultaneous*, *alternative*, and *parallel*
combination.

.. [Hayes2016] Hayes, Bradley and Scassellati, Brian *Autonomously constructing
   hierarchical task networks for planning and human-robot collaboration*, IEEE
   International Conference on Robotics and Automation (ICRA 2016)
"""

import os
import pdb
import numpy as np
import copy
from itertools import permutations, product
from pandas.core.common import flatten
try:

    import rospkg
    from chefbot_utils.learning_util import (DATA_PATH, ROLL, MUFFIN, JELLYPASTRY, PIE, EGG, CEREAL,
                               MILK, WATER, PLAINOATMEAL, FRUITYOATMEAL, CHOCOLATEOATMEAL,
                               PBBANANAOATMEAL, FRUITYCHOCOLATEOATMEAL, PBCHOCOLATEOATMEAL)

    from chefbot_utils.pddl_util import load_environment, PDDL_DIR


except ModuleNotFoundError:
    from learning_util import (DATA_PATH, ROLL, MUFFIN, JELLYPASTRY, PIE, EGG, CEREAL,
                               MILK, WATER, PLAINOATMEAL, FRUITYOATMEAL, CHOCOLATEOATMEAL,
                               PBBANANAOATMEAL, FRUITYCHOCOLATEOATMEAL, PBCHOCOLATEOATMEAL)


    from pddl_util import load_environment, PDDL_DIR



def int_generator():
    i = -1
    while True:
        i += 1
        yield i


class BaseCombination(object):

    kind = 'Undefined'

    def __init__(self, idx, parent, name='unnamed', highlighted=False):
        self._name = name
        self.highlighted = highlighted
        self._idx = idx
        self.parent = parent

    @property
    def name(self):
        return self._name
    @property
    def idx(self):
        return self._idx

    def _set_idx(self, id_generator):
        self._idx = next(id_generator)

    def _meta_dictionary(self, parent_id, id_generator):
        attr = []
        if self.highlighted:
            attr.append('highlighted')
        return {'name': self.name,
                # 'id': next(id_generator),
                'id': self.idx,
                'parent': parent_id,
                'combination': self.kind,
                'attributes': attr,
                }


class Combination(BaseCombination):

    kind = 'Undefined'

    def __init__(self, children, idx=None, parent=None, name='unnamed', highlighted=False,
                 probabilities=None):
        super(Combination, self).__init__(idx, parent, name, highlighted)
        self.children = children  # Actions or combinations
        self.proba = probabilities

    def set_idx(self, id_generator):
        self._set_idx(id_generator)
        for c in self.children:
            c.set_idx(id_generator)


    def as_dictionary(self, parent_id, id_generator):
        d = self._meta_dictionary(parent_id, id_generator)
        d['children'] = [
            c.as_dictionary(d['id'], id_generator)
            for c in self.children
        ]
        return d

    def _deep_copy_children(self, rename_format='{}'):
        return [c.deep_copy(rename_format=rename_format)
                for c in self.children]


class LeafCombination(BaseCombination):

    kind = None

    def __init__(self, action, idx=None, parent=None, highlighted=False):
        super(LeafCombination, self).__init__(idx, parent, action.name, highlighted)
        self.action = action
        self.visited = False

    @property
    def name(self):
        return self.action.name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def children(self):
        raise ValueError('A leaf does not have children.')

    def visit(self):
        self.visited = True

    def set_idx(self, id_generator):
        self._set_idx(id_generator)

    def as_dictionary(self, parent, id_generator):
        return self._meta_dictionary(parent, id_generator)

    def deep_copy(self, rename_format='{}'):
        return LeafCombination(self.action.copy(rename_format=rename_format),
                               highlighted=self.highlighted)


class SequentialCombination(Combination):

    kind = 'Sequential'

    def __init__(self, children, **xargs):
        super(SequentialCombination, self).__init__(children, **xargs)

    def deep_copy(self, rename_format='{}'):
        return SequentialCombination(
            self._deep_copy_children(rename_format=rename_format),
            probabilities=self.proba,
            name=rename_format.format(self.name),
            highlighted=self.highlighted)


# TODO: Unless we want to validate the proba inputs
# in __init__, I'm open to an alternative for
# how to do this without using properties
class AlternativeCombination(Combination):

    kind = 'Alternative'

    def __init__(self, children, **xargs):
        super(AlternativeCombination, self).__init__(children, **xargs)
        self.proba = None

    @property
    def proba(self):
        return self._proba

    @proba.setter
    def proba(self, probabilities):
        if probabilities is None:
            num_children = len(self.children)
            prob = float(1) / num_children
            self._proba = [prob] * num_children
        elif np.min(probabilities) < 0:
            raise ValueError("At least one prob value is < 0.")
        elif np.max(probabilities) > 1:
            raise ValueError("At least one prob value is > 1.")
        elif not (np.isclose(np.sum(probabilities), 1)):
            raise ValueError("Probs should sum to 1.")
        else:
            self._proba = probabilities

    def deep_copy(self, rename_format='{}'):
        return AlternativeCombination(
            self._deep_copy_children(rename_format=rename_format),
            probabilities=self.proba,
            name=rename_format.format(self.name),
            highlighted=self.highlighted)


class ParallelCombination(Combination):

    kind = 'Parallel'

    def __init__(self, children, **xargs):
        super(ParallelCombination, self).__init__(children, **xargs)

    def deep_copy(self, rename_format='{}'):
        return ParallelCombination(
            self._deep_copy_children(rename_format=rename_format),
            probabilities=self.proba,
            name=rename_format.format(self.name),
            highlighted=self.highlighted)

    def to_alternative(self):
        sequences = [
            SequentialCombination(
                [c.deep_copy('{{}} order-{}'.format(i)) for c in p],
                name='{} order-{}'.format(self.name, i))
            for i, p in enumerate(permutations(self.children))
        ]
        return AlternativeCombination(sequences, name=self.name,
                                      highlighted=self.highlighted)


class HierarchicalTask(object):

    """Tree representing a hierarchy of tasks with leaves are actions."""

    def __init__(self,root, data):
        """
        @input: root, Combination. The root of node of the HTN
        @input: main, str: the main course (oatmeal or cereal currently) of the cooking task represented by the HTN.
        @input: side, str: the side course of the cooking task (pastries currently) represented by the HTN.

        Represents a cooking task a composition of combination an transforms into a flat sequence(s) of actions.
        """
        self.root = root
        self._data = data
        self.name = "{}_{}".format(data["main"]["meal"],
                                   data["side"]["meal"])
        # List of lists where each sublist is a valid flattening
        # of the leaves of the HTN
        self._action_seqs = self.flatten_tree(root)
        self._unrepeatable_actions = ["pour", "gather", "serveoatmeal"]
        # Tracks current leaf sequence of interest
        self._seq_idx = 0 # if not rand_seq else np.random.randint(len(self._action_seqs))
        self._action_idx = 0 # tracks current action in sequence
        # Number of steps in one of the valid leaf sequences of the tree
        # NOTE: All sequences should be of equal length which is why below.
        # should work.
        self.n_steps = len(self._action_seqs[self._seq_idx])
        # self.set_idx()

    @property
    def main(self):
        return self._data["main"]["meal"]
    @property
    def side(self):
        return self._data["side"]["meal"]

    def get_k_previous_actions(self, action_space, k=1):
        """
        Returns the k actions preceding the current action (self._action_idx) at the current
        Meal permutation (self._seq_idx)
        """
        seq = self._action_seqs[self._seq_idx]
        action_strs = seq[self._action_idx - k: self._action_idx]


        return [a for a_str in action_strs for a in action_space if str(a) == a_str]

    def get_next_action(self, action_space, performed):
        """
        Gets the next ground truth action to perform based on the HTN structure.
        However, we also take into account that some actions may have already (incorrectly)
        been performed, and cannot be performed again.

        @input action_space, np.ndarray of PDDLLiterals: The grounded action literals the robot can perform.
        @input performed, list of PDDLLiterals: Actions already performed by the agent

        Returns PDDLLiteral: The current action to perform.
        """
        # The current HTN leaf sequence
        seq = self._action_seqs[self._seq_idx]
        action_found = False
        to_add_to_trace = []

        if self._action_idx == len(seq):
            print("[get_next_action] No more actions in sequence!")
            return None, to_add_to_trace
        # print("[HTN] Number of sequence: ", len(self._action_seqs))
        # print("[HTN] sequence length: ", self.n_steps)
        # print("[HTN] Current  seq_idx: {}".format(self._seq_idx))
        # print("[HTN] Current action idx: {}".format(self._action_idx))
        # print("[get_next_action] performed: ", performed)
        # Which actions should not be performed?
        avoid_repeats = [a for avoid in self._unrepeatable_actions for a in performed if avoid in a]
        # print("[get_next_action] avoid_repeats: ", avoid_repeats)
        while not action_found:
          try:
              ret = seq[self._action_idx]
              self._action_idx += 1
              # action space is a list of PDDLLiterals so we
              # need to check against each literals str rep
              if not ret in avoid_repeats:
                  ret = [a for a in action_space if str(a) == ret][0]
                  action_found = True
              else:
                  to_add_to_trace.append(ret)
                  print("[get_next_action] skipping: {} alreay performed!".format(ret))
          except IndexError:
              print("[HTN] idx: {} out of range".format(self._action_idx))
              # print("[HTN] current sequence: ", seq)
              print("[HTN] ret: ", ret)
              ret = None
              action_found = True

        return ret, to_add_to_trace


    def is_empty(self):
        return self.root is None

    def reset(self):
        """
        Sets the seq_idx and and action_idx back to 0
        """
        self._seq_idx = 0
        self._action_idx = 0

    def set_idx(self,):
        if self.root is not None:
            self.root.set_idx(int_generator())

    def as_dictionary(self, name=None):
        return {
            'name': 'Hierarchical task tree' if name is None else name,
            'nodes': None if self.is_empty() else self.root.as_dictionary(
                None, int_generator()),
        }



    def flatten_tree(self, node):
        """
        @input node, a Combination object. The root node of the HTN to flatten
        returns: list of list<str>: [["a1, "a4", ..."an"], where each sublist is
                                     ["a1", "a4"... "an"], ...]  a valid permutation of the HTN

        Roughly the number of permutations is equivalent to:
        !P1.n_children * !P2.n_children *... !Pn.n_children where Pk is a parallel node
        and n_children is the number of children it has.
         """

        # import pdb; pdb.set_trace()
        # Base case. We've hit a leaf node.
        if node.kind == None:
            return node.name
        elif node.kind is 'Sequential':
            # If a node is sequential its children must be executed
            # in sequence.
            ret_container = [self.flatten_tree(c) for c in node.children]
            # If ret_container is a list of list, then elements of each subsequence
            # are a part of a single permutation of the tree, so need to be concatenated
            # together
            if isinstance(ret_container[0], list):
                ret_container = [list(flatten(i)) for i in product(*ret_container)]
            else: # Otherwise its a single complete sequence.
                ret_container = [ret_container]
            # else:
            #     print("ret_container: ", ret_container)
            #     ret_container = list(flatten(ret_container))
        elif node.kind is 'Parallel':
            # If a node is parallel, all permutations of its children are permitted.
            ret_container = []
            for perm in permutations(node.children):
                perm_ret = [self.flatten_tree(c) for c in perm]
                ret_container.append(perm_ret)
        return ret_container


    def check_action_in_seq(self, root, candidate_action):
        """
        @input root, Combination: The root of an HTN.
        @input candidate_action, PDDLLiteral: Action to be performed by the robot.

        Return bool: is the current sequence of action + candidate action a valid
        action sequence according the HTN?
        """
        full_gt_seq = self._action_seqs[self._seq_idx]
        # Add the candidate action the to th current sequence of performed actions.
        possible_seq = full_gt_seq[:self._action_idx - 1] + [str(candidate_action)]
        # print("[check_action_in_seq] possible candidate_action: ", possible_seq)
        # Is this potential sequence in the tree?
        return self.check_seq_in_tree(root, possible_seq)

    def check_seq_in_tree(self, root, seq):
        """
        @input root, Combination: The root of an HTN.
        @input seq, ["a1", .."ak"]: A sequence of actions.

        Returns: bool indicating if seq is a valid sequence of
        actions according to the root of the HTN.
        """

        seq_len = len(seq)
        # print("[check_seq_in_tree] Checking from seq: ", self._seq_idx)
        for i in range(self._seq_idx, len( self._action_seqs )):
            # print("[check_seq_in_tree] checking ", i)
            gt_seq = self._action_seqs[i]
            # print(gt_seq)
            if gt_seq[:seq_len] == seq:
                print("[check_seq_in_tree] Found seq: ", i)
                # print(gt_seq)
                self._seq_idx = i
                return True
        # If the sequence is not found, then no subsequent sequence will be found until
        # the task is reset
        # self._seq_idx = len(self._action_seqs) - 1
        self._seq_idx = 0
        print("[check_seq_in_tree] No seq found")
        return False



COMBINATION_CLASSES = {'Sequential': SequentialCombination,
                       'Parallel': ParallelCombination,
                       'Alternative': AlternativeCombination,
                       }

class Action(object):
    """Base class for actions that provide a check method."""

    def __init__(self, name="unnamed-action", agent="human"):
        self.name = name
        self.agent = agent

    def __repr__(self):
        return "{}<{}>".format(self.__class__.__name__, self.name)

    def __str__(self):
        return self.name


class CookingHTNFactory(object):
    """
    Factory class for generating HTNs for the chefbot cooking task.
    Each HTN models a users preferences for a breakfast making task.
    """
    def __init__(self,  main, side, liquid, action_types, order):
        """
        @input: root, Combination. The root of node of the HTN
        @input: main, str: the main course (oatmeal or cereal currently) of the cooking task represented by the HTN.
        @input: side, str: the side course of the cooking task (pastries currently) represented by the HTN.
        @input liquid, str: the liquid base used in making oatmeal (MILK or WATER). NOTE: ignored if main is cereal
        @input action_types, {"main":"do"/"say", "say": "do"/"say"}, dictionary denoting desired interventions for each course
        @input order, str, either "main first" or "side first". The order of dishes in the meal.


        """

        self.main = main
        self.side = side
        self.liquid = liquid
        self.action_type_dict = action_types
        self.order=order

        assert side in [PIE, MUFFIN, JELLYPASTRY, ROLL, EGG]
        assert main in [PLAINOATMEAL, CHOCOLATEOATMEAL, FRUITYOATMEAL,
                        FRUITYCHOCOLATEOATMEAL, PBBANANAOATMEAL,
                        PBCHOCOLATEOATMEAL, CEREAL]
        assert liquid in [MILK, WATER]
        assert 'main' in action_types and 'side' in action_types

    def set_meal_type(self, **kwargs):
        self.main = kwargs["main"]
        self.side = kwargs["side"]
        self.liquid = kwargs["liquid"]
        self.action_type_dict = kwargs["action_types"]
        self.order = kwargs["order"]

        return self

    def _generate_side_combo(self):

        at = self.action_type_dict["side"]
        plan = ["gather({side}:pastry,do:action_type)",
                "putinmicrowave({side}:pastry,{at}:action_type)",
                "turnon(microwave:appliance,{at}:action_type)",
                "microwavepastry({side}:pastry,say:action_type)",
                "takeoutmicrowave({side}:pastry,{at}:action_type)",
                # "check{side}(side:meal,say:action_type)"
                ]

      # p = random.choice(pastries)
        leaves = [LeafCombination(Action(name=a.format(side=self.side,
                                                       at=at))) for a in plan]
        # import pdb; pdb.set_trace()
        return SequentialCombination(leaves, name="{}_{}".format(self.side, at))


    def _generate_cereal_combo(self, variation="ingredients first"):

        """
        @input variation, str: indicating which variation of the meal to generate.
        There are two options: 1) "ingredients first" which has the robot collects
                                   all relevant ingredients before using them
                               2) "use immediately" uses ingredient immediately after
                                  retrieving them.

        Generates a subtree representing making cereal
        """
        at = self.action_type_dict["main"]

        steps = {
            "get_cocopuffs": "gather(cocopuffs:ingredient,do:action_type)",
            "get_milk": "gather(milk:liquid,do:action_type)",
            "get_bowl": "gather(bowl:container,do:action_type)",
            # "get_mixingspoon": "gather(mixingspoon:tool,do:action_type)",
            "get_eatingspoon": "gather(eatingspoon:tool,do:action_type)",
            "pour_cocopuffs": "pour(cocopuffs:ingredient,bowl:container,{at}:action_type)",
            "pour_milk_bowl": "pour(milk:liquid,bowl:container,{at}:action_type)",
        }
        # instantiate the template with specified action type
        steps = {k:Action(name=v.format(at=at)) for k,v in steps.items()}

        if variation == "ingredients first":
            get_ingredients = ParallelCombination([LeafCombination(a) \
                                                   for k,a in steps.items() if "get" in k])
            pour_ingredients = ParallelCombination([LeafCombination(a) \
                                                    for k,a in steps.items() if "pour" in k])
            ret_tree = SequentialCombination([get_ingredients, pour_ingredients],
                                             name="cereal_{}".format(at))

        elif variation == "use immediately":
            part_1 = SequentialCombination([LeafCombination(steps[k]) \
                                                            for k in ["get_cocopuffs",
                                                                      "pour_cocopuffs"]])
            part_2 = SequentialCombination([LeafCombination(steps[k]) \
                                                            for k in ["get_milk",
                                                                      "pour_milk_bowl"]])
            # parallel_part = ParallelCombination([part_1, part_2])
            parallel_part_1 = ParallelCombination([LeafCombination(steps["get_bowl"]),
                                                   LeafCombination(steps["get_eatingspoon"]),])
            parallel_part_2 = ParallelCombination([part_1, part_2])
            ret_tree = SequentialCombination([
                parallel_part_1,
                parallel_part_2])

        return ret_tree

    def _generate_oatmeal_combo(self, variation="ingredients first"):
        """
        @input variation, str: indicating which variation of the meal to generate.
        There are two options: 1) "ingredients first" which has the robot collects
                                   all relevant ingredients before using them
                               2) "use immediately" uses ingredient immediately after
                                  retrieving them.
                               3) "Toppings last 1" get plain oatmeal ingredients first
                                  then use them, then get topping ingredients, then use them
                               4) "Toppings last 2" get plain oatmeal ingredients first
                                  then use them, then use toppings after getting each one.

        Generates a subtree representing making cereal
        """

        at = self.action_type_dict["main"]
        steps = {
            "get_oats": "gather(oats:ingredient,do:action_type)",
            "get_salt": "gather(salt:ingredient,do:action_type)",
            "get_bowl": "gather(bowl:container,do:action_type)",
            "get_salt": "gather(salt:ingredient,do:action_type)",
            "get_milk": "gather(milk:liquid,do:action_type)",
            "pour_milk_pan": "pour(milk:liquid,pan:container,{at}:action_type)",
            "get_blueberry": "gather(blueberry:ingredient,do:action_type)",
            "get_strawberry": "gather(strawberry:ingredient,do:action_type)",
            "get_banana": "gather(banana:ingredient,do:action_type)",
            "get_chocolate": "gather(chocolatechips:ingredient,do:action_type)",
            "get_pb": "gather(peanutbutter:ingredient,do:action_type)",
            "get_cup": "gather(measuringcup:container,do:action_type)",
            # "put_sink": "putinsink(measuringcup:container,sink:appliance,{at}:action_type)",
            "collect_water": "collectwater(measuringcup:container,water:liquid,{at}:action_type)",
            "pour_water": "pourwater(pan:container,{at}:action_type)",
            "pour_salt": "pour(salt:ingredient,pan:container,{at}:action_type)",
            "turn_on": "turnon(stove:appliance,{at}:action_type)",
            "pour_oats": "pour(oats:ingredient,pan:container,{at}:action_type)",
            # "grab_spoon": "grabspoon(mixingspoon:tool,do:action_type)",
            # "mix": "mix({liquid}:liquid,pan:container,main:meal,{at}:action_type)",
            "mix": "mix(pan:container,main:meal,{at}:action_type)",
            "get_mixingspoon": "gather(mixingspoon:tool,do:action_type)",
            "get_eatingspoon": "gather(eatingspoon:tool,do:action_type)",
            "reduce": "reduceheat(pan:container,main:meal,{at}:action_type)",
            # "boil": "boilliquid(pan:container,{liquid}:liquid,say:action_type)",
            "boil": "boilliquid(main:meal,pan:container,say:action_type)",
            "cook_oatmeal": "cookoatmeal(stove:appliance,main:meal,say:action_type)",
            "pour_blueberry": "pour(blueberry:ingredient,bowl:container,{at}:action_type)",
            "pour_banana": "pour(banana:ingredient,bowl:container,{at}:action_type)",
            "pour_strawberry": "pour(strawberry:ingredient,bowl:container,{at}:action_type)",
            "pour_chocolate": "pour(chocolatechips:ingredient,bowl:container,{at}:action_type)",
            "pour_pb": "pour(peanutbutter:ingredient,bowl:container,{at}:action_type)",
            "serve":"serveoatmeal(pan:container,bowl:container,main:meal,{at}:action_type)",
        }
        steps = {k:LeafCombination(Action(name=v.format(at=at,liquid=self.liquid))) \
                 for k,v in steps.items()}

        if self.liquid == MILK:
            liquid_get = [steps["get_milk"]]
            liquid_add = [steps["pour_milk_pan"]]
        elif self.liquid == WATER:
            liquid_get = [steps["get_cup"]] # Getting cup puts it in sink
            liquid_add = [steps[a] for a in ["collect_water",
                                                "pour_water"]]

        plain_get = [steps[a] for a in ["get_oats","get_bowl", "get_eatingspoon", "get_salt"] ]
        if at == "do":
          make_oatmeal = [steps[a] for a in ["pour_salt", "turn_on", "boil",
                                              "pour_oats", "mix",
                                              "reduce", "cook_oatmeal", "serve"]]
        elif at == "say":
            plain_get.append(steps["get_mixingspoon"])
            make_oatmeal = [steps[a] for a in ["pour_salt", "turn_on", "boil",
                                              "pour_oats" , "mix",
                                              "reduce", "cook_oatmeal", "serve"]]

        if self.main == PLAINOATMEAL:
            toppings_get = []
            toppings_pour = []
        elif self.main == FRUITYOATMEAL:
            toppings_get = [steps[a] for a in ["get_strawberry", "get_blueberry",
                                            "get_banana"]]
            toppings_pour = [steps[a] for a in ["pour_strawberry", "pour_blueberry",
                                            "pour_banana"]]
        elif self.main == CHOCOLATEOATMEAL:
            toppings_get = [steps["get_chocolate"]]
            toppings_pour = [steps["pour_chocolate"]]
        elif self.main == FRUITYCHOCOLATEOATMEAL:
            toppings_get = [steps[a] for a in ["get_strawberry", "get_blueberry",
                                                "get_banana", "get_chocolate"]]
            toppings_pour = [steps[a] for a in ["pour_strawberry", "pour_blueberry",
                                                "pour_banana", "pour_chocolate"]]
        elif self.main == PBBANANAOATMEAL:
            toppings_get = [steps["get_pb"], steps["get_banana"]]
            toppings_pour = [steps["pour_pb"], steps["pour_banana"]]
        elif self.main == PBCHOCOLATEOATMEAL:
            toppings_get = [steps["get_pb"],
                            steps["get_chocolate"]]
            toppings_pour = [steps["pour_pb"],
                            steps["pour_chocolate"]]

        # import pdb; pdb.set_trace()
        if variation == "ingredients first":

            get_combo = ParallelCombination(plain_get + liquid_get + toppings_get)
            make_oatmeal_combo = SequentialCombination(liquid_add + make_oatmeal)
            toppings_combo = ParallelCombination(toppings_pour)

            ret_tree = SequentialCombination([get_combo, make_oatmeal_combo,
                                                  toppings_combo])

        elif "use immediately" in variation:

            if at == "do":
              make_oatmeal = [steps[a] for a in ["get_salt", "pour_salt", "turn_on", "boil", "get_oats",
                                                "pour_oats", "mix",
                                                "reduce", "cook_oatmeal", "get_bowl", "get_eatingspoon", "serve"]]
            elif at == "say":
              make_oatmeal = [steps[a] for a in ["get_salt", "pour_salt", "turn_on", "boil", "get_oats",
                                                "pour_oats", "get_mixingspoon", "mix",
                                                "reduce", "cook_oatmeal", "get_bowl", "get_eatingspoon", "serve"]]


            liquid_and_oatmeal_combo = SequentialCombination(liquid_get + \
                                                             liquid_add + make_oatmeal)
            toppings_combo = ParallelCombination(
                [SequentialCombination(list(a)) for a in zip(toppings_get,
                                                             toppings_pour)]
                    )
            ret_tree = SequentialCombination([liquid_and_oatmeal_combo, toppings_combo])

        elif "toppings last" in variation:
            plain_get_combo = ParallelCombination(plain_get)
            liquid_and_oatmeal_combo = SequentialCombination(liquid_get + liquid_add + make_oatmeal)
            get_toppings_combo = ParallelCombination(toppings_get)
            pour_toppings_combo = ParallelCombination(toppings_pour)

            if "1" in variation:

                toppings_combo = SequentialCombination([get_toppings_combo,
                                                        pour_toppings_combo])
            elif "2" in variation:
                toppings_combo = ParallelCombination(
                    [SequentialCombination(list(a)) for a in zip(toppings_get,
                                                                 toppings_pour)]
                    )

            ret_tree = SequentialCombination([plain_get_combo, liquid_and_oatmeal_combo,
                                              toppings_combo])


        return ret_tree


    def _generate_meal(self, main_type):
        assert main_type  in ["use immediately","ingredients first"]
        # assert order in ["main first", "side first"]

        if self.main == CEREAL:
            main = self._generate_cereal_combo(main_type)
        else:
            main = self._generate_oatmeal_combo(main_type)


        side = self._generate_side_combo()

        if self.order == "side first":
            meal_order = [side, main]
            main_order = "first"
            side_order = "last"
        elif self.order == "main first":
            meal_order = [main, side]
            main_order = "last"
            side_order = "first"

        data = {"main": {"meal": self.main,
                         "order": main_order,
                         "action_type": self.action_type_dict["main"]},
                "side": {"meal": self.side,
                         "order": side_order,
                         "action_type": self.action_type_dict["side"]
                         },
                "liquid": self.liquid
                }
        return HierarchicalTask(SequentialCombination(meal_order),
                                data=data
                                )




    def use_immediately(self):
        return self._generate_meal(main_type="use immediately")


    def ingredients_first(self):
        return self._generate_meal(main_type="ingredients first")



def generate_action_space(pddl_dir=None, save_dir=None):
    """
    Function for generating all the relevant possible actions for our experiments.
    Because actions are predicated defined using PDDL, the complete action space
    is actually all possible groundings of each of these predicates. However, since
    in practice the robot can only enact a subset of these, we want to limit the action
    space to only robot-viable ones. This function enables you to automatically generate
    these actions based on our collaborative cooking task.
    """
    ret_action_space = set()

    if save_dir == None:
        save_dir = DATA_PATH

    env = load_environment(pddl_dir, False)
    state, _  = env.reset()

    complete_action_space = list(env.action_space.all_ground_literals(state, False))
    side = [MUFFIN, JELLYPASTRY, ROLL, EGG, PIE]
    # pastries = ['jellypastry']
    # main = [CEREAL, FRUITYOATMEAL, CHOCOLATEOATMEAL, PLAINOATMEAL, FRUITYCHOCOLATEOATMEAL,
    #                      PBBANANAOATMEAL, PBCHOCOLATEOATMEAL]
    main = [CEREAL, FRUITYCHOCOLATEOATMEAL ,PBCHOCOLATEOATMEAL]
    # meal = ['fruity']
    # liquids = [MILK, WATER]
    liquids = [WATER, MILK]
    action_type = ['do']

    for mt in main:
        for p in side:
            for l in liquids:
                for at in action_type:
                    # plan = Plan(mt, p, l, {'main':"do", 'side':"do"}, 1)
                    env = load_environment(pddl_dir, False, goal_main=mt,
                                           goal_side=p)
                    # env.fix_problem_index(0)
                    htn_factory = CookingHTNFactory(mt, p, l, {'main':"do", 'side':"do"}, order="main first")
                    htn = htn_factory.use_immediately()
                    state, _  = env.reset()
                    # for action in plan.get_plan(complete_action_space):
                    action, _ = htn.get_next_action(complete_action_space, [])
                    while not action is None:
                        # import pdb; pdb.set_trace()
                        # print(env.action_space.all_ground_literals(state))
                        # print(action)
                        if "check" in action.predicate.name:
                            print("[generate_action_plan] skipping: ", action)
                        else:
                            action_space_filtered = set([a for a in env.action_space.all_ground_literals(state) if not "check" in a.predicate.name])
                            ret_action_space = ret_action_space.union(action_space_filtered)
                        next_state, _,done,_ = env.step(action)
                        # assert that the next state is not the same as the current state
                        assert not next_state == state, print("action: {}\nstate:{}".format(action, state))
                        assert not next_state == state, pdb.set_trace()
                        action, _ = htn.get_next_action(complete_action_space, [])

                        state = next_state
                        # print("Done: ", done)

                    state,_ = env.reset()
                    print("No. of actions: ", len(ret_action_space))
                    # print(ret_action_space)

                    np.save(os.path.join(save_dir, "ACTION_SPACE.npy"),
                            list(ret_action_space),
                            allow_pickle=True)

if __name__ == '__main__':
    print(PDDL_DIR)
    generate_action_space(PDDL_DIR)

    # print(task.generate_all_permutations(flat_tree, []))