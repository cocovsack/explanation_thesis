import os
from pyswip import Prolog, Functor, newModule, call, Variable
import numpy as np

# from pddlgym_planners.ff import FF
from pddlgym.core import PDDLEnv
from pddlgym.inference import find_satisfying_assignments, ProofSearchTree
from collections import OrderedDict, defaultdict
try:
    from learning_util import (ACTION_DICT, INGREDIENT_DICT,
                               ALL_NUTRITIONAL_PREDS, ALL_MEAL_PREDS,
                               ALL_ACTION_TYPE_PREDS, ALL_PRECURSOR_PREDS,
                               NO_ACTION_TYPE_VARIANTS, PASTRY_LIST, MAIN_LIST,
                               OATMEAL_LIST
                               )
except ModuleNotFoundError:
    from chefbot_utils.learning_util import (ACTION_DICT, INGREDIENT_DICT,
                                             ALL_NUTRITIONAL_PREDS, ALL_MEAL_PREDS,
                                             ALL_ACTION_TYPE_PREDS, ALL_PRECURSOR_PREDS,
                                             NO_ACTION_TYPE_VARIANTS, PASTRY_LIST, MAIN_LIST,
                                             OATMEAL_LIST
                                             )



OATMEAL_INGREDIENTS = set(['bowl', 'oats', 'measuringcup', 'salt', 'mixingspoon',
                           'stove', "eatingspoon", "measuringcup", "milk", "pan"])
FRUITY_OATMEAL_INGREDIENTS = OATMEAL_INGREDIENTS.union(set(['blueberry', 'strawberry',
                                                            "banana"]))
FRUITY_CHOCOLATE_OATMEAL_INGREDIENTS = OATMEAL_INGREDIENTS.union(set(['blueberry', 'strawberry',
                                                                      "banana", "chocolatechips"]))


PASTRY_INGREDIENTS = set(['pie', 'muffin', 'microwave', "egg", "roll", "jellypastry"])
INTERVENTION_ACTION = set(['gather', 'pour', 'appliance', 'mix', 'collect_water', 'turn_on',
                           'spread'])
WAIT_ACTION = set(['toast_bread', 'boil_water', "check_jelly_toast", "check_tomato_toast",
                   'check_pb_banana_toast', 'microwave_pastry', 'check_pie', "check_muffin",
                   "check_plain_oatmeal", "check_fruity_oatmeal"])

CHECK_ACTION  = set(['check_jelly_toast', 'check_tomato_toast', 'check_pb_banana_toast',
                     'microwave_pastry', 'check_pie', "check_muffin", "check_plain_oatmeal",
                     "check_fruity_oatmeal"])


def add_shelf_classifaction(ingredient_dict):
    # TODO: Change these to rosparams as this is subject to change.
    top_shelf = ['oats', 'milk', 'bowl']
    mid_top_shelf = ['measuringcup', 'jellypastry', 'banana']
    mid_shelf = ['roll', 'cocopuffs', 'peanutbutter']
    mid_bottom_shelf = ['blueberry', 'strawberry', 'pie']
    bottom_shelf = ['muffin', 'egg', 'chocolatechips']

    for i in top_shelf:
        ingredient_dict[i].add("top_shelf")
    for i in mid_top_shelf:
        ingredient_dict[i].add("mid_top_shelf")
    for i in mid_shelf:
        ingredient_dict[i].add("mid_shelf")
    for i in mid_bottom_shelf:
        ingredient_dict[i].add("mid_bottom_shelf")
    for i in bottom_shelf:
        ingredient_dict[i].add("bottom_shelf")


    return ingredient_dict



class StateTransformer(PDDLEnv):
    """"
    This class transforms a pddl state into a fixed length boolean vector where each feature evaluates some overlay.
    We use ANOTHER pddl domain to define these overlays and evaluate them on the state.
    """
    def __init__(self, domain_file, problem_dir, overlays = None, goal_condition=True):
        super(StateTransformer, self).__init__(domain_file, problem_dir,
                                               operators_as_actions=True,
                                               dynamic_action_space=True,
                                               )
        # Goal features
        oatmeal_features = ["is{}".format(o) for o in OATMEAL_LIST]
        main_features = ["iscompleted{}".format(m) for m in MAIN_LIST]
        side_features = ["iscompleted{}".format(p) for p in PASTRY_LIST]
        self.goal_features = ["goal_{}".format(f) for f in oatmeal_features + side_features + main_features]
        self.state_features =  [f for f in self.domain.operators.keys()]
        self._goal_condition = goal_condition

        # self.action_features = ['is_making_oatmeal', 'is_making_cereal', 'is_making_pastry']
        self._overlays = overlays

    @property
    def goal_condition(self):
        return self._goal_condition

    @goal_condition.setter
    def goal_condition(self, g_c):
        self._goal_condition = g_c


    def n_features(self):
        if self._goal_condition:
            return len(self.state_features) + len(self.goal_features)
        else:
            return len(self.state_features)


    # @property
    # def overlays(self):
    #     return self._overlays

    # @overlays.setter
    # def overlays(self, overlays):
    #     self._overlays = overlays


    def transform(self, state, actions=None):
        """
        Transforms a PDDLEnv state into a fixed length binary vector.

        @state: A frozen set of grounded literals
        @returns: binary np.ndarray representation of state
        """
        all_vars = {}
        state_vars = self.eval_pddl_vars(state)
        all_vars.update(state_vars)

        if self._goal_condition:
            goal_vars = self._eval_goal_vars(state)
            all_vars.update(goal_vars)
            all_features = self.state_features + self.goal_features
        else:
            all_features = self.state_features

        feature_vec  = np.zeros(len(all_features),
                                dtype=np.bool)

        for i, var in enumerate(all_features):
            # feature_vec[i] = state_vars[var]
            feature_vec[i] = all_vars[var]

        return feature_vec


    def _eval_goal_vars(self, state):
        """
        Get vars encoding the current meal goal
        """
        ret_vars = {f: False for f in self.goal_features}
        ret_vars["goal_iscompletedpastry"] = True
        curr_goal_set = set(["goal_{}".format(g.predicate.name) for g in state.goal.literals])
        print("[state_transformer] curr_goal_set: ", curr_goal_set)
        for g in curr_goal_set.intersection(set(ret_vars)):
            ret_vars[g] = True

        return ret_vars



    def eval_pddl_vars(self, state):
        """
        Evaluates simple FOL statements on the world state defined as PDDL actions/operators.
        This is a bit of a hacky to do FOL by exploting PDDLGym implementation.

        @input: World state as defined by PDDLGym
        @returns: a dict {'state_feature_name':bool}
        """
        feats = {}
        kb  = set(state.literals)
        for name, operator in self.domain.operators.items():
            # operators.add(operator)
            cond = operator.preconds.literals
            # import pdb; pdb.set_trace()
            # print(name, cond)
            # print(self.sample_transition(lit))
            # print(self.domain.predicates)
            # print("Operators: ", operators)
            # print("Possible operators: ", set(self.domain.operators.values()))
            verbose = False
            # if name == "oatmeal_done":
            #     state_str = ["iscompletedoatmeal" in str(s) for s in state.literals]
            #     if any(state_str):
            #         import pdb
            #         pdb.set_trace()
            assigments =  ProofSearchTree(kb,constants=None,
                                 allow_redundant_variables=True,
                                 type_to_parent_types=self.domain.type_to_parent_types
                                 ).prove(list(cond), max_assignment_count=5,
                                            verbose=verbose)

            # print("Conds: ", cond)
            # print("assignments: ", assigments)
            feats[name] = len(assigments)  > 0

        # print(feats)
        return feats





class PrologClassifier(object):
    def __init__(self):
        """
        Enables the ability to perform prolog queries on the state of the world.
        Primarily use by the overlays to evaluate their rules.

        Prolog.assertz(fact): adds a fact or rule to the knowledge base to be used
        to do reasoning
        Prolog.query(q): queries the knowledge base and returns either a bool in the case
        of completely grounded queries, or a list of groundings when the query is lifted.
        Prolog.retract(fact): remove a grounded fact from KM
        Prolog.retracatll(Fact): is a lifted form of retract and removes all facts that
        satify the lifted fact.
        """
        # self.env_domain = env_domain
        self.p = Prolog()

        self.init_state = True
        self.state_classifier = StateClassifier()


        preds = set.union(*[ALL_NUTRITIONAL_PREDS, ALL_MEAL_PREDS,
                            ALL_ACTION_TYPE_PREDS, ALL_PRECURSOR_PREDS])
        self._action_classifier_dict = {p:("{}(X)".format(p), 1) for p in preds}
        self._action_classifier_dict['equiv_action'] = ("equiv_action({X},{Y})", 2)
        self._action_classifier_dict['do_only'] = ("do_only({X})", 1)
        self._action_classifier_dict['say_only'] = ("say_only({X})", 1)


        self._state_classifier_dict = {
            "state": ("state({X})", 1)
        }

        self.curr_state_dict = {}
        print("About to initalize predicates")
        self._initialize_predicates()
        print("Done initializing")
        # self._gen_equiv_dict()



    def _initialize_predicates(self):
        """
        Need to initialize all the predicates we will use as dynamic
        in order to modify them successfully.
        """
        # import pdb; pdb.set_trace()
        dyn_pred_template = ":- dynamic {pred}/{arity}"
        for pred in self._action_classifier_dict:
            _, arity = self._action_classifier_dict[pred]
            rule = dyn_pred_template.format(pred=pred, arity=arity)
            self.p.assertz(rule)
            # except Exception as e:
            #     import pdb
            #     pdb.set_trace()

        for pred in self._state_classifier_dict:
            _, arity = self._state_classifier_dict[pred]
            self.p.assertz(dyn_pred_template.format(pred=pred, arity=arity))

    def _update_state(self, new_state_dict):
        """
        Adds/removes predicates to knowledge based on new state info
        @new_state_dict dict of predicates of the form {pred_name: bool}
        """

        # Add abstract features to state dict.
        abstract_feat_dict = self.state_classifier.classfify_state_action(new_state_dict)
        new_state_dict.update(abstract_feat_dict)

        new_state_set = set(new_state_dict.items())
        curr_state_set = set(self.curr_state_dict.items())

        for update in (new_state_set - curr_state_set):
            name, state_bool = update
            if state_bool:
                self.p.assertz("state({})".format(name))
                if "dish" in name:
                    print("[update_state] state: ", name)
            else:
                list(self.p.query("retract(state({}))".format(name)))

        self.curr_state_dict = new_state_dict


    def query(self, q):
        """
        Performs a Prolog query using the current knowledge base and returns the results.
        Queries will be of the form "precond -> postcond", where precond is an
        arbitrarily long prolog expression comprised of lifted predicates
        (e.g. pred(X)) and/or grounded ones (e.g. pred(x)) operators AND (,),
        OR (;), NOT (\+), and IMPLY (->), and postcond is a lifted predicate.

        @input: q, str: A prolog query
        @returns a list of dicts of the form [{Var1: pddlgym.struct.TypedEntity, ...}...]
        where each dict contains possible groundings for all lifted variables in the query.

        NOTE: A lifted predicate is denote by a predicate with an uppercase parameter name
        e.g. pred(X) or pred(Lifted_var_1, grounded_var_2)

        An empty list means there were no possible groundings in the current state.


        """

        query = self.p.query(q)
        ret_list = []
        # Action groundings need to be converted from their simplified str representation to
        # PDDL operator representation
        # print("[query] ", self._str_to_action_dict)
        # print("[query] res: ", [a for a in query])
        # TKTKTK error here when given " you make the oatemeal and ill make the pastry"
        # import pdb
        # pdb.set_trace()
        # print("[query] ", self._str_to_action_dict)
        for res in query:
            # If all of the results is of type Variable, the result is invalid
            if  all([isinstance(v, Variable) for v in res.values()]):
                continue
            # If only one of the values is a Variable...
            elif any([isinstance(v, Variable) for v in res.values()]):
                # ...then replace it with the action of the other key
                assert len(res) == 2
                valid_a  = [v for k,v in res.items() if not isinstance(v, Variable)][0]
                res = {k:valid_a for k in res.keys()}

            try:
                 # ret_list.append({k:self._str_to_action_dict[v] for k,v in res.items()})
                 ret_list.append({k:self._str_to_action_dict[v] for k, v in res.items() \
                                  if v in self._str_to_action_dict})
            except KeyError:
                print(
                    "Error: query: {} returned an invalid res! This probably means the KB wasnt reset properly".format(q))
                print("[query] res: ", res)
                print("[query] ", self._str_to_action_dict)
                import pdb; pdb.set_trace()
                raise
        # remove any duplicate query results as they will be counted twice during  reasoning
        ret_list = [dict(t) for t in set(tuple(d.items()) for d in ret_list)]
        print("HERREEEEE", ret_list)
        return ret_list


    def reset(self):
        """
        Removes action related predicates. This needs to be done or else
        the reasoner will consider actions that are no longer relevant
        """
        for a, _ in self._action_classifier_dict.values():
            lifted_a = a.format(X="X", Y="Y")
            # print("retracting: ", lifted_a)
            # NOTE:  Prolog.query returns a generator that need to
            # be iterated in order to work properly
            # list(self.p.query("retractall({})".format(lifted_a)))
            list(self.p.query("retractall({})".format(lifted_a)))
            # print("retracting: ", lifted_a)
            # Check if knowledge base was successfully reset
            assert [a for a in self.p.query(lifted_a)] == []
            # import pdb; pdb.set_trace()


    def build_kb(self, state_dict, possible_actions):
        self.reset()
        self._gen_action_str_mapping(possible_actions)
        self._check_action_type(possible_actions)
        self._check_equiv_action(possible_actions)
        self._update_state(state_dict)
        # import pdb; pdb.set_trace()

    def _gen_action_str_mapping(self, possible_actions):
        """
        Generate dictionaries that map possible actions to simplified string name
        Prolog doesn't like strips style actions (e.g. action_name(foo:bar, ...))
        """
        self._action_to_str_dict = {}
        self._str_to_action_dict = {}
        for i, a in enumerate(possible_actions):
            a_str = "a{}".format(i)
            self._action_to_str_dict[a]  = a_str
            self._str_to_action_dict[a_str] = a


    def _check_action_type(self, possible_actions):
        """
        Add facts about the possible actions to KB
        @input: possible_actions, [pddlgym.structs.TypedEntity, ...], a list
        of possible grounded actions that can be performed in this state.
        """

        for p_a in possible_actions:

            # print("Adding facts for: ", p_a)
            # Get simplified action name from mapping dict.

            a_str = self._action_to_str_dict[p_a]
            self.p.assertz("action({})".format(a_str))

            # for action_fact in ACTION_DICT[p_a.predicate.name]:
            #     self.p.assertz("{fact}({name})".format(fact=action_fact, name=a_str))
                # print("[check_action_type] Adding fact: {fact}({action_name})".format(fact=action_fact,
            #                                                                           action_name=a_str))

            # Get a list of all the facts for all ingredients paramterizing action
            # ingredient_facts = set.intersection(*[INGREDIENT_DICT[v.name] for v in p_a.variables])
            fact_set_list = [ACTION_DICT[p_a.predicate.name]] +  \
                [INGREDIENT_DICT[v.name] for v in p_a.variables]
            # print("ingreident facts: ", ingredient_facts)
            # fact_set_list = [ACTION_DICT[p_a.predicate.name]] + [ingredient_facts]
            # fact_set_list = [a_f for a_f in ACTION_DICT[p_a.predicate.name]] + fact_set_list
            # action_type = fact_set_list.pop(-1)
            # get th intersection of all the facts
            # Make sure the last set of facts in the fact_set_list are action_type related
            # print(p_a)
            # import pdb; pdb.set_trace()
            assert len(set(['say', 'do']).intersection(fact_set_list[-1])) > 0
            # Get the facts that are shared in common amongst all action parameters
            try:
                shared_facts = set.intersection(*fact_set_list[:-1])
            except TypeError:
                import pdb; pdb.set_trace()

            # If the current action doesn't has not action type variants (e.g. gather actions),
            # we pretend like we do in order to simplify overlay rules.
            if p_a.predicate.name in NO_ACTION_TYPE_VARIANTS:
                for action_fact in ["say", "do"]:
                    self.p.assertz("{fact}({action_name})".format(fact=action_fact,
                                                              action_name=a_str))

            if len(shared_facts) > 0:
                for v in p_a.variables:
                    for action_fact in INGREDIENT_DICT[v.name]:
                        # print("[check_action_type] Adding fact: {fact}({action_name})".format(fact=action_fact, action_name=a_str))
                        # An action shoul never be associated simultaneously with more than one meal type.
                        # (e.g. oatmeal, pastry, cereal) this helps avoid that
                        if "making" in action_fact and not action_fact in shared_facts:
                            pass
                        else:
                            self.p.assertz("{fact}({action_name})".format(fact=action_fact,
                                                                          action_name=a_str))
            # Otherwise the action is weirdly grounded, so we only add the action_type
            # so as to prevent rules from inadvertantly promoting non-relevant actions
            else:
                action_fact = list(fact_set_list[-1])[0]
                self.p.assertz("{fact}({action_name})".format(fact=action_fact,
                                                            action_name=a_str))
                # print("[check_action_type] Adding fact: {fact}({action_name})".format(fact=action_fact, action_name=a_str))



    def _check_equiv_action(self, possible_actions):
        """
        Checks if two actions have the same effect, and if so
        considers them equivalent and adds this knowledge to the KB
        """

        action_diff = set(["say", "do"])

        for a1 in possible_actions:
            a1_name = a1.predicate.name
            if a1_name in NO_ACTION_TYPE_VARIANTS:
                continue
            # List of variables parameterizing the action
            a1_vars = set([v.name for v in a1.variables])
            for a2 in possible_actions:
                a2_name = a2.predicate.name
                if a2_name in NO_ACTION_TYPE_VARIANTS:
                    continue
                # List of variables parameterizing the action
                a2_vars = set([v.name for v in a2.variables])
                # Are the two action types equivalent?
                diff = a1_vars.symmetric_difference(a2_vars)
                # if the actions have the same name and only differ due to the say/do param
                # then they are equivalent.
                if a1_name == a2_name and diff == action_diff:
                    self.p.assertz("equiv_action({}, {})".format(self._action_to_str_dict[a1],
                                                                 self._action_to_str_dict[a2]))



class PrologShield(PrologClassifier):
    def __init__(self, action_space):
        "docstring"
        super(PrologShield, self).__init__()
        self.action_space = action_space

        self._state_classifier_dict = {
            "action": ("action({X})", 1),
        }
        self._state_classifier_dict = {
             "state": ("state({X})", 1)
        }

        self._gen_action_str_mapping(action_space)

    def _format_query(self, query_dict):
        """
        Formats a query for Prolog
        @input: pred_list, [str, ...], a list of predicates to be queried
        """
        if "refine" in query_dict:
            ret = query_dict["refine"]

        elif "alternate" in query_dict:
            curr_action = query_dict["alternate"]["curr_action"]
            alt_action = query_dict["alternate"]["fail_action"]

            curr_action_str = self._action_to_str_dict[curr_action]
            alt_action_str = self._action_to_str_dict[alt_action]
            ret = "action({}) = action({})".format(curr_action_str,
                                                   alt_action_str)
        elif "forbid" in query_dict:
            ret = query_dict["forbid"]



        return ret


    def build_kb(self, state_dict, possible_actions):
        self.reset()
        self._check_action_type(possible_actions)
        self._update_state(state_dict)

    def query(self, query_dict):
        """
        Queries the Prolog KB for the given query_dict
        @input: query_dict, {str: [str, ...], ...}, a dictionary of queries
        """
        query = self._format_query(query_dict)
        res = [ a for a in self.p.query(query)]  # if res is empty, then there is no solution

        return len(res) > 0


    def _update_state(self, new_state_dict):
        """
        Adds/removes predicates to knowledge based on new state info
        @new_state_dict dict of predicates of the form {pred_name: bool}
        """

        # Add abstract features to state dict.
        abstract_feat_dict = self.state_classifier.classfify_state_action(new_state_dict)
        new_state_dict.update(abstract_feat_dict)

        for s in self.curr_state_dict:
            list(self.p.query("retract(state({}))".format(s)))

        for s, b in new_state_dict.items():
            if b:
                self.p.assertz("state({})".format(s))


        self.curr_state_dict = new_state_dict

class StateClassifier(object):
    """
    A class for creating high level state features from lower level state features.
    """
    def __init__(self):
        self._feature_dict = {}
        self._abstract_feature_dict = {}


    def classfify_state_action(self, state_dict):
        self._feature_dict = state_dict
        ret_dict = {}

        abstract_features = [self._meal_count_state,
                             ]

        for f in abstract_features:
            ret_dict.update(f())


        return ret_dict

    def _meal_count_state(self):
        """
        Counts and creates state feature relating to how many meals (e.g. pastry, oatmeal, or cereal) have currently been completed.
        """
        ret_dict = {}
        # completed_list = ["pastry_done", "oatmeal_done", "toast_done"]
        completed_list = ["pastry_done", "oatmeal_done", "cereal_done"]
        completed_counts = [self._feature_dict[c] for c in completed_list if c in self._feature_dict]

        sum_counts = sum(completed_counts)

        ret_dict['no_completed_dish'] = sum_counts == 0
        ret_dict['one_completed_dish'] = sum_counts == 1
        ret_dict['two_completed_dish'] = sum_counts == 2
        # ret_dict['three_completed_dish'] = sum_counts == 3

        return ret_dict





if __name__ == "__main__":
    # gen.save_domain_to_file()
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pddl")
    name = "cooking"
    domain_file = os.path.join(dir_path, '{}_domain.pddl'.format(name))
    feature_domain = os.path.join(dir_path, 'generated_feature_domain.pddl'.format(name))
    problem_dir = os.path.join(dir_path, name)

    env  = PDDLEnv(domain_file, problem_dir,
            operators_as_actions=False,
            dynamic_action_space=True,
                  )


    gen = FeatureDomainGenerator(env)
    gen.save_domain_to_file()