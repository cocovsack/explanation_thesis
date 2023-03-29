#!/usr/bin/env python

import rospy

from rospy_message_converter import message_converter
# from pyswip import Prolog
from rosprolog_client import PrologException, Prolog
from chefbot.srv import PrologQuery, PrologQueryRequest, PrologQueryResponse
from chefbot.srv import StateActionUpdate, StateActionUpdateRequest, StateActionUpdateResponse
from chefbot_utils.learning_util import (ACTION_DICT, INGREDIENT_DICT, ALL_CPSC573_PREDS
                                             ALL_NUTRITIONAL_PREDS, ALL_MEAL_PREDS,
                                             ALL_ACTION_TYPE_PREDS, ALL_PRECURSOR_PREDS)


class Literal(object):
    def __init__(self, pred_str):
        self._pred_str  = pred_str
        self._variables = None
        self._predicate = None
        self._process_pred_str()

    def _process_pred_str(self):
        self._predicate, variables = self._pred_str.split("(")
        variables = variables.strip(")")
        variables = variables.split(",")
        self._variables = [v.split(":")[0] for v in variables]

    @property
    def predicate(self):
        return self._predicate
    @property
    def variables(self):
        return self._variables

    def __str__(self):
        return "{}".format(self._pred_str)


class StateClassifier(object):
    def __init__(self):
        self._feature_dict = {}
        self._abstract_feature_dict = {}


    def classfify_state_action(self, state_dict):
        self._feature_dict = state_dict
        ret_dict = {}

        abstract_features = [self._meal_count_state]
                             # self._making_oatmeal]

        for f in abstract_features:
            ret_dict.update(f())

        # print("[classify_state_action] abstract features: ", ret_dict)
        # ret_dict.update(self._feature_dict)
        # ret_dict.update(self._abstract_feature_dict)

        return ret_dict

    def _meal_count_state(self):
        ret_dict = {}
        # completed_list = ["pastry_done", "oatmeal_done", "toast_done"]
        completed_list = ["pastry_done", "oatmeal_done", "cereal_done"]
        completed_counts = [self._feature_dict[c] for c in completed_list]

        sum_counts = sum(completed_counts)

        ret_dict['no_completed_dish'] = sum_counts == 0
        ret_dict['one_completed_dish'] = sum_counts == 1
        ret_dict['two_completed_dish'] = sum_counts == 2
        # ret_dict['three_completed_dish'] = sum_counts == 3

        return ret_dict




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
        print("Inside prolgclassifier")
        self.p = Prolog()
        print("Started prolog!")

        self.init_state = True
        self.state_classifier = StateClassifier()

        # import pdb
        # pdb.set_trace()
        # self.kb = newModule("kb")

        # self.equiv_action = Functor("equiv_action", 2)
        # self.intervention = Functor("intervention")
        # self.verbal = Functor("verbal")
        # These track the different action classifier rules.
        preds = set.union(*[ALL_NUTRITIONAL_PREDS, ALL_MEAL_PREDS,
                            ALL_ACTION_TYPE_PREDS, ALL_PRECURSOR_PREDS, ALL_CPSC573_PREDS])
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
        rospy.Service('chefbot/prolog/kb_update', StateActionUpdate, self._kb_update_cb)
        rospy.Service('chefbot/prolog/query', PrologQuery, self._query_cb)
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
            # self.p.assertz(rule)
            self.p.query("assertz({})".format(rule))

        for pred in self._state_classifier_dict:
            _, arity = self._state_classifier_dict[pred]
            rule = dyn_pred_template.format(pred=pred, arity=arity)
            self.p.query("assertz({})".format(rule))
            # self.p.assertz(rule)

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
                # self.p.assertz("state({})".format(name))
                self.p.query("assertz(state({}))".format(name))
                if "dish" in name:
                    print("[update_state] state: ", name)
            else:
                # list(self.p.query("retract(state({}))".format(name)))
                query = self.p.query("retract(state({}))".format(name))
                # for sol in query.solutions():
                #     print("sol: ", sol)
                # query.finish()

        self.curr_state_dict = new_state_dict


    def _query_cb(self, srv):
        # rospy.loginfo("Recieved query: {}".format(srv))
        res = self.query(srv.query)
        resp = PrologQueryResponse
        # print("res: ", res)
        resp = [str(r) for r in res]
        # rospy.loginfo("Returning response: {}".format(resp))
        return PrologQueryResponse(resp)


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

        print("INSIDE QUERY FUNCTION")

        query = self.p.query(q)
        ret_list = []
        # Action groundings need to be converted from their simplified str representation to
        # PDDL operator representation
        # import pdb; pdb.set_trace()
        # print("[query] ", self._str_to_action_dict)
        # print("[query] res: ", [a for a in query])

        try:
            for res in query.solutions():
                # If all of the results is of type Variable, the result is invalid
                # print(res)
                # if  all([isinstance(v, Variable) for v in res.values()]):
                #     continue
                # # If only one of the values is a Variable...
                # elif any([isinstance(v, Variable) for v in res.values()]):
                #     # ...then replace it with the action of the other key
                #     assert len(res) == 2
                # valid_a  = [v for k,v in res.items() if not isinstance(v, Variable)][0]
                # valid_a  = [v for k,v in res.items() ][0]
                # res = {k:valid_a for k in res.keys()}
                try:

                    ret_list.append({k:self._str_to_action_dict[v] for k,v in res.items() if v in self._str_to_action_dict})
                    # print("ret list: ", ret_list)
                except KeyError:
                    print(
                        "Error: query: {} returned an invalid res! This probably means the KB wasnt reset properly".format(q))
                    print("[query] res: ", res)
                    print("[query] ", self._str_to_action_dict)
                    # import pdb; pdb.set_trace()
                    raise
            # remove any duplicate query results as they will be counted twice during  reasoning
            ret_list = [dict(t) for t in set(tuple(d.items()) for d in ret_list)]

            query.finish()
        except PrologException:
            rospy.loginfo("Query failed!")
            ret_list = []

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
            # list(self.p.query("retractall({})".format(lifted_a)))
            query = self.p.query("retractall({})".format(lifted_a))
            # for sol in query.solutions():
            #     pass
            # query.finish()
            # print("retracting: ", lifted_a)
            # Check if knowledge base was successfully reset
            # test_query = self.p.query(lifted_a)
            # assert [a for a in self.p.query(lifted_a)] == []
            # import pdb; pdb.set_trace()


    def _process_srv(self, srv):
        features = srv.state_features
        bools = srv.state_bools
        # bools = ["True" == b for b in bools]
        state_dict = dict(zip(features, bools))
        # import pdb; pdb.set_trace()
        possible_actions = [Literal(a) for a in srv.possible_actions]
        # for a in possible_actions:
        #     print(a)
        return state_dict, possible_actions


    def _kb_update_cb(self, srv):
        rospy.loginfo("Building knowledge base")
        state_dict, possible_actions = self._process_srv(srv)
        # self.reset()
        self.reset()
        self._gen_action_str_mapping(possible_actions)
        self._check_action_type(possible_actions)
        self._check_equiv_action(possible_actions)
        self._update_state(state_dict)

        rospy.loginfo("Finished kb update!")
        return StateActionUpdateResponse(True)

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
            self._str_to_action_dict[a_str] = str(a)


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
            # self.p.assertz("action({})".format(a_str))
            self.p.query("assertz(action({}))".format(a_str))

            # for action_fact in ACTION_DICT[p_a.predicate]:
            #     self.p.assertz("{fact}({name})".format(fact=action_fact, name=a_str))
            #     print("[check_action_type] Adding fact: {fact}({action_name})".format(fact=action_fact,
            #                                                                           action_name=a_str))

            # Get a list of all the facts for all ingredients paramterizing action
            fact_set_list = [ACTION_DICT[p_a.predicate]] +  \
                [INGREDIENT_DICT[v] for v in p_a.variables]
            # fact_set_list = [a_f for a_f in ACTION_DICT[p_a.predicate]] + fact_set_list
            # action_type = fact_set_list.pop(-1)
            # get th intersection of all the facts
            # Make sure the last set of facts in the fact_set_list are action_type related
            # print(p_a)
            assert len(set(['say', 'do']).intersection(fact_set_list[-1])) > 0
            # Get the facts that are shared in common amongst all action parameters
            try:
                shared_facts = set.intersection(*fact_set_list[:-1])
                # print("action: {}\n\tshared facts: {}".format(p_a,
                #                                               shared_facts))
            except TypeError:
                import pdb; pdb.set_trace()



            # print(p_a)
            # print("action_lists: ",fact_set_list)
            # print("shared_facts: ", shared_facts)
            # print("\n")

            # if there is at least one shared fact, then we add all facts to the kb
            if len(shared_facts) > 0:
                for v in p_a.variables:
                    for action_fact in INGREDIENT_DICT[v]:
                        # print("[check_action_type] Adding fact: {fact}({action_name})".format(fact=action_fact,
                        #                                                                     action_name=a_str))
                        # self.p.assertz("{fact}({action_name})".format(fact=action_fact,
                        #                                             action_name=a_str))
                        if "making" in action_fact and not action_fact in shared_facts:
                            pass
                        else:
                            self.p.query("assertz({fact}({action_name}))".format(fact=action_fact,
                                                                        action_name=a_str))
            # Otherwise the action is weirdly grounded, so we only add the action_type
            # so as to prevent rules from inadvertantly promoting non-relevant actions
            else:
                action_fact = list(fact_set_list[-1])[0]
                # self.p.assertz("{fact}({action_name})".format(fact=action_fact,
                #                                             action_name=a_str))
                self.p.query("assertz({fact}({action_name}))".format(fact=action_fact,
                                                            action_name=a_str))
                # print("[check_action_type] Adding fact: {fact}({action_name})".format(fact=action_fact,
                #                                                                             action_name=a_str))



    def _check_equiv_action(self, possible_actions):
        """
        Checks if two actions have the same effect, and if so
        considers them equivalent and adds this knowledge to the KB
        """

        action_diff = set(["say", "do"])

        # These actions dont have alternate say/do versions
        EXCLUSION_LIST = ["gather", "grabspoon", "boilliquid", "cookoatmeal", "microwavepastry"]
        for a1 in possible_actions:
            a1_name = a1.predicate
            if a1_name in EXCLUSION_LIST:
                continue
            # List of variables parameterizing the action
            a1_vars = set([v for v in a1.variables])
            for a2 in possible_actions:
                a2_name = a2.predicate
                if a2_name in EXCLUSION_LIST:
                    continue
                # List of variables parameterizing the action
                a2_vars = set([v for v in a2.variables])
                # Are the two action types equivalent?
                diff = a1_vars.symmetric_difference(a2_vars)
                # if the actions have the same name and only differ due to the say/do param
                # then they are equivalent.
                if a1_name == a2_name and diff == action_diff:
                    # self.p.assertz("equiv_action({}, {})".format(self._action_to_str_dict[a1],
                    #                                              self._action_to_str_dict[a2]))
                    self.p.query("assertz(equiv_action({}, {}))".format(self._action_to_str_dict[a1],
                                                                 self._action_to_str_dict[a2]))



if __name__ == '__main__':
    rospy.init_node('prolog_interface')
    rate = rospy.Rate(10)
    pc = PrologClassifier()

    while not rospy.is_shutdown():
        rate.sleep()
