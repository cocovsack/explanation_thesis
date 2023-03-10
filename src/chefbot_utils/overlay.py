#!/usr/bin/env python

import re
import json

from sympy.parsing.sympy_parser import parse_expr
from sympy import true, false, sympify
from sympy.logic.boolalg import ITE

try:
    from state_transformer import PrologClassifier
except ModuleNotFoundError:
    import rospy
    import ast
    from chefbot_utils.state_transformer import PrologClassifier
    from chefbot.srv import PrologQuery, PrologQueryRequest, PrologQueryResponse
    from chefbot.srv import StateActionUpdate, StateActionUpdateRequest, StateActionUpdateResponse





# 
PERMIT_OVERLAY = "permit"
PROHIBIT_OVERLAY = "prohibit"
TRANSFER_OVERLAY = "transfer"

class Overlay(object):
    """
    Expects a logical rule of the form A --> B where A is a boolean expression
    containing & (and) | (or), ~ (not) operators and arbitrary number of variables and
    B is a (possibly negated variable) 

    NOTE: In practice we use this class as an AbstractClass.

     @name: str, the name of the overlay
     @precond: str: a boolean expression
     @postcond: str: a boolean expression of the form ~b or b where b is a single variable
    """
    def __init__(self, name, expr, overlay_type):
        self.name = name
        self.expr = expr
        self.rule_is_satisfied = False # Is the rule satisfied in this state?
        self.is_active = True # Is the overlay currently being used
        self.overlay_type = overlay_type

        if overlay_type == PERMIT_OVERLAY:
            self.padding =  0.0
            self.coeff = 1.0
        elif overlay_type == PROHIBIT_OVERLAY:
            # self.padding = -1e5
            self.padding = 0
            self.coeff = 0.0
        elif overlay_type == TRANSFER_OVERLAY:
            self.coeff = 1.0
            self.padding = 0


    def eval(self, var_dict):
        """

        """
        print("[eval] expr: {} dict: {}".format(self.expr, var_dict))
        res = parse_expr(self.expr, local_dict=var_dict)
        # print("[eval] type: ", type(res))
        if res == True:
            self.rule_is_satisfied = True
        else:
            self.rule_is_satisfied = False

        # print("[eval] res ", res)
        # print("[eval] rule_is_satisfied ", self.rule_is_satisfied)
        # Equivalent to if res then postcond else do nothing

    def __str__(self):
        return "{}: {}: {}".format(self.name, self.expr,
                                   self.rule_is_satisfied)

class OverlayList(object):
    """
    A class for storing and evaluating multiple overlays
    """

    def __init__(self, overlay_list):
        self._overlay_list = self._order_overlays(overlay_list)

    def eval(self, var_dict):
        # Need to convert booleans to sympy bools for expression to evaluate.
        sympy_dict = {k:sympify(v) for k, v in var_dict.items()}
        prop_dict = {}
        for ov in self._overlay_list:
            if ov.rule_is_satisfied:
                ov.eval(sympy_dict)
                prop_dict[ov.name] = ov

        return prop_dict


    def pretty_print(self, var_dict):
        for prop in self._overlay_list:
            print(str(prop) + " --> " + str(prop.eval(var_dict)))


    def append(self, prop):
        # Assert prop is a Overlay
        assert isinstance(prop, Overlay)
        print("[append] Adding overlay: ", prop)
        self._overlay_list.append(prop)
        self._overlay_list = self._order_overlays(self._overlay_list)

    def __str__(self):
        ret_str = ""
        for prop in self._overlay_list:
            ret_str += str(prop) + "\n"

        return ret_str

    def _order_overlays(self, overay_list):
        """"
        Organizes overlays such that permissive overlays come first and 
        prohibitive overlays last. This ordering ensures that the latter
        have the 'final say.
        """

        permissive_overlays = []
        prohibitive_overlays = []

        for o in overay_list:
            if o.overlay_type == PERMIT_OVERLAY:
                permissive_overlays.append(o)
            else:
                prohibitive_overlays.append(o)

        return permissive_overlays + prohibitive_overlays

    def __iter__(self):
        return iter(self._overlay_list)

    def __next__(self):
        return next(self._overlay_list)

    def __getitem__(self, key):
        return self._overlay_list[key]

    def __len__(self):
        return len(self._overlay_list)

class PrologOverlay(Overlay):
    """
    A version of the overlay that interfaces with prolog.
    This enables rules to constructed in predicate logic
    rather than just boolean logic.

    @input name, str: The name of the overlay
    @input expr, str: A predicate logic expression of the form

    "precond(X,Y,..) then postcond(X,Y,...) "

    Where precond is a clause written in predicate logic with grounded and lifted predicates
    and operators and, or, and not, and postcond is a single, positive literal.
    If the precond is satisfied in the current state then the overlay will
    return a list of groundings that satisfy the rule.

    """
    def __init__(self, name, in_expr, overlay_type=None):
        self._in_expr = in_expr
        # convert in_expr into prolog_expr
        prolog_expr, derived_overlay_type = self._process_expr(in_expr)
        if overlay_type is None:
            overlay_type = derived_overlay_type
        super(PrologOverlay, self).__init__(name, prolog_expr, overlay_type)
        self._res = None

    @property
    def res(self):
        return self._res

    @res.setter
    def res(self, res):
        self._res = res

    def eval(self):
        pass

    def reset(self):
        self._res = None
        self.rule_is_satisfied = False

    @property
    def n_predicates(self):
        """
        Calculates the cyclomatic complexity of the overlay which is defined as
        v(G) = P + 1 where P is the number if predicates in the overlay.
        """
        operators = [",", ";", "->", "\+"]

        return len([p for p in self.expr.split() if not p in operators])  + 1



    def info(self):
        return {self.name: {"rule": self.expr, "type": self.overlay_type, "complexity": self.n_predicates}}

    def _process_expr(self, expr):
        """
        Converts input rule written using pythonic logic
        to prolog equivalent.

        @input expr, str: a IFTTT rule of the form precond then postcond where poscond
        MUST be some valid lifted predicate

        returns str, the prolog version of the input.
        """
        overlay_type = "permit"

        print("input rule: ", expr)
        # if the postcondition is negated then prolog will never return groundings
        # so we remove the negation here and handle the logic of negation via 
        # the instantiation of "prohibitive overlay"."
        if bool(re.search(r'then\s+not', expr)):
            expr = re.sub(r'then\s+not', 'then', expr)
            overlay_type = "prohibit"
        # replace 'and' with ';'
        expr = re.sub(r'\band\b', ',', expr)
        # replace 'or' with ','
        expr = re.sub(r'\bor\b', ';', expr)
        # replace 'not' with '\+'
        expr = re.sub(r'\bnot\b', '\+', expr)
        # remove everything after 'then'
        # expr = re.sub(r'\bthen\s+.*', '', expr)
        expr = re.sub(r'\bthen\b', '->', expr)
        print("output rule: {} type: {}".format(expr, overlay_type))

        return expr, overlay_type



class PrologOverlayList(OverlayList):
    """
    The interface between the overlays and prolog.
    """
    def __init__(self, overlay_list):
        super(PrologOverlayList, self).__init__(overlay_list)
        print("Inside PrologOverlays!")
        self.pc = PrologClassifier()
        print("Finished constructing")

    @property
    def n_predicates(self):
        """
        Calculates the cyclomatic complexity of all the overlays
        """
        if len(self._overlay_list) == 0:
            return 1
        else:
            return sum([ov.n_predicates for ov in self._overlay_list]) + 1

    def build_kb(self, state_dict, possible_actions):
        """
        @input state_dict, dict of bools: Each entry represents a feature in the
                                          state space (see StateTransformer)
        @input possible_actions, np.ndarray of PDDLLiterals: The grounded action
                                                             literals the robot can perform.
        """
        # self.pc.reset()
        self.pc.build_kb(state_dict, possible_actions)


    def eval(self, state_dict, possible_actions):
        self.build_kb(state_dict, possible_actions)
        self.reset()

        for ov in self._overlay_list:
            # import pdb; pdb.set_trace()
            ov.res = self.pc.query(ov.expr)
            # If ov.res is empty then rule is not satisfied
            self.rule_is_satisfied = len(ov.res) > 0


    def reset(self):
        for ov in self._overlay_list:
            ov.reset()

class RosOverlayList(OverlayList):
    # ""
    # The interface between the overlays and prolog and ROS.
    # ROS and swi-prolog dont play nice together so we need to
    # implement ROS specific versions of prolog stuff.
    # """
    def __init__(self, overlay_list):
        super(RosOverlayList, self).__init__(overlay_list)
        rospy.wait_for_service("chefbot/prolog/query")
        self._prolog_query_sp = rospy.ServiceProxy("chefbot/prolog/query", PrologQuery)
        rospy.wait_for_service("chefbot/prolog/kb_update")
        self._kb_update_sp = rospy.ServiceProxy("chefbot/prolog/kb_update", StateActionUpdate)

    @property
    def n_predicates(self):
        """
        Calculates the cyclomatic complexity of all the overlays
        """
        if len(self._overlay_list) == 0:
            return 1
        else:
            return sum([ov.n_predicates for ov in self._overlay_list]) + 1

    def build_kb(self, state_dict, possible_actions):
        """
        @input state_dict, dict of bools: Each entry represents a feature in the
                                          state space (see StateTransformer)
        @input possible_actions, np.ndarray of PDDLLiterals: The grounded action
                                                             literals the robot can perform.
        """
        # self.pc.reset()
        self.pc.build_kb(state_dict, possible_actions)


    def eval(self, state_dict, possible_actions):
        self.reset()
        state_features = []
        bools = []

        action_str_to_lit_dict = {str(a):a for a in possible_actions}

        for s, b in state_dict.items():
            state_features.append(s)
            bools.append(b)

        req = StateActionUpdateRequest()
        req.state_features = state_features
        req.state_bools = bools
        req.possible_actions = [str(a) for a in possible_actions]

        update_resp = self._kb_update_sp(req)
        rospy.loginfo("Sending query over...")
        rospy.loginfo(update_resp)
        if update_resp.success:
            for ov in self._overlay_list:
                if ov.is_active:
                    query_req = PrologQueryRequest()
                    query_req.query = ov.expr
                    print(ov.expr)
                    resp = self._prolog_query_sp(query_req)
                    ov.res = self._process_query_resp(resp.resp, action_str_to_lit_dict)
                    print("query results: ", ov.res)
                    # import pdb; pdb.set_trace()
                    # ov.res = self.pc.query(ov.expr)
                    # If ov.res is empty then rule is not satisfied
                    ov.rule_is_satisfied = len(ov.res) > 0
                else:
                    print("[ROSOverlays] Skipping Overlay: {}, not active!".format(ov))
                    ov.res = []


    def _process_query_resp(self, resp, action_dict):
        """
        Convert the srv response of querying prolog into a list of actions.

        @input resp list of strs: list of strs representations of dictionaries.
        @input action_dict, {str:PDDLLiteral, ...}: dict mapping str to their PDDLLiteral
                                                     representation

        Returns list of actions.
        """
        # return [json.loads(d_str) for d_str in resp]
        ret_dict_list = []
        # Convert from str rep to dict
        dict_list =[ast.literal_eval(d_str) for d_str in resp]
        for d in dict_list:
            # convert action str to action Literal
            new_dict = {k:action_dict[a] for k, a in d.items()}
            ret_dict_list.append(new_dict)
        return ret_dict_list

    def reset(self):
        for ov in self._overlay_list:
            ov.reset()



class OverlayFactory(object):
    state_temp = "state({state})"
    dish_temp = "making_{dish}(A_out)"
    pred_temp = "({pred}(A_out) or {pred}_precursor(A_out))"
    order_dict = {
        "first": "no_completed_dish",
        "last": "one_completed_dish",
        "all": "two_completed_dish"
    }

    def _get_name(self, name_list):
        ret_list = []
        for i in name_list:
            if i is None:
                pass
            else:
                ret_list += i
        # name = "_".join([i for i in [dish, order, preds]if not i is None])
        return "_".join(ret_list)

    def _get_precond(self, state_list):
        state_pred = [self.order_dict[o] for o in state_list] if not state_list is None else []
        state_pred = [self.state_temp.format(state=s) for s in state_pred]

        return " and ".join(state_pred)

    def make_meal_overlay(self, dish=None, order=None, preds=None, logical_op="and"):
        # state_pred = "no_completed_dish" if order == "first" else "one_completed_dish"
        assert dish is None or isinstance(dish, list)
        assert order is None or isinstance(order, list)
        assert preds is None or isinstance(preds, list)

        precond = self._get_precond(order)
        # dish_preds = self.dish_temp.format(dish=dish) if not dish is None else []
        dish_preds = [self.dish_temp.format(dish=m) for m in dish] if not dish is None else []
        other_preds  = [self.pred_temp.format(pred=p) for p in preds] \
            if not preds is None else []
        logical_join = " {} ".format(logical_op)
        postcond = logical_join.join(dish_preds + other_preds).strip(" and ")

        rule = "{precond} then ({postcond})".format(precond=precond,
                                                  postcond=postcond)
        name = self._get_name([dish, preds, order])

        return PrologOverlay(name, rule)

    def action_type_assignment_overlays(self, action_types, main, side):
        assert main in ["pastry", "cereal", "oatmeal"]
        assert side in ["pastry", "cereal", "oatmeal"]
        precond = self._get_precond(["all"])

        DO_TEMP = "do(A_out) and making_{dish}(A_out)"
        SAY_TEMP = "say(A_out) and making_{dish}(A_out)"
        # rule_template = "not {precond} then (equiv_action(A_in, A_out) and (({pm}) or ({ps})))"
        rule_template = "not {precond} then  (({pm}) or ({ps}))"
        if action_types["main"] == "say":
            main_temp = SAY_TEMP
        else:
            main_temp = DO_TEMP

        if action_types["side"] == "say":
            side_temp = SAY_TEMP
        else:
            side_temp = DO_TEMP

        pm = main_temp.format(dish=main)
        ps = side_temp.format(dish=side)
        rule = rule_template.format(precond=precond, pm=pm, ps=ps)
        name = "{}_main_{}_side".format(action_types["main"], action_types["side"])

        # return [PrologOverlay(name, rule, overlay_type="transfer")]
        return [PrologOverlay(name, rule, overlay_type="permit")]

    def change_action_type_overlay(self, order=None, action_type="do", dish=None):
        # assert dish is None or isinstance(dish, str)
        assert dish is None or dish in ["pastry", "cereal", "oatmeal"]
        assert order is None or isinstance(order, list)
        assert isinstance(action_type, str)

        precond = self._get_precond(order)
        dish_pred = "" if dish is None else self.dish_temp.format(dish=dish)

        orig_action_type = "do" if action_type == "say" else "say"
        name = "_".join([action_type, dish] + order)
        rule_template = "{precond} then {dish_pred} and equiv_action(A_in, A_out) and {new}(A_out)"

        return PrologOverlay(name, rule_template.format(precond=precond,
                                                        orig=orig_action_type,
                                                        new=action_type,
                                                        dish_pred=dish_pred),
                             overlay_type="transfer")

    def forbid_overlay(self, to_forbid_list, order="first"):
        precond = self._get_precond(order)
        name = self._get_name(to_forbid_list)
        name = "forbid_{}".format("_".join(name))
        postcond = " and ".join([self.pred_temp.format(pred=f) for f in to_forbid_list])
        rule_template = "{precond} then not {postcond}"

        return PrologOverlay(name, rule_template.format(precond=precond,
                                                        postcond=postcond),
                             overlay_type="forbid")
if __name__ == '__main__':
    rule_str = "dairy(A_out) then not action(A_out)"



