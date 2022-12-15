#!/usr/bin/env python
import re
from copy import deepcopy
from pddlgym.structs import Type, TypedEntity
from collections import defaultdict
from scipy.special import softmax
try:
    from state_transformer import PrologShield, ACTION_DICT, INGREDIENT_DICT
    from pddl_util import ACTION_TEMPLATE_DICT
    import learning_util
    from learning_util import PASTRY_LIST

except ModuleNotFoundError:
    from chefbot_utils.state_transformer import PrologShield, ACTION_DICT, INGREDIENT_DICT
    import chefbot_utils.learning_util as learning_util
    from chefbot_utils.learning_util import PASTRY_LIST




class Shield(object):
    """
    Class for creating RL shields. Uses Prolog
    in order to evaluate the shield conditional.

    @input name, str: The name of the field
    @input failiure_cond, str OR PDDLLiteral: The condition that causes the shield to activate
    @input actions, list of strs: list of action(s) that the shield returns
    @input action_space, np.ndarray of PDDLLiterals: The grounded action literals the robot can perform.

    """
    def __init__(self, name, failiure_cond, actions, action_space):
        self.name = name
        self.failiure_cond = failiure_cond
        self.actions = actions
        self.ps = PrologShield(action_space)
        self._shield_type = "shield"

    def info(self):
        return {self.name: {"type": self._shield_type, "shield_when": self.failiure_cond,
                            "alternative_action": self.actions, "complexity": self.n_predicates}}
    @property
    def n_actions(self):
       return len(self.actions)

    @property
    def n_predicates(self):
        """
        Calculates the cyclomatic complexity of the shield which is defined as
        v(G) = P + 1 where P is the number if predicates in the shield.
        """
        operators = ['and', "or", "not"]
        return len([p for p in self.failiure_cond.split() if not p in operators]) + 1


    def __str__(self):
        ret_str = "NAME: {} TYPE: {}\n".format(self.name, self._shield_type)
        ret_str += "if {}:\n".format(self.failiure_cond)
        for a in self.actions:
            ret_str += "\t{}\n".format(a)

        return ret_str



    def build_kb(self, state_dict, action):
        self.ps.build_kb(state_dict, action)


    def shield(self, state_dict, action):
        # import pdb; pdb.set_trace()
        self.build_kb(state_dict, [action])
        res = self.ps.query(self.apply(action))

        print("[Shield] name: {} type: {}".format(self.name, self._shield_type))
        if res:
            ret =  self.actions
        else:
            ret = [action]
        print("[Shield] Shielding action: {} -> {} ? : {}".format(action,
                                                                  self.actions,
                                                                  res))
        # next action is the first action in the list and the one
        # the agent should take next. action_horizon (could be empty)
        # is the list of actions that the agent should take following next_action.
        next_action, action_horizon = ret[0], ret[1:]
        return next_action, action_horizon


class AlternateShield(Shield):
    """
    """
    def __init__(self, name, action_name, shield_dict, action_space):
        # assert not isinstance(action_name, str) and not isinstance(alt_action_type, str), \
        #     print("[Shield] shield name: {}".format(name))

        assert "orig" in shield_dict and "alt" in shield_dict
        fail_action, alt_action = self._format_shield(action_name, shield_dict)
        super(AlternateShield, self).__init__(name, fail_action,
                                              [alt_action],
                                              action_space)
        self._action_name = action_name
        self._shield_dict = defaultdict(set, shield_dict)
        # self_at_dict = {"say": TypedEntity("say"), Type("action_type")}


    def _format_shield(self, action_name, shield_dict):
        orig = shield_dict["orig"]["var"]
        alt = shield_dict["alt"]["var"]
        failiure_action = "{}_{}".format(action_name, orig)
        alt_action = "{}_{}".format(action_name, alt)

        return failiure_action, alt_action

    def shield(self, state_dict, action):
        ret_action = deepcopy(action)
        # import pdb; pdb.set_trace()
        # First check if the name of the action matches the target name.
        if self._action_name in action.predicate.name:

            orig_var = self._shield_dict["orig"]["var"]
            # Check if the other vars defined in the shield_dict are present in the action
            other_vars_present = self._shield_dict["other"].issubset(set([v.name for v in action.variables]))
            # Then check if the item to replaced is present.
            idx_list = [i for i, v in enumerate(action.variables) if orig_var in v.name]
            if len(idx_list) > 0 and other_vars_present:
                idx = idx_list[0]
                alt_lit = TypedEntity(self._shield_dict["alt"]["var"],
                                      Type(self._shield_dict["alt"]["type"]))

                ret_action.update_variable(idx, alt_lit)
                print("[shield] Shielding: {} --> {}".format(action, ret_action))


        return ret_action, []


class RefineShield(Shield):
    def __init__(self, name, failiure_cond, actions, action_space):
        actions = self._str_to_literal(actions, action_space)
        super(RefineShield, self).__init__(name, failiure_cond, actions, action_space)
        self._query_dict = {"refine": self._process_expr(failiure_cond)}
        self._shield_type = "refine"

    def apply(self, action):
        return self._query_dict

    def _str_to_literal(self, actions, action_space):
        """
        Convert actions which are strs to PDDL literals
        """
        action_space_str = [str(a) for a in action_space]
        ret_actions = []
        for a in actions:
            if a in action_space_str:
                a_idx = action_space_str.index(a)
                ret_actions.append(action_space[a_idx])

        return ret_actions


    def _process_expr(self, expr):
        """
        Converts input rule written using pythonic logic
        to prolog equivalent.

        @input expr, str: a IFTTT rule of the form precond then postcond where poscond
        MUST be some valid lifted predicate

        returns str, the prolog version of the input.
        """

        operators = ['and', 'or', 'not', 'then']
        print("input rule: ", expr)
        # if the postcondition is negated then prolog will never return groundings
        # so we remove the negation here and handle the logic of negation via
        # the instantiation of "prohibitive overlay"."

        # Surround only predicates with "state(<predicate>)"
        expr_list = expr.split(' ')
        new_expr_list = []
        for e in expr_list:
            if e in operators:
                new_expr_list.append(e)
            else:
                new_expr_list.append('state({})'.format(e))

        expr = ' '.join(new_expr_list)
        # replace 'and' with ';'
        expr = re.sub(r'\band\b', ',', expr)
        # replace 'or' with ','
        expr = re.sub(r'\bor\b', ';', expr)
        # replace 'not' with '\+'
        expr = re.sub(r'\bnot\b', '\+', expr)
        # remove everything after 'then'
        # expr = re.sub(r'\bthen\s+.*', '', expr)
        expr = re.sub(r'\bthen\b', '->', expr)

        return expr

class ForbidShield(RefineShield):
    def __init__(self, name, failiure_cond, action_space):
        super(ForbidShield, self).__init__(name, failiure_cond, [], action_space)
        self._query_dict = {"forbid": self._process_expr(failiure_cond)}
        self._shield_type = "forbid"

    def _process_expr(self, expr):

        expr = re.sub(r'\band\b', ',', expr)
        # replace 'or' with ','
        expr = re.sub(r'\bor\b', ';', expr)
        # replace 'not' with '\+'
        expr = re.sub(r'\bnot\b', '\+', expr)
        # remove everything after 'then'
        # expr = re.sub(r'\bthen\s+.*', '', expr)
        expr = re.sub(r'\bthen\b', '->', expr)

        return expr

    def shield(self, state_dict, action, action_space):
        self.build_kb(state_dict, [])
        res = self.ps.query(self.apply(action))
        a_idx = action_space.index(action)
        new_prob = 1.0 # This value will ultimately be used to update the value of policy

        if res: # Then the shield condition is true, so we should shield action
            new_prob = 0.0
        # print("[shield] Shielding: {} ? {} new prob: {}".format(action, res, new_prob))
        return a_idx, new_prob




class ShieldFactory(object):
    def __init__(self, action_space):
        "Class for more easily generating shields"
        self.action_space = action_space

    def pastry_forbid_shield(self, desired_pastry):
        name="forbid_not_{}".format(desired_pastry)
        failiure_conds = ["{}_in_workspace".format(p) for p in PASTRY_LIST \
                         if not p is desired_pastry]
        failiure_conds = " or ".join(failiure_conds)

        return ForbidShield(name, failiure_conds, self.action_space)

    def pastry_refine_shield(self, failiure_cond, alt_pastry, action_type):
        name="pastry_refine_shield"
        actions_template = [
                         "gather({pastry}:pastry,do:action_type)",
                         "putinmicrowave({pastry}:pastry,{at}:action_type)",
                         "turnon(microwave:appliance,{at}:action_type)",
                         "microwavepastry({pastry}:pastry,say:action_type)",
                         "takeoutmicrowave({pastry}:pastry,{at}:action_type)",
                         "check{pastry}(side:meal,say:action_type)"]

        actions = [a.format(pastry=alt_pastry,
                            at=action_type) for a in actions_template]

        return RefineShield(name,
                            failiure_cond=failiure_cond,
                            actions=actions,
                            action_space=self.action_space)



    def milk_to_water_refine_shield(self, failure_cond, action_type):
        name = "water_refine_shield"
        action_template = [
            "gather(measuringcup:container,do:action_type)",
            # "putinsink(measuringcup:container,sink:appliance,{at}:action_type)",
            "collectwater(measuringcup:container,water:liquid,{at}:action_type)",
            "pourwater(pan:container,{at}:action_type)"
        ]
        actions = [a.format(at=action_type) for a in action_template]

        return RefineShield(name,
                            failiure_cond=failiure_cond,
                            actions=actions,
                            action_space=self.action_space)

    def alternate_action_type(self, name, action):
        SAY = "say"
        DO = "do"
        if SAY in action:
            alt_action = re.sub(",say:", ",do:", action)
        elif DO in action:
            alt_action = re.sub(",do:", ",say:", action)

        return AlternateShield(name,
                               alt_action=alt_action,
                               failiure_action=action,
                               action_space=self.action_space
                               )

    def gather_pastry_alternate(self, orig_item, alt_item):
        name="{}_to_{}_alternate".format(orig_item, alt_item)
        orig_dict = {"var":orig_item, "type": "pastry"}
        alt_dict = {"var":alt_item, "type": "pastry"}
        shield_dict = {"orig": orig_dict, "alt":alt_dict}
        return AlternateShield(name, "gather", shield_dict, self.action_space)

    def _substitute_item_shield(self, name, item_type, orig_action, alt_item):
        regex = r"(?<=\().+?(:{item_type})".format(item_type=item_type)
        ret_action = re.sub(regex, alt_item,
                            orig_action)
        return AlternateShield(name,
                               alt_action=ret_action,
                               failiure_action=orig_action,
                               action_space=self.action_space)

    def alternate_pastry(self, name, orig_action, alt_pastry):
        """
        substitute the pastry preceding ':pastry' sub strin in orig_action with alt_pastry
        """
        return self._substitute_item_shield(name, "pastry", orig_action,
                                            alt_pastry)

    def alternate_action_type_by_meal(self, meal, alt_action_type):
        orig_action_type = "do" if alt_action_type == "say" else "say"
        orig_dict = {"var": orig_action_type, "type": "action_type"}
        alt_dict = {"var": alt_action_type, "type": "action_type"}
        ret_shields = []

        cereal_actions = [("pour", {"orig":orig_dict, "alt":alt_dict})]
        oatmeal_actions = [
            ("pour", {"orig":orig_dict, "alt":alt_dict}),
            ("mix", {"orig":orig_dict, "alt":alt_dict}),
            ("turnon", {"orig":orig_dict, "alt":alt_dict, "other": set("stove")}),
            ("reduceheat", {"orig":orig_dict, "alt":alt_dict}),
            ("serveoatmeal", {"orig":orig_dict, "alt":alt_dict}),
        ]
        pastry_actions = [
            ("putinmicrowave", {"orig":orig_dict, "alt":alt_dict}),
            ("turnon", {"orig":orig_dict, "alt":alt_dict, "other": set("microwave")}),
            ("takeoutmicrowave", {"orig":orig_dict, "alt":alt_dict}),
        ]

        if meal == "cereal":
            action_list = cereal_actions
        elif meal == "oatmeal":
            action_list = oatmeal_actions
        elif meal == "pastry":
            action_list = pastry_actions


        for action, d in action_list:
            name ="{}_{}_to_{}".format(action, d["orig"]["var"], d["alt"]["var"])
            ret_shields.append(AlternateShield(name, action, d, self.action_space) )

        return ret_shields


    def forbid_meal_from_order(self, first_meal, last_meal):
        EXCLUDE_ING_SET = set(["pan", "stove", "sink"])
        ALL_MAIN_INGS = learning_util.CEREAL_INGREDIENTS.union(learning_util.ALL_OATMEAL_INGREDIENTS)
        # dish_preds_dict = {'cereal': (set(PASTRY_LIST), ['microwave'] , ['cereal_done', 'oatmeal_done']),
        #                          'oatmeal': (set(PASTRY_LIST), ['microwave'] , ['cereal_done', 'oatmeal_done']),
        #                          'pastry': (ALL_MAIN_INGS, ['stove'], ['pastry_done'])
        #                    }
        dish_preds_dict = {'cereal': (learning_util.CEREAL_INGREDIENTS, ['microwave'] , ['cereal_done']),
                                 'oatmeal': (learning_util.ALL_OATMEAL_INGREDIENTS, ['stove'] , ['oatmeal_done']),
                                 'pastry': (set(PASTRY_LIST), ['microwave'], ['pastry_done'])
                           }

        print("first: ", first_meal)
        print("last: ", last_meal)
        if first_meal in PASTRY_LIST:
            first = "pastry"
        else:
            first = [k for k in dish_preds_dict if k in first_meal][0]
        if last_meal in PASTRY_LIST:
            last = "pastry"
        else:
            last = [k for k in dish_preds_dict if k in last_meal][0]

        first_ing_list, first_appliance_list, first_meal_preds = dish_preds_dict[first]
        last_ing_list, last_appliance_list, _ = dish_preds_dict[last]
        first_ing_list -= EXCLUDE_ING_SET
        last_ing_list -= EXCLUDE_ING_SET


        # first_failiure_conds = ["state({}_in_workspace)".format(i) for i in first_ing_list]
        # first_failiure_conds += ["state({}_is_on)".format(a) for a in first_appliance_list]
        # first_failiure_conds = " or ".join(first_failiure_conds)
        # first_meal_preds = ["state({})".format(p) for p in first_meal_preds]
        # first_meal_preds = " or ".join(first_meal_preds)

        # last_failiure_conds = ["state({}_in_workspace)".format(i) for i in last_ing_list]
        # last_failiure_conds += ["state({}_is_on)".format(a) for a in last_appliance_list]
        # last_failiure_conds = " or ".join(last_failiure_conds)
        first_failiure_conds = ["state({}_in_workspace)".format(i) for i in last_ing_list]
        first_failiure_conds += ["state({}_is_on)".format(a) for a in last_appliance_list]
        first_failiure_conds = " or ".join(first_failiure_conds)
        first_meal_preds = ["state({})".format(p) for p in first_meal_preds]
        first_meal_preds = " or ".join(first_meal_preds)

        last_failiure_conds = ["state({}_in_workspace)".format(i) for i in first_ing_list]
        last_failiure_conds += ["state({}_is_on)".format(a) for a in first_appliance_list]
        last_failiure_conds = " or ".join(last_failiure_conds)
        failiure_conds = "(({first}) and (not ({first_pred}))) or (({last}) and ({first_pred}))".format(first=first_failiure_conds,
                                                                                                        first_pred=first_meal_preds,
                                                                                                        last=last_failiure_conds)

        # if last_meal == "cereal":
        #     first_meal_preds = "pastry_done"
        #     ing_list = learning_util.CEREAL_INGREDIENTS.union(learning_util.ALL_OATMEAL_INGREDIENTS)
        #     appliance_list = ["stove"]
        #     ing_list = ing_list - EXCLUDE_ING_SET
        # # elif  last_meal == "oatmeal":
        # elif  "oatmeal" in last_meal:
        #     first_meal_preds = "pastry_done"
        #     ing_list = learning_util.ALL_OATMEAL_INGREDIENTS.union(learning_util.CEREAL_INGREDIENTS)
        #     appliance_list = ["microwave"]
        #     ing_list = ing_list - EXCLUDE_ING_SET
        # else:
        #     if "oatmeal" in first_meal:
        #         first_meal_preds = "oatmeal_done"
        #         ing_list = PASTRY_LIST
        #         # ing_list = learning_util.ALL_OATMEAL_INGREDIENTS
        #         # ing_list = ing_list - EXCLUDE_ING_SET
        #     elif first_meal == "cereal":
        #         first_meal_preds = "cereal_done"
        #         ing_list = PASTRY_LIST

        # failiure_conds = ["state({}_in_workspace)".format(i) for i in ing_list]
        # failiure_conds = " or ".join(failiure_conds)
        # failiure_conds = "({}) and (not state({}))".format(failiure_conds, first_meal_preds)

        print("forbid: ", failiure_conds)
        # import pdb; pdb.set_trace()
        name = "forbid_{}_first".format(last_meal)
        return [ForbidShield(name, failiure_conds, self.action_space)]

        # if last_meal == learning_util.PLAINOATMEAL:
        #     ingredients = learning_util.PLAINOATMEAL_INGREDIENTS
        # elif last_meal == learning_util.FRUITYOATMEAL:
        #     ingredients = learning_util.FRUITYOATMEAL_INGREDIENTS
        # elif last_meal == learning_util.FRUITYOATMEAL:
        #     ingredients = learning_util.FRUITYOATMEAL_INGREDIENTS
        # elif last_meal == learning_util.CHOCOLATEOATMEAL:
        #     ingredients = learning_util.CHOCOLATEOATMEAL_INGREDIENTS
        # elif last_meal == learning_util.PBBANANAOATMEAL:
        #     ingredients = learning_util.PBBANANAOATMEAL_INGREDIENTS
        # elif last_meal == learning_util.FRUITYCHOCOLATEOATMEAL:
        #     ingredients = learning_util.FRUITYCHOCOLATEOATMEAL_INGREDIENTS
        # elif last_meal == learning_util.PBCHOCOLATEOATMEAL:
        #     ingredients = learning_util.PBCHOCOLATEOATMEAL_INGREDIENTS



    def alternate_liquid(self, alt_liquid):
        """
        Substitute gathering milk/water with water/milk
        """
        action = "gather"
        if alt_liquid == "water":
            orig_liquid = "milk"
            orig_type = "liquid"
            alt_liquid = "measuringcup"
            alt_type = "container"
        elif alt_liquid == "milk":
            orig_liquid = "measuringcup"
            orig_type = "container"
            alt_liquid = "milk"
            alt_type = "liquid"

        shield_dict = {"orig": {"var":orig_liquid, "type": orig_type},
                       "alt": {"var":alt_liquid, "type": alt_type}}
        name = "{}_to_{}_alternate".format(orig_liquid, alt_liquid)
        return AlternateShield(name, action, shield_dict,
                               self.action_space)
        # milk_action = "gather(milk:liquid,do:action_type)"
        # water_action = "gather(measuringcup:container,do:action_type)"
        # if alt_liquid == "milk":
        #     sheiled_action = water_action
        #     alt_action = milk_action
        #     name = "water_to_milk_alternate"
        # elif alt_liquid == "water":
        #     sheiled_action = milk_action
        #     alt_action = water_action
        #     name = "milk_to_water_alternate"

        # return AlternateShield(name,
        #                        alt_action=alt_action,
        #                        failiure_action=sheiled_action,
        #                        action_space=self.action_space
        #                        )

    def alternate_action_type_pastry(self, alt_action_type, pastry):
        orig_action_type = "do" if alt_action_type == "say" else "say"
        pastry_actions = [
            # "gather({pastry}:pastry,do:action_type)",
            "putinmicrowave({pastry}:pastry,{at}:action_type)",
            "turnon(microwave:appliance,{at}:action_type)",
            # "microwavepastry({pastry}:pastry,say:action_type)",
            "takeoutmicrowave({pastry}:pastry,{at}:action_type)",
            # "check{pastry}(side:meal,say:action_type)"
        ]
        pastry_actions = [p.format(pastry=pastry,at=alt_action_type) for p in pastry_actions]
        return self.alternate_action_type_from_list(pastry_actions)




    def alternate_action_type_from_list(self, action_list):
        """
        Create alt_shields for action_type given list of actions
        @input action_list, [str, str,...,str] list of str PDDL literals
        Returns a list of action shields for each action in the list.
        """
        ret_list = []
        for i, orig_action in enumerate(action_list):
            name = "alt_a_t_shield_{}".format(i)
            ret_list.append(self.alternate_action_type(name,
                                                       orig_action)
                            )
        return ret_list

    def alternate_gather_from_list(self, item_dict_list):
        ret_list = []
        for i, item_dict in enumerate(item_dict_list):
            name = "alt_gather_shield_{}".format(i)
            ret_list.append(
                self.gather_alternate(name,
                                      item_dict['orig'],
                                      item_dict['alt'])
            )

        return ret_list




if __name__ == '__main__':

    sf = ShieldFactory()
    # f._filter_by_predicates(["sweet", "making_oatmeal"], ["protien"], "OR")
