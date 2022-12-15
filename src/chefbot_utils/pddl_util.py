#!/usr/bin/env python
import os
from collections import OrderedDict
from pddlgym.core import PDDLEnv
from pddlgym.parser import PDDLDomainParser, PDDLProblemParser
from pddlgym.structs import LiteralConjunction, Not
try:
    from state_transformer import StateTransformer
    # from pddlgym_planners.ff import FF
except ModuleNotFoundError:
    import rospkg
    from chefbot_utils.state_transformer import StateTransformer


try:
    PKG_PATH = rospkg.RosPack().get_path("chefbot")
except Exception as e:
    PKG_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../")

PDDL_DIR = os.path.join(PKG_PATH, "pddl")

ACTION_TEMPLATE_DICT= {
    "gather": "gather({pastry}:pastry,do:action_type)",
    "grab": "grabspoon(mixingspoon:tool,do:action_type)",
    "putinmicrowave": "putinmicrowave({pastry}:pastry,{at}:action_type)",
    "microwavepastry": "microwavepastry({pastry}:pastry,say:action_type)",
    "takeoutmicrowave": "takeoutmicrowave({pastry}:pastry,{at}:action_type)",
    "putinsink" : "putinsink(measuringcup:container,sink:appliance,{at}:action_type)",
    "collectwater": "collectwater(measuringcup:container,water:liquid,{at}:action_type)",
    "pourwater": "pourwater(pan:container,{at}:action_type)",
    "pour": "pour({ingredient}:ingredient,pan:container,{at}:action_type)",
    "turnon": "turnon({appliance}:appliance,{at}:action_type)",
    "boilliquid": "boilliquid(pan:container,{liquid}:liquid,say:action_type)",
    "mix": "mix({liquid}:liquid,pan:container,main:meal,{at}:action_type)",
    "reduceheat": "reduceheat(pan:container,main:meal,{at}:action_type)",
    "serveoatmeal": "serveoatmeal(pan:container,bowl:container,main:meal,{at}:action_type)",
    "cookoatmeal": "cookoatmeal(stove:appliance,main:meal,say:action_type)",
    "checkcereal": "checkcereal(main:meal,say:action_type)",
    "checkfruityoatmeal": "checkfruityoatmeal(pan:container,main:meal,say:action_type)",
    "checkplainoatmeal": "checkplainoatmeal(pan:container,main:meal,say:action_type)",
    "checkchocolateoatmeal": "checkchocolateoatmeal(pan:container,main:meal,say:action_type)" ,
    "checkfruitychocolateoatmeal": "checkfruitychocolateoatmeal(pan:container,main:meal,say:action_type)",
    "checkpeanutbutterchocolateoatmeal":"checkpeanutbutterchocolateoatmeal(pan:container,main:meal,say:action_type)",
    "checkpanutbutterbananaoatmeal": "checkpeanutbutterbananaoatmeal(pan:container,main:meal,say:action_type)"
                       }


CHECK_TEMPLATE_DICT = OrderedDict({
    "checkcereal": "checkcereal(main:meal,say:action_type)",
    "checkfruityoatmeal": "checkfruityoatmeal(bowl:container,main:meal,say:action_type)",
    "checkchocolateoatmeal": "checkchocolateoatmeal(bowl:container,main:meal,say:action_type)" ,
    "checkfruitychocolateoatmeal": "checkfruitychocolateoatmeal(bowl:container,main:meal,say:action_type)",
    "checkpeanutbutterchocolateoatmeal":"checkpeanutbutterchocolateoatmeal(bowl:container,main:meal,say:action_type)",
    "checkpeanutbutterbananaoatmeal": "checkpeanutbutterbananaoatmeal(bowl:container,main:meal,say:action_type)",
    "checkmuffin": "checkmuffin(new:meal,say:action_type)",
    "checkroll": "checkroll(new:meal,say:action_type)",
    "checkjellypastry": "checkjellypastry(new:meal,say:action_type)",
    "checkegg": "checkegg(new:meal,say:action_type)",
    "checkplainoatmeal": "checkplainoatmeal(bowl:container,main:meal,say:action_type)",
    
})


def update_goal(env, state, main, side):

    meal_type = env.domain.types['meal']
    main_pred_high_level  = env.domain.predicates["iscompleted{}".format(main)]
    main_pred_high_level  = main_pred_high_level(meal_type("main"))
    side_pred_high_level  = env.domain.predicates["iscompleted{}".format(side)]
    side_pred_high_level  = side_pred_high_level(meal_type("side"))

    new_goal = LiteralConjunction([main_pred_high_level, side_pred_high_level])

    return state.with_goal(new_goal)

def create_problem_pddl(pddl_dir, env, goal_main, goal_side):
    """
    Generates a problem PDDL file using the goal dishes specified in the input.
    This makes determining reward a bit easier.
    """
    meal_type = env.domain.types['meal']

    if goal_main == "cereal":
        main_pred = env.domain.predicates["iscompletedcereal"]
    else:
        main_pred = env.domain.predicates["is{}".format(goal_main)]
        main_pred_high_level  = env.domain.predicates["iscompletedoatmeal"]
        main_pred_high_level  = main_pred_high_level(meal_type("main"))

    main_pred = main_pred(meal_type("main"))
        # main_str_high_level ="iscompletedoatmeal"

    side_pred = env.domain.predicates["iscompleted{}".format(goal_side)]
    side_pred_high_level = env.domain.predicates["iscompletedpastry"]
    side_pred = side_pred(meal_type("side"))
    side_pred_high_level = side_pred_high_level(meal_type("side"))

    # make tmp_dir if it doesnt exist
    tmp_dir = os.path.join(pddl_dir, "tmp")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    # Delete all the files in the tmp_dir
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))


    # Want to avoid errors (e.g. not mixing multiple liquids in the oatmeal)
    error_pred = env.domain.predicates["error"]
    error_pred = Not(error_pred(meal_type("main")))

    tmp_problem_file = os.path.join(tmp_dir, "tmp_problem.pddl")
    problem_parser = env.problems[0]

    if goal_main == "cereal":
        # goal = LiteralConjunction([main_pred, side_pred, error_pred])
        goal = LiteralConjunction([main_pred, side_pred,
                                   side_pred_high_level, error_pred])
    else:
        goal = LiteralConjunction([main_pred, main_pred_high_level,
                                   side_pred, side_pred_high_level,
                                   error_pred])


    print("Creating problem file {}".format(tmp_problem_file))
    PDDLProblemParser.create_pddl_file(tmp_problem_file,
                                       objects=problem_parser.objects,
                                       initial_state=problem_parser.initial_state,
                                       goal=goal,
                                       domain_name=env.domain.domain_name,
                                       problem_name="tmp_problem",
                                       fast_downward_order=True)

    return tmp_dir

def dist_to_goal(env, state):
    ff_planner = FF()
    NO_PLAN_DIST = -1
    try:
        plan = ff_planner(env.domain, state)
        # for a in plan:
        #     print(a)
        return len(plan)
    except Exception as e:
        # print("Error in planner: {}".format(e))
        return NO_PLAN_DIST


def load_environment(pddl_dir=None, load_feature_env=False, goal_main="plainoatmeal",
                     goal_side="muffin"):
    """
    Load the cookingcheckjellypastry PDDL environment from the corresponding files.
    Also loads the feature PDDL environment which we use to generate the
    fixed length observation vec to train a DQN
    Returns PDDLEnv object
    """

    if pddl_dir == None:
        pddl_dir = PDDL_DIR

    name = "cooking"
    # domain_file = os.path.join(pddl_dir, '{}_domain.pddl'.format(name))
    domain_file = os.path.join(pddl_dir, '{}_domain.pddl'.format(name))
    # feature_domain = os.path.join(pddl_dir, '{}_feature_domain.pddl'.format(name))
    feature_domain = os.path.join(pddl_dir, 'generated_feature_domain.pddl'.format(name))
    problem_template_dir = os.path.join(pddl_dir, name)

    # import pdb; pdb.set_trace()
    template_env  = PDDLEnv(domain_file, problem_template_dir,
                            operators_as_actions=False,
                            dynamic_action_space=True,
                  )
    problem_dir = create_problem_pddl(pddl_dir, template_env, goal_main, goal_side)
    env  = PDDLEnv(domain_file, problem_dir,
                   operators_as_actions=False,
                   dynamic_action_space=True,
                  )

    if load_feature_env:
        feature_env  = StateTransformer(feature_domain, problem_template_dir)
        ret = (env, feature_env)
    else:
        ret = env


    return ret


class FeatureDomainGenerator(object):
    """
    Class for generating a PDDL domain file used to generate
    fixed length feature vector
    """
    def __init__(self, Env, pddl_dir=None):
        self.env = Env
        self._domain_parser = self.env.domain
        self._problem_parser = self.env.problems[0]
        # env.fix_problem_index(0)

        self._feature_str_list = []
        self._get_objects()

        if pddl_dir is None:
            pddl_dir = PDDL_DIR

        self._domain_file = os.path.join(pddl_dir, 'generated_feature_domain.pddl')
        self._domain_str ="""
(define (domain cooking)
      (:requirements :typing)
  (:types
    appliance moveable robot food location action_type - object
    ingredient container tool - moveable
    pastry liquid meal ingredient - food
  )
  (:predicates
  (dummyeffect ?x - object)
  {preds}
  ) ; (:actions {actions})

  {operators}
)
        """


    def save_domain_to_file(self):
        f_str = self._gen_domain_str()
        with open(self._domain_file, 'w+') as f:
            f.write(f_str)

        return self._domain_file


    def _gen_domain_str(self):
        """
        Generates the domain template for the feature domain.
        """
        print("[_gen_domain_template]")
        # Reference: https://github.com/tomsilver/pddlgym/blob/d60feafec14625d3df2f70aff4884e9f1fb51746/pddlgym/parser.py#L378
        preds = "\n\t".join([lit.pddl_str() for lit in self._domain_parser.predicates.values()])
        actions = " ".join(map(str, self._domain_parser.actions))

        return self._domain_str.format(preds=preds,actions=actions,
                                       operators=self._get_all_operators())


    def _get_objects(self):
        self._ingredient_list = []
        self._tool_list  = []
        self._appliance_list = []
        self._container_list = []
        self._bread_list = []
        self._topping_list = []
        self._pastry_list = []
        self._liquid_list = []
        self._completed_meal_list = ['cereal', 'oatmeal']
        # print(env.observation_space.predicates)
        print(self._problem_parser.objects)
        # print(self._problem_parser.types)
        for obj in self._problem_parser.objects:
            # obj is a TypedEntity object
            # if obj.var_type in ["ingredient", "topping", "bread", "pastry"] and not "new" in obj.name:
            if obj.var_type in ["ingredient", "pastry", "liquid"]:
            # if obj.var_type in ["ingredient", "topping", "bread"]:
                # print(obj.name)
                self._ingredient_list.append(obj.name)
            if obj.var_type in ["tool"]:
              self._tool_list.append(obj.name)
            if obj.var_type in ["appliance"]:
              self._appliance_list.append(obj.name)
            if obj.var_type in ["container"]:
              self._container_list.append(obj.name)
            if obj.var_type in ["bread"]:
              self._bread_list.append(obj.name)
            if obj.var_type in ["topping"]:
              self._topping_list.append(obj.name)
            if obj.var_type in ["pastry"]:
                self._pastry_list.append(obj.name)
            if obj.var_type in ["liquid"]:
                self._liquid_list.append(obj.name)



        # TODO: Some objects such as the "new" objects should not be considered here
        # and so should be remoed from the ingredient list.

        print("Ingredients: ", self._ingredient_list)
        # print("Tool list: ", self._tool_list)
        print("Appliances:" , self._appliance_list)
        print("Containers: ", self._container_list)
        print("Toppings: ", self._topping_list)
        print("Breads: ", self._bread_list)
        print("Pastry: ", self._pastry_list)
        print("Liquids: ", self._liquid_list)


    def _get_all_operators(self):
        ret_str = ""
        op_str_funcs = [self._in_storage_str, self._in_worskpace_str,
                        self._in_appliance_str,
                        self._in_microwave_str, self._in_stove_str,
                        self._appliance_on_str,
                        self._is_completed_str, self._is_in_container_str,
                        # self._topping_on_str,
                        self._state_change_str,
                        self._holding_tool_str,
                        self._error_too_many_liquids_in_pan,
                        ]
        # Concatenate all operator strings together
        for f in op_str_funcs:
            ret_str += "".join(f())

        return ret_str


    def _error_too_many_liquids_in_pan(self):
        operator_template = """
        (:action error_milk_and_water_in_pan
          :parameters (?m - liquid ?w - liquid ?p - container)
          :precondition (and (ispan ?p)
                        (iswater ?w) (incontainer ?w ?p)
                        (ismilk ?m) (incontainer ?m ?p)
        )
          :effect (dummyeffect ?m))\n
        """

        return [operator_template]

    def _in_storage_str(self):
        operator_template = """
        (:action {name}_in_storage
          :parameters (?x - object)
          :precondition (and (is{name} ?x) (instorage ?x))
          :effect (dummyeffect ?x))\n
        """
        moveable_objs = self._ingredient_list + self._tool_list + self._container_list + self._liquid_list

        return [operator_template.format(name=obj) for obj in moveable_objs]

    def _in_worskpace_str(self):
        operator_template = """
        (:action {name}_in_workspace
          :parameters (?x - object)
          :precondition (and (is{name} ?x) (inworkspace ?x))
          :effect (dummyeffect ?x))\n
        """
        operator_str_list = []
        moveable_objs = self._ingredient_list + self._tool_list + \
        self._container_list + self._liquid_list

        return [operator_template.format(name=obj) for obj in moveable_objs]


    def _in_appliance_str(self):
        operator_template = """
        (:action {obj}_in_{appliance}
          :parameters (?x - object ?a - appliance)
          :precondition (and (is{obj} ?x) (is{appliance} ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))\n
        """
        # moveable_objs = self._ingredient_list + self._container_list
        # moveable_objs = self._pastry_list + self._bread_list + self._container_list
        moveable_objs = self._container_list
        operator_str_list = []
        for a in self._appliance_list:
            for obj in moveable_objs:
                operator_str = operator_template.format(obj=obj, appliance=a)
                operator_str_list.append(operator_str)
        return operator_str_list

    def _in_stove_str(self):
        operator_template = """
        (:action {container}_in_stove
          :parameters (?x - object ?a - appliance)
          :precondition (and (is{container} ?x) (isstove ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))\n
        """
        # moveable_objns = self._ingredient_list + self._container_list
        # moveable_objs = self._pastry_list + self._bread_list + self._container_list
        moveable_objs = self._container_list
        operator_str_list = []
        for container in moveable_objs:
            operator_str = operator_template.format(container=container)
            operator_str_list.append(operator_str)

        return operator_str_list


    def _in_microwave_str(self):
        operator_template = """
        (:action {pastry}_in_microwave
          :parameters (?x - object ?a - appliance)
          :precondition (and (is{pastry} ?x) (ismicrowave ?a) (inappliance ?x ?a))
          :effect (dummyeffect ?x))\n
        """
        # moveable_objns = self._ingredient_list + self._container_list
        # moveable_objs = self._pastry_list + self._bread_list + self._container_list
        moveable_objs = self._pastry_list
        operator_str_list = []
        for pastry in moveable_objs:
            operator_str = operator_template.format(pastry=pastry)
            operator_str_list.append(operator_str)

        return operator_str_list



    def _topping_on_str(self):
        operator_template = """
        (:action {top}_spread_on_{bread}
          :parameters (?x - topping ?b - ingredient)
          :precondition (and (is{top} ?x) (is{bread} ?b) (isspreadon ?x ?b))
          :effect (dummyeffect ?x))\n
        """

        bread = self._bread_list
        topping = self._topping_list
        operator_str_list = []
        for t in topping:
            for b in bread:
                operator_str = operator_template.format(top=t, bread=b)
                operator_str_list.append(operator_str)

        return operator_str_list

    def _is_in_container_str(self):
        operator_template = """
        (:action {obj}_in_{container}
          :parameters (?x - object ?c - container)
          :precondition (and (is{obj} ?x) (is{container} ?c) (incontainer ?x ?c))
          :effect (dummyeffect ?x))\n
        """

        operator_str_list = []
        obj_list = self._ingredient_list + self._liquid_list + self._completed_meal_list
        for obj in obj_list:
            for container in self._container_list:
                operator_str = operator_template.format(obj=obj, container=container)
                operator_str_list.append(operator_str)


        return operator_str_list


    def _is_completed_str(self):
        # course_names = ['oatmeal', "pastry", "toast"]
        course_names = ['oatmeal', "pastry", 'cereal']

        operator_template = """
        (:action {name}_done
          :parameters (?x - food)
          :precondition (and (iscompleted{name} ?x))
          :effect (dummyeffect ?x))\n
        """

        return [operator_template.format(name=name) for name in course_names]


    def _appliance_on_str(self):

        operator_template = """
        (:action {name}_is_on
          :parameters (?x - appliance)
          :precondition (and (is{name} ?x)(ison ?x))
          :effect (dummyeffect ?x))\n
        """

        return [operator_template.format(name=name) for name in self._appliance_list]

    def _state_change_str(self):
        operator_template = """
        (:action is_{state}
          :parameters (?x - food)
          :precondition (and ({state} ?x))
          :effect (dummyeffect ?x))\n
        """

        # states = ["simmering", "boiling", "heated", "mixed", "warmed", "toasted"]
        states = ["simmering", "boiling", "heated", "mixed", "warmed"]

        return [operator_template.format(state=state) for state in states]


    def _holding_tool_str(self):
        operator_template = """
        (:action holding_{tool}
          :parameters (?x - tool)
          :precondition (and (is{tool} ?x)(inhand  ?x))
          :effect (dummyeffect ?x))\n
        """

        tools  = ["mixingspoon"]

        return [operator_template.format(tool=tool) for tool in tools ]

if __name__ == '__main__':
    env = load_environment(goal_side="jellypastry", goal_main="chocolateoatmeal")
    gen = FeatureDomainGenerator(env)
    gen.save_domain_to_file()
     # state, _ = env.reset()
     # for i in range(10):
     #     action = env.action_space.sample(state)
     #     print("Dist to goal: ", dist_to_goal(env, state))
     #     state, reward, done, info = env.step(action)
