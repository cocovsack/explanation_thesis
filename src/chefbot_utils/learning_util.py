#!/usr/bin/env python

import numpy as np
import os
from itertools import product
from collections import defaultdict

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../models")

TOP_SHELF = ['oats', 'milk', 'bowl']
MID_TOP_SHELF = ['measuringcup', 'jellypastry', 'banana']
MID_SHELF = ['roll', 'cocopuffs', 'peanutbutter']
MID_BOTTOM_SHELF = ['blueberry', 'strawberry', 'pie']
BOTTOM_SHELF = ['muffin', 'egg', 'chocolatechips']

PASTRY = "pastry"
ROLL = "roll"
MUFFIN = "muffin"
JELLYPASTRY = "jellypastry"
PIE = "pie"
EGG = "egg"
CEREAL = "cereal"
MILK = "milk"
WATER = "water"
OATMEAL = "oatmeal"
PLAINOATMEAL = "plainoatmeal"
FRUITYOATMEAL = "fruityoatmeal"
CHOCOLATEOATMEAL = "chocolateoatmeal"
PBBANANAOATMEAL = "peanutbutterbananaoatmeal"
FRUITYCHOCOLATEOATMEAL = "fruitychocolateoatmeal"
PBCHOCOLATEOATMEAL = "peanutbutterchocolateoatmeal"
MAIN_LIST = [CEREAL, OATMEAL, PASTRY]
PASTRY_LIST = [ROLL, MUFFIN, JELLYPASTRY, PIE, EGG]
OATMEAL_LIST = [PLAINOATMEAL, FRUITYOATMEAL, CHOCOLATEOATMEAL, PBBANANAOATMEAL,
             FRUITYCHOCOLATEOATMEAL, PBCHOCOLATEOATMEAL]

CEREAL_INGREDIENTS = set(['bowl', "cocopuffs", "eatingspoon", "milk"])
PLAINOATMEAL_INGREDIENTS = set(['bowl', 'oats', 'measuringcup', 'salt', 'mixingspoon',
                           'stove', "eatingspoon", "measuringcup", "milk", "pan"])
FRUITYOATMEAL_INGREDIENTS = PLAINOATMEAL_INGREDIENTS.union(set(['blueberry', 'strawberry',
                                                            "banana"]))
FRUITYCHOCOLATEOATMEAL_INGREDIENTS = PLAINOATMEAL_INGREDIENTS.union(set(['blueberry',
                                                                      'strawberry',
                                                                      "banana",
                                                                      "chocolatechips"]))

CHOCOLATEOATMEAL_INGREDIENTS = PLAINOATMEAL_INGREDIENTS.union(set(["chocolatechips"]))
PBCHOCOLATEOATMEAL_INGREDIENTS = PLAINOATMEAL_INGREDIENTS.union(set(["chocolatechips",
                                                                     "peanutbutter"]))
PBBANANAOATMEAL_INGREDIENTS = PLAINOATMEAL_INGREDIENTS.union(set(["banana",
                                                                  "peanutbutter"]))


ALL_OATMEAL_INGREDIENTS = FRUITYCHOCOLATEOATMEAL_INGREDIENTS.union(PBBANANAOATMEAL_INGREDIENTS)

ALL_INGREDIENTS = set(PASTRY_LIST).union(ALL_OATMEAL_INGREDIENTS).union(CEREAL_INGREDIENTS)
ALL_CPSC573_PREDS = set(["has_{}".format(i) for i in ALL_INGREDIENTS])

ALL_NUTRITIONAL_PREDS = set(["fruity", "sweet", "gluten", "sodium", "protein", "nonvegan", "dairy", "healthy", "bread"])
ALL_MEAL_PREDS = set(["making_oatmeal", "making_cereal", "making_pastry"])
ALL_ACTION_TYPE_PREDS = set(["do", "say"])

NUTRITIONAL_PREDS_NO_DAIRY = ALL_NUTRITIONAL_PREDS  - set(["dairy"])
ALL_PRECURSOR_PREDS = set(["{}_precursor".format(n) for n in ALL_NUTRITIONAL_PREDS])
PRECURSOR_PREDS_NO_DAIRY = set(["{}_precursor".format(n) for n in NUTRITIONAL_PREDS_NO_DAIRY])

INGREDIENT_DICT = defaultdict(set, {'oats': set(['has_oats', 'making_oatmeal', "main_precursor"]).union(ALL_PRECURSOR_PREDS),
                                    'bowl' :set(['making_oatmeal', 'making_cereal', 'quick', 'object']).union(ALL_PRECURSOR_PREDS),
                                    'water': set(['has_water', 'making_oatmeal']).union(PRECURSOR_PREDS_NO_DAIRY),
                                    'salt' : set(['has_salt', 'making_oatmeal', 'sodium']).union(ALL_PRECURSOR_PREDS),
                                    'mixingspoon': set(['making_oatmeal', 'object', 'quick']).union(ALL_PRECURSOR_PREDS),
                                    'eatingspoon': set(['making_oatmeal', 'making_cereal','quick']).union(ALL_PRECURSOR_PREDS),
                                    'milk': set(['has_milk', 'making_oatmeal','making_cereal',
                                                 'dairy', 'protein', 'nonvegan', "quick", ]).union(ALL_PRECURSOR_PREDS),
                                    'measuringcup': set(['making_oatmeal', 'object']).union(PRECURSOR_PREDS_NO_DAIRY),
                                    'egg' :set(['has_egg', 'making_pastry', 'nonvegan', 'protein', "healthy"]),
                                    'pie': set(['has_pie', 'making_pastry', 'sweet', 'gluten', "fruity"]),
                                    'muffin': set(['has_muffin', 'making_pastry', "healthy", "gluten"]),
                                    'roll': set(['has_roll', 'making_pastry', 'gluten', "bread"]),
                                    "jellypastry": set(["has_jellypastry", "making_pastry", "gluten","sweet", "bread"]),
                                    "microwave": set(["making_pastry", "object"]).union(ALL_PRECURSOR_PREDS),
                                    "stove": set(["making_oatmeal", "object"]).union(ALL_PRECURSOR_PREDS),
                                    "banana": set(["has_banana", "making_oatmeal", "fruity", "sweet", "topping", "healthy"]),
                                    "strawberry": set(["has_strawberry", "making_oatmeal", "fruity", "sweet", "topping", "healthy"]),
                                    "blueberry": set(["has_blueberry", "making_oatmeal", "fruity", "sweet", "topping", "healthy"]),
                                    "peanutbutter": set(["has_peanutbutter", "making_oatmeal", "protein", "topping"]),
                                    "chocolatechips": set(["has_chocolatechips", "making_oatmeal", "sweet",
                                                           "dairy", "nonvegan", "topping"]),
                                    "nuts": set(['has_nuts','making_oatmeal', 'protein']),
                                    "cocopuffs": set(["has_cocopuffs", "making_cereal", "sweet", "gluten", 'quick']),
                                    "say": set(["say"]),
                                    "do": set(["do"]),
                                    "pan": set(["making_oatmeal", "object"]).union(ALL_PRECURSOR_PREDS),
                                    "sink": set(["making_oatmeal", "object"]),
                                    "stove": set(["making_oatmeal", "object"]).union(ALL_PRECURSOR_PREDS),
                                    "main": set(["making_oatmeal", "making_cereal"]).union(ALL_PRECURSOR_PREDS),
                                    "side": set(["making_pastry"]),

                   })


ACTION_DICT = defaultdict(set,
                          {'gather': set(["do_only", "making_oatmeal",
                                          "making_pastry", "making_cereal"]),
                           'pour': set(["making_oatmeal", "making_cereal"]),
                           'collectwater': set(["making_oatmeal"]),
                           'reduceheat': set(["making_oatmeal"]),
                           'mix': set(["making_oatmeal"]),
                           'grabspoon': set(["making_oatmeal","do_only"]),
                           'pourwater': set(["making_oatmeal"]),
                           # 'putinsink': set(["making_oatmeal"]),
                           'turnon': set(["making_oatmeal", "making_pastry"]),
                           'serveoatmeal': set(["making_oatmeal"]),
                           'boilliquid': set(["making_oatmeal", "say_only"]),
                           'cookoatmeal': set(["making_oatmeal", "say_only"]),
                           'putinmicrowave': set(["making_pastry"]),
                           'microwavepastry': set(["making_pastry", "say_only"]),
                           'takeoutmicrowave': set(["making_pastry"]),
                           # 'checkegg': set(["making_pastry"]),
                           # 'checkjellypastry': set(["making_pastry"]),
                           #  'checkpie': set(["making_pastry"]),
                           #  'checkmuffin': set(["making_pastry"]),

                           })


NO_ACTION_TYPE_VARIANTS  = ["boilliquid", "gather", "cookoatmeal", "microwavepastry", "grabspoon"]
# ACTION_DICT = defaultdict(set,
#                           {'gather': set(["do_only", "making_oatmeal",
#                                           "making_pastry", "making_cereal"]).union(ALL_NUTRITIONAL_PREDS),
#                            'pour': set(["making_oatmeal", "making_cereal"]).union(ALL_NUTRITIONAL_PREDS),
#                            'collectwater': set(["making_oatmeal"]).union(NUTRITIONAL_PREDS_NO_DAIRY),
#                            'reduceheat': set(["making_oatmeal",]).union(ALL_NUTRITIONAL_PREDS),
#                            'mix': set(["making_oatmeal"]).union(ALL_NUTRITIONAL_PREDS),
#                            'grabspoon': set(["making_oatmeal","do_only"]).union(ALL_NUTRITIONAL_PREDS),
#                            'pourwater': set(["making_oatmeal"]).union(NUTRITIONAL_PREDS_NO_DAIRY),
#                            # 'putinsink': set(["making_oatmeal"]),
#                            'turnon': set(["making_oatmeal", "making_pastry"]).union(ALL_NUTRITIONAL_PREDS),
#                            'serveoatmeal': set(["making_oatmeal",]).union(ALL_NUTRITIONAL_PREDS),
#                            'boilliquid': set(["making_oatmeal", "say_only"]).union(ALL_NUTRITIONAL_PREDS),
#                            'cookoatmeal': set(["making_oatmeal"]).union(ALL_NUTRITIONAL_PREDS),
#                            'putinmicrowave': set(["making_pastry"]).union(ALL_NUTRITIONAL_PREDS),
#                            'microwavepastry': set(["making_pastry", "say_only"]).union(ALL_NUTRITIONAL_PREDS),
#                            'takeoutmicrowave': set(["making_pastry"]).union(ALL_NUTRITIONAL_PREDS),
#                            # 'checkegg': set(["making_pastry"]),
#                            # 'checkjellypastry': set(["making_pastry"]),
#                            #  'checkpie': set(["making_pastry"]),
#                            #  'checkmuffin': set(["making_pastry"]),

#                            })




def generate_train_test_meals(rng, n_meals=None, split=.7):

    sides = PASTRY_LIST
    oatmeal_mains = [PLAINOATMEAL, FRUITYOATMEAL, CHOCOLATEOATMEAL,
                PBBANANAOATMEAL, FRUITYCHOCOLATEOATMEAL, PBCHOCOLATEOATMEAL]
    cereal_mains = [CEREAL]
    # orders = [("main first", "side last"), ("main last", "side first")]
    orders = ["main first", "side first"]
    liquids = [WATER, MILK]
    main_action_type = ["say", "do"]
    side_action_type = ["say", "do"]

    all_meals = []
    train_list, test_list = [], []

    for o, s, order,l,  m_at, s_at in product(oatmeal_mains, sides, orders, liquids,
                                             main_action_type, side_action_type):
        # m_order, s_order = order
        # all_meals.append({"main":o, "side":s, "main_order":m_order, "side_order": s_order,
        #                   "liquid":l, "action_types": {"main":m_at, "side":s_at}})
        all_meals.append({"main":o, "side":s, "order":order,
                          "liquid":l, "action_types": {"main":m_at, "side":s_at}})
    for c, s, order, m_at, s_at in product(cereal_mains, sides, orders,
                                             main_action_type, side_action_type):
        # m_order, s_order = order
        # all_meals.append({"main":c, "side":s, "main_order":m_order, "side_order": s_order,
        #                   "liquid":"milk", "action_types": {"main":m_at, "side":s_at}})
        all_meals.append({"main":c, "side":s, "order":order,
                          "liquid":"milk", "action_types": {"main":m_at, "side":s_at}})

    print("Total n meals: ", len(all_meals))
    if n_meals is None:
        n_meals = len(all_meals)
    all_meals = rng.choice(all_meals, n_meals, replace=False)
    split_idx = int(n_meals * split)

    return all_meals[:split_idx], all_meals[split_idx:]


if __name__ == '__main__':
    rng = np.random.default_rng(seed=42)
    X, y = generate_train_test_meals(rng)
    print(y)
