import numpy as np
import gym

from stable_baselines.common.env_checker import check_env
from stable_baselines import ACER
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.gail import generate_expert_traj

from gym import spaces


# Experimental code to try and get code working with AI Gym
# In some sense this would be ideal because stable_baselines
# has a bunch of nice models already pre implemented

class DiscreteWrapper(spaces.Discrete):
    """
    Wrapper around AI gyms discrete action space.
    This was done to leverage the fact that the pddlenv can easily sample
    only viable actions in current state rather than actions across the entire (large)
    state space.

    """
    def __init__(self, n, pddlenv, action_space_list):
        super(DiscreteWrapper, self).__init__(n)
        self.pddlenv = pddlenv
        self._action_space_list = action_space_list
        self.planner = FF()


    def sample(self):
        action = self.pddlenv.action_space.sample(obs)
        return self._action_space_list.index(action)



class CookingEnv(gym.Env):
    """
    Custom Ai Gym environment using PDDL domain.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, pddlenv, feature_env, action_space_list):
        super(CookingEnv, self).__init__()

        self.pddlenv = pddlenv
        self.pddlenv.reset()
        self.feature_env = feature_env

        # state, _ = self.pddlenv.reset()
        # self.pddlenv.action_space._update_objects_from_state(state)
        self._action_space_list = action_space_list
        #list(self.pddlenv.action_space._compute_all_ground_literals(state))

        # self.action_space = spaces.Discrete(len(self._action_space_list))
        self.action_space = DiscreteWrapper(len(self._action_space_list), self.pddlenv,
                                            self._action_space_list)
        # size of state_space, int
        # self.observation_space = spaces.Discrete(len(feature_env.state_features)) 
        self.observation_space = spaces.MultiBinary(len(feature_env.state_features))

        self._finished_oatmeal = False
        self._finished_toast = False
        self._finished_pastry = False
        self._intermediate_reward_val = .33


    def step(self, action_idx):
        action = self._action_space_list[action_idx]
        state, reward, done, info = self.pddlenv.step(action)
        intermediate_reward = self._check_rewards(state)

        state = self.feature_env.transform(state)
        reward += intermediate_reward

        return state, reward, done, info

    def reset(self):
        self._finished_oatmeal = False
        self._finished_pastry = False
        self._finished_toast = False

        state, _  = self.pddlenv.reset()
        return self.feature_env.transform(state)

    def render(self):
        pass

    def close(self):
        pass

    def _check_rewards(self, state):
        """
        Receive intermediate rewards for completing a single course
        """
        state_vars = self.feature_env.eval_pddl_vars(state)
        course_names = ['oatmeal', "pastry", "toast"]
        completed_dict = {"oatmeal_done": self._finished_oatmeal,
                          "pastry_done": self._finished_pastry,
                          "toast_done": self._finished_toast}

        reward = 0.0

        # print("[check_rewards] ", completed_dict)
        for k in completed_dict:
            if state_vars[k] and not completed_dict[k]:
                reward += self._intermediate_reward_val
                # print("[check_rewards] Receiving intermediate rewards!")
                # completed_dict[k] = True
                if "oatmeal" in k:
                    self._finished_oatmeal = True
                elif "pastry" in k:
                    self._finished_pastry = True
                elif "toast" in k:
                    self._finished_toast = True


        return reward





def test_ai_gym():
    pddlenv, feature_env = load_environment(True)

    state, _ = pddlenv.reset()
    pddlenv.action_space._update_objects_from_state(state)
    action_space_list = list(pddlenv.action_space._compute_all_ground_literals(state))
    cookingenv = CookingEnv(pddlenv, feature_env, action_space_list)


    model = ACER(MlpPolicy, cookingenv, verbose=2)
    model.learn(total_timesteps=3000)
    # generate_expert_traj(plan_expert, './models/test_expert', cookingenv, n_episodes=3)

    obs = cookingenv.reset()

    print("OBS: ", obs)
    for i in range(40):
        action, _states = model.predict(obs)
        # print(_states
        # action = cookingenv.action_space.sample()
        obs, rewards, done, info = cookingenv.step(action)
        print("action: ", action_space_list[action])
        print("reward: ", rewards)
        # print("obs: ", obs)
        print("Done: {} at {} ".format(done, i))

    # check_env(cookingenv)





def softmax_action_selection(Q_func, s_a, actions, temp=.4):
    "Implements soft-max action selection"
    # probs = np.zeros(s_a.shape[0])
    EPSILON = 1e-10
    Q_vals = Q_func.predict(s_a)
    top = np.exp(Q_vals/temp) + EPSILON
    probs = top / np.sum(top)
    # print("[softmax] Q vals: ", Q_vals)
    # print("[softmax] probs: ", probs)
    # Use probs to make a a weighted random choice of action
    action = np.random.choice(list(actions), p=probs)
    print("[softmax] Action choice: ", action)
    # sort actions by Q values and print top 3
    sorted_actions = sorted(zip(actions, Q_vals), key=lambda x: x[1], reverse=True)
    for a in sorted_actions[:3]:
        print("[softmax] Action: {}, Q val: {}".format(a[0], a[1]))


    return action

# Use monte carlo learning regime to quickly train model
def train_model_monte_carlo(f_path):
    # Reference: https://plusreinforcement.com/2018/07/05/rl-tutorial-part-1-monte-carlo-methods/
    # https://oneraynyday.github.io/ml/2018/05/24/Reinforcement-Learning-Monte-Carlo/
    env = load_environment()
    # Q = Q_model(env)
    # Q = DQN(env)

    # N = defaultdict(lambda: np.zeros(Q.n_actions)) # Nested (s,a) dict to keep track of counts

    discount = .9

    data_df = pd.read_pickle(data_path)
    # Group df by episode value
    data_df = data_df.groupby('ep')
    for ep_df in data_df:
        ep, df = ep_df
        episode = zip(df.state.values, df.state_int.values, 
                      df.action.values, df.reward.values)
        visited_states = []
        prev_reward = 0
        # print counts for occurences of each state_int
        episode_counts = df.state_int.value_counts()
        print("Episode counts: ", episode_counts)

        for s, s_int, a, r in reversed(list(episode)):
            r = r + discount * prev_reward
            print("Skipping state: ", s_int)
            if s_int not in visited_states:
                # a_idx = Q.action_space.index(a)
                # print(a_idx)
                print(a)
                print(s_int)
                print()
                # N[s_int][a_idx] += 1
                # q = Q.get(s,a)
                # print("State val: ", s_int)
                # print("Old Q: ",q)
                # q += (discount /  N[s_int][a_idx]) * (r - q)
                # print("Q update: ",q)
                # Q.update(s,a,q) 
                # print("Q pred update: ",Q.get(s,a))
                
                visited_states.append(s_int)
                prev_reward = r
    # return Q


# Classic Q model using lookup table
class Q_model(object):
    def __init__(self, env):
        state, _ = env.reset()
        env.action_space._update_objects_from_state(state)
        self.action_space = list(env.action_space._compute_all_ground_literals(state))
        self.n_actions = len(self.action_space)
        self.Q = self._initialize_model()

    def _initialize_model(self):
        # self._state_int_dict = {} # {state:int}
        # return defaultdict(lambda: np.zeros(self.n_actions))
        # stores idx of action in n_action len vec and q val
        return defaultdict(lambda: defaultdict(int)) 

    def update(self, s, a, q):
        s_idx = bool_vec_to_int(s)
        a_idx = self.action_space.index(a)
        self.Q[s_idx][a_idx] = q

    def get(self, s, a=None):
        s_idx = bool_vec_to_int(s)
        if a is None:
            ret_vec = np.zeros(self.n_actions)
            a_idx = list(self.Q[s_idx].keys())
            ret_vec[a_idx] = [self.Q[s_idx][i] for i in a_idx]
            ret = ret_vec
        else:
            a_idx = self.action_space.index(a)
            ret = self.Q[s_idx][a_idx]

        return ret


# DQN implemented using Sklearn MLP
class DQN_sklearn(Q_model):
    def __init__(self, env):
        super().__init__(env)
        self.gamma = 0.95
        self.epsilon = 1.0

    def _initialize_model(self):
        kwargs = {'activation':'relu', 'solver':'adam',
                  'hidden_layer_sizes':(30,) }
        return MLPRegressor(**kwargs)

    def get(self, s, a=None):
        try:
            print("[get] Successfully predicting")
            q_vec = self.Q.predict(s.reshape(1,-1)).reshape(1,-1)

        # Model may not be trained yet so we need to catch this
        except sklearn.exceptions.NotFittedError:
            print("[get] Failed! Model not trained")
            q_vec =np.zeros(self.n_actions).reshape(1,-1)

        if a is None:
            ret = q_vec
        else:
            a_idx = self.action_space.index(a)
            ret = q_vec[0,a_idx]

        return ret

    def update(self, s, a, q):
        a_idx = self.action_space.index(a)
        q_vec = self.get(s)
        q_vec[0, a_idx] = q

        print("[update] Training model")
        self.Q.partial_fit(s.reshape(1,-1), q_vec)


    def max_Q(self, s):
        q_vec = self.get(s)
        max_id = np.argmax(q_vec)
        return self.action_space[max_id], q_vec[0,max_id]



    # compare_agents_htn(start=30, end=40)
    # test_int = SimInteractions(action_space)
    # for t in test_int:
    #     name, stuff= t
    #     htn, o, sh = stuff
    #     print(name)
    #     print(htn)
    #     print(o)
    #     print(sh)
    # egg_to_roll_shield = AlternateShield("gather_roll_not_egg",
    #                                failiure_action="gather(egg:pastry,do:action_type)",
    #                                alt_action="gather(roll:pastry,do:action_type)",
    #                                action_space=action_space)

    # egg_to_jelly_shield = AlternateShield("gather_jelly_not_egg",
    #                                failiure_action="gather(egg:pastry,do:action_type)",
    #                                alt_action="gather(jellypastry:pastry,do:action_type)",
    #                                action_space=action_space)
    # put_in_jelly_shield = AlternateShield("putin_jelly_not_egg",
    #                                failiure_action="putinmicrowave(egg:pastry,do:action_type)",
    #                                alt_action="putinmicrowave(jellypastry:pastry,say:action_type)",
    #                                action_space=action_space)

    # microwave_jelly_shield = AlternateShield("microwave_jelly_not_egg",
    #                                failiure_action="microwavepastry(egg:pastry,say:action_type)",
    #                                alt_action="microwavepastry(jellypastry:pastry,say:action_type)",
    #                                action_space=action_space)
    # # check_jelly_shield = AlternateShield("check_jelly_not_egg",
    # #                                   failiure_action="checkjellypastry(side:meal,say:action_type)",
    # #                                     alt_action="checkegg(side:meal,say:action_type)",
    # #                                     action_space=action_space)


    # egg_refienement_shield = RefineShield("test_refinement",
    #                             failiure_cond="jellypastry_in_workspace or roll_in_workspace",
    #                             actions=[
    #                                 "gather(egg:pastry,say:action_type)",
    #                                 "putinmicrowave(egg:pastry,say:action_type)",
    #                                 "turnon(microwave:appliance,say:action_type)",
    #                                 "microwavepastry(egg:pastry,say:action_type)",
    #                                 "takeoutmicrowave(egg:pastry,say:action_type)",
    #                                 "checkegg(side:meal,say:action_type)"],
    #                             action_space=action_space)


    # jellypastry_refienement_shield = \
    #     RefineShield("test_refinement",
    #                  failiure_cond="egg_in_workspace or roll_in_workspace",
    #                  actions=[
    #                      "gather(jellypastry:pastry,say:action_type)",
    #                      "putinmicrowave(jellypastry:pastry,say:action_type)",
    #                      "turnon(microwave:appliance,say:action_type)",
    #                      "microwavepastry(jellypastry:pastry,say:action_type)",
    #                      "takeoutmicrowave(jellypastry:pastry,say:action_type)",
    #                      "checkjellypastry(side:meal,say:action_type)"],
    #                  action_space=action_space)


    # train_plan_list = [Plan(side=EGG, main=FRUITYOATMEAL,
    #                   liquid=WATER, action_types={'main':'say', 'side':'do'},
    #                         problem_idx=1),
    #                    Plan(side=JELLYPASTRY, main=CEREAL,
    #                         liquid=MILK, action_types={'main':'say', 'side':'do'},
    #                         problem_idx=1),
    #                    Plan(side=ROLL, main=FRUITYCHOCOLATEOATMEAL,
    #                         liquid=WATER, action_types={'main':'do', 'side':'say'},
    #                         problem_idx=1),
    #                    Plan(side=EGG, main=PLAINOATMEAL,
    #                         liquid=WATER, action_types={'main':'do', 'side':'do'},
    #                         problem_idx=1),
    #                    Plan(side=ROLL, main=CHOCOLATEOATMEAL,
    #                         liquid=MILK, action_types={'main':'do', 'side':'do'},
    #                         problem_idx=1),
    #                    ]
    
    # train_plan_list = [CookingHTNFactory(side=EGG, main=FRUITYOATMEAL,
    #                   liquid=WATER, action_types={'main':'say', 'side':'do'},
    #                                      ).ingredients_first_main_first(),
    #                    CookingHTNFactory(side=JELLYPASTRY, main=CEREAL,
    #                         liquid=MILK, action_types={'main':'say', 'side':'do'},
    #                         ).ingredients_first_main_first(),
    #                    CookingHTNFactory(side=ROLL, main=FRUITYCHOCOLATEOATMEAL,
    #                         liquid=WATER, action_types={'main':'do', 'side':'say'},
    #                         ).ingredients_first_main_first(),
    #                    CookingHTNFactory(side=EGG, main=PLAINOATMEAL,
    #                         liquid=WATER, action_types={'main':'do', 'side':'do'},
    #                         ).ingredients_first_main_first(),
    #                    CookingHTNFactory(side=ROLL, main=CHOCOLATEOATMEAL,
    #                         liquid=MILK, action_types={'main':'do', 'side':'do'},
    #                         ).ingredients_first_main_first(),
    #                    ]

    # sf = ShieldFactory(action_space)
    # egg_refine_shield = sf.pastry_refine_shield("not egg_in_workspace and not pastry_done",
    #                             EGG,
    #                             "do")
    # gather_egg_shield = sf.gather_pastry_alternate("test", JELLYPASTRY, EGG)
    # # Tests one of the meals from the training set
    # test_plan_1 = CookingHTNFactory(side=EGG, main=FRUITYOATMEAL,
    #                                 liquid=MILK, action_types={'main':'say', 'side':'do'},
    #                                 ).ingredients_first_side_first()
    # overlay_list_1 = [protien_pastry_overlay,
    #                   # making_oatmeal_overlay,
    #                   fruity_oatmeal_overlay
    #                   ]
    # shield_list_1 = [egg_refine_shield]

    # # Combination of two meals from the training set
    # test_plan_2 = CookingHTNFactory(side=JELLYPASTRY, main=FRUITYOATMEAL,
    #                                 liquid=MILK, action_types={'main':'say', 'side':'do'},
    #                       )
    # overlay_list_2 = [sweet_pastry_overlay, fruity_oatmeal_overlay]

    # # Unseen "vegan" an healthy variation
    # test_plan_3 = CookingHTNFactory(side=ROLL, main=PLAINOATMEAL,
    #                                 liquid=MILK, action_types={'main':'say', 'side':'do'},
    #                                 )
    # overlay_list_3 = [healthy_overlay]
    # # Combination of two meals from the training set
    # test_plan_4 = CookingHTNFactory(side=EGG, main=CEREAL,
    #                                 liquid=MILK, action_types={'main':'say', 'side':'do'},
    #                                 )
    # overlay_list_4 = [protien_pastry_overlay, making_cereal_overlay]
    # shield_list_4 = [gather_egg_shield]

    # # Combination of two meals from the training set and changed action types
    # test_plan_5 = CookingHTNFactory(side=JELLYPASTRY, main=CHOCOLATEOATMEAL,
    #                                 liquid=MILK, action_types={'main':'do', 'side':'do'},
    #                                 )
    
    # # train_plan_list = [CookingHTNFactory(side=EGG, main=FRUITYOATMEAL,
    # #                   liquid=WATER, action_types={'main':'say', 'side':'do'},
    # #                         ),
    # #                    CookingHTNFactory(side=JELLYPASTRY, main=FRUITYOATMEAL,
    # #                         liquid=WATER, action_types={'main':'do', 'side':'do'},
    # #                         ),
    # #                    ]

    # # import pdb; pdb.set_trace()
    # test_plan_list = [CookingHTNFactory(side=JELLYPASTRY, main=CEREAL,
    #                                     action_types={'main':'do', 'side':'say'}
    #                                     , liquid=MILK),
    #                   CookingHTNFactory(side=JELLYPASTRY, main=FRUITYOATMEAL,
    #                                     action_types={'main':'do', 'side':'say'},
    #                                     liquid=WATER)
    #                   ]
    
    # overlay_list = [sweet_pastry_overlay,
    #                 sweet_oatmeal_overlay
    #                 ]

    # # print(shield_list_1[0])
    # print(test_plan_1)
    # # compare_agents_no_plan(test_plan_1,
    # #                        shield_list_4,
    # #                        overlay_list_1, start=25, end=55, n_iter=30)


    # pb_htn_gen_1 = CookingHTNFactory(PBCHOCOLATEOATMEAL, MUFFIN,
    #                             "water", {"main":"do", "side":"do"})
    # cereal_htn_gen = CookingHTNFactory(FRUITYOATMEAL, JELLYPASTRY,
    #                             WATER, {"main":"do", "side":"do"})
    # # htn_list = [pb_htn_gen_1.use_immediately_side_first()]
    # htn_list = [
    #     pb_htn_gen_1.ingredients_first_side_first(),
    #     cereal_htn_gen.ingredients_first_side_first(),
    #             ]
    # shield_list =[sf.pastry_forbid_shield(EGG)]
    # # DQN_trainer_from_htn(train_plan_list, num_episodes=40)
    # # DQN_trainer_from_plan(plan_list=train_plan_list, num_episodes=30)
    # # experiment 1
    # # Training on 5 different meal plans
    # # Target meal: Jelly pastry and cereal  (robot makes ceral), human makes pastry. This is presentt in the training set with the exception that the robot has only made jelly pastries
    # # Utilized two overlays: 1)  Do cereal , 2) say jelly pastry
    # #
