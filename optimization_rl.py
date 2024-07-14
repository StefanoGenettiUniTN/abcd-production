from typing import List, Dict, Tuple, Set
from alpypeopt import AnyLogicModel
from ast import literal_eval
from gymnasium import spaces
from ray.tune.registry import register_env
from ray.rllib.algorithms import ppo
import gymnasium as gym
import matplotlib.pyplot as plt
import csv
import argparse
import numpy as np
import pandas as pd
import ray

class ABCDEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, num_orders=100):
        # observation space
        #self.observation_space = spaces.Sequence(spaces.Dict({
            #'order_id': spaces.Box(low=0, high=num_orders, shape=(), dtype=np.int32),
            #'components': spaces.Box(low=0, high=20, shape=(3,), dtype=np.int32),
            #'deadline': spaces.Box(low=0, high=3000000000, shape=(), dtype=np.int64),
        #}))
        low_bounds = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        high_bounds = np.array([num_orders, 20, 20, 20, 3000000000], dtype=np.int64)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.int64)

        # action space
        self.action_space = spaces.Discrete(2)

        # anylogic
        self.abcd_model = AnyLogicModel(
            env_config={
                'run_exported_model': False
            }
        )
        self.design_variable_setup = self.abcd_model.get_jvm().abcdproduction.DesignVariableSetup()   # init design variable setup
        self.design_variable = []
        self.current_order_id = 0

    def reset(self, seed=42, options=None):
        super().reset(seed=seed)

        # read the exel file of orders
        df = pd.read_excel("orders.xlsx")

        self.orders = []
        for _, row in df.iterrows():
            order = {
                'order_id': int(row['id']),
                'components': np.array([int(row['a']), int(row['b']), int(row['c'])]),
                'deadline': pd.to_datetime(row['deadline']).timestamp()
            }
            self.orders.append(order)
        
        self.design_variable = []
        self.current_order_id = 0

        return self.flat_orders(self.orders), {}

    def step(self, action):
        # action is 0 or 1
        self.design_variable.append(action)
        
        # an episode is terminated if we have taken a decision for each order
        terminated = len(self.design_variable)==100
        
        # compute reward
        reward = self.simulation(self.design_variable) if terminated else 0
        
        # update observation space
        self.current_order_id += 1
        observation = self.orders[self.current_order_id:]

        return self.flat_orders(observation), reward, terminated, False, {}

    def render(self):
        print(f"[render] current order: {self.current_order_id}")

    def close(self):
        self.abcd_model.close()

    def simulation(self, x, reset=True):
        for orderId, outsourceFlag in enumerate(x):
            if outsourceFlag==1:
                self.design_variable_setup.setToOne(orderId)
            else:
                self.design_variable_setup.setToZero(orderId)
        
        # pass input setup and run model until end
        self.abcd_model.setup_and_run(self.design_variable_setup)
        
        # extract model output or simulation result
        model_output = self.abcd_model.get_model_output()
        if reset:
            # reset simulation model to be ready for next iteration
            self.abcd_model.reset()
        return model_output.getTotalRevenue()

    def flat_orders(self, orders):
        if(len(orders)==0):
            return np.array([0, 0, 0, 0, 0])
        order = orders[0]
        order_id = order["order_id"]
        components_a = order["components"][0]
        components_b = order["components"][1]
        components_c = order["components"][2]
        deadline = order["deadline"]
        return np.array([order_id, components_a, components_b, components_c, deadline])

def read_arguments():
    parser = argparse.ArgumentParser(description="ABCD production optimizer.")
    
    parser.add_argument("--input_csv", type=str, help="File path of the CSV where the optimization output has been stored.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials.")

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()

    # history of evaluations
    history_x:List[List[int]] = []
    history_y:List[int] = []

    if args["input_csv"]:
        csv_file_path = args["input_csv"]
        csv_file = open(csv_file_path, mode='r', newline='')
        csv_reader = csv.reader(csv_file)
        next(csv_reader)    # skip the header
        for row in csv_reader:
            x = literal_eval(row[0])  # convert the string representation of list back to a list
            y = int(row[1])           # convert the string representation of integer back to an integer
            history_x.append(x)
            history_y.append(y)

        # plot fitness trent
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(history_y)), history_y, marker='o', linestyle='-', color='b')
        plt.xlabel('Evaluation')
        plt.ylabel('Revenue')
        plt.title('Fitness Trend')
        plt.grid(True)
        plt.show()
    else:
        # random action sampling
        """
        env = ABCDEnv()
        np.random.seed(42)
        observation, info = env.reset()

        for i in range(args["n_trials"]):
            for _ in range(100):
                action = env.action_space.sample()
                observation, reward, done, _, _ = env.step(action)
                print(f"Iteration {i}: Action {action}, Reward {reward}, Done {done}")
                print(f"Observation: {observation}")
                if done:
                    print(f"Episode {i} finished")
                    observation, info = env.reset()
        """

        # PPO training
        # register environment
        def env_creator(env_config):
            return ABCDEnv(num_orders=env_config.get("num_orders"))
        register_env("ABCDEnv-v0", env_creator)

        # checkpoint directory
        checkpoint_dir = "rl-checkpoint"

        # ray initialization
        context = ray.init()
        print(f"DASHBOARD: {context}")
        config = {
            "env": "ABCDEnv-v0",
            "env_config": {
                "num_orders": 100
            },
            "framework": "torch",
            "num_workers": 1,
            "num_envs_per_worker": 1
        }
        algo = ppo.PPO(config=config)
        for i in range(500):
            result = algo.train()
            history_x.append([1,1,1,1,1])
            history_y.append(result['episode_reward_max'])
            print(f"Iteration {i}:\nreward_min = {result['episode_reward_min']}\nreward_mean = {result['episode_reward_mean']}\nreward_max = {result['episode_reward_max']}\nlen_mean = {result['episode_len_mean']}")
        ray.shutdown()

        # store fitness trend history in csv output file
        csv_file_path = "history_rl.csv"
        csv_file = open(csv_file_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["x", "y"])
        for x, y in zip(history_x, history_y):
            csv_writer.writerow([str(x), y])
        csv_file.close()
        print(f"Data successfully written to history.csv")

