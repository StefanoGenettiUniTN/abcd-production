from typing import List, Dict, Tuple, Set
from alpypeopt import AnyLogicModel
from ast import literal_eval
import matplotlib.pyplot as plt
import csv
import argparse
import pandas as pd
import random

def read_arguments():
    parser = argparse.ArgumentParser(description="ABCD production optimizer.")
    
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--optuna_csv", type=str, help="Optuna fitness trend.")
    parser.add_argument("--random_search_csv", type=str, help="Random search fitness trend.")
    parser.add_argument("--aco_csv", type=str, help="Ant Colony Optimization fitness trend.")
    parser.add_argument("--rl_csv", type=str, help="Reinforcement learning fitness trend.")

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    rng = random.Random(args["random_seed"])

    plt.figure(figsize=(10, 6))

    if args["optuna_csv"]:
        history_x:List[List[int]] = []
        history_y:List[int] = []
        csv_file_path = args["optuna_csv"]
        csv_file = open(csv_file_path, mode='r', newline='')
        csv_reader = csv.reader(csv_file)
        next(csv_reader)    # skip the header
        for row in csv_reader:
            x = literal_eval(row[0])  # convert the string representation of list back to a list
            y = int(row[1])           # convert the string representation of integer back to an integer
            history_x.append(x)
            history_y.append(y)
        plt.plot(range(len(history_y)), history_y, marker='o', linestyle='-', markersize=3, label="optuna")

    if args["random_search_csv"]:
        history_x:List[List[int]] = []
        history_y:List[int] = []
        csv_file_path = args["random_search_csv"]
        csv_file = open(csv_file_path, mode='r', newline='')
        csv_reader = csv.reader(csv_file)
        next(csv_reader)    # skip the header
        for row in csv_reader:
            x = literal_eval(row[0])  # convert the string representation of list back to a list
            y = int(row[1])           # convert the string representation of integer back to an integer
            history_x.append(x)
            history_y.append(y)
        plt.plot(range(len(history_y)), history_y, marker='o', linestyle='-', markersize=3, label="random search")

    if args["aco_csv"]:
        history_x:List[List[int]] = []
        history_y:List[int] = []
        csv_file_path = args["aco_csv"]
        df = pd.read_csv(csv_file_path)
        plt.plot(df['generation'], df['best_fitness'], marker='o', linestyle='-', markersize=3, label="ant colony optimization")
    
    if args["rl_csv"]:
        history_x:List[List[int]] = []
        history_y:List[int] = []
        csv_file_path = args["rl_csv"]
        csv_file = open(csv_file_path, mode='r', newline='')
        csv_reader = csv.reader(csv_file)
        next(csv_reader)    # skip the header
        for row in csv_reader:
            x = literal_eval(row[0])  # convert the string representation of list back to a list
            y = float(row[1])           # convert the string representation of integer back to an integer
            history_x.append(x)
            history_y.append(y)
        plt.plot(range(len(history_y)), history_y, marker='o', linestyle='-', markersize=3, label="reinforcement learning")

    
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel('Evaluation', fontsize=14)
    plt.ylabel('Revenue', fontsize=14)
    plt.title('Fitness Trend', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()