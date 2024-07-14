from typing import List, Dict, Tuple, Set
from alpypeopt import AnyLogicModel
from ast import literal_eval
import matplotlib.pyplot as plt
import csv
import argparse
import random

def read_arguments():
    parser = argparse.ArgumentParser(description="ABCD production optimizer.")
    
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--input_csv", type=str, help="File path of the CSV where the optimization output has been stored.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials.")

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    rng = random.Random(args["random_seed"])

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
        abcd_model = AnyLogicModel(
            env_config={
                'run_exported_model': False
            }
        )

        # init design variable setup
        design_variable_setup = abcd_model.get_jvm().abcdproduction.DesignVariableSetup()

        def simulation(x, reset=True):
            for orderId, outsourceFlag in enumerate(x):
                if outsourceFlag==1:
                    design_variable_setup.setToOne(orderId)
                else:
                    design_variable_setup.setToZero(orderId)
            
            # pass input setup and run model until end
            abcd_model.setup_and_run(design_variable_setup)
            
            # extract model output or simulation result
            model_output = abcd_model.get_model_output()
            if reset:
                # reset simulation model to be ready for next iteration
                abcd_model.reset()
            
            return model_output.getTotalRevenue()

        # run the optimizer
        best_y = -100000
        best_x = [random.randint(0,1) for _ in range(100)]
        for _ in range(args["n_trials"]):
            x = [random.randint(0,1) for _ in range(100)]
            y = simulation(x)
            if y>best_y:
                best_y = y
                best_x = x.copy()
            history_x.append(best_x.copy())
            history_y.append(int(y))

        # print result
        print(f"Solution is {best_x} for a value of {best_y}")

        # run simulation with optimal result to use UI to explore results in AnyLogic
        simulation(best_x, reset=False)

        # close model
        abcd_model.close()

        # store fitness trend history in csv output file
        csv_file_path = "history_random_search.csv"
        csv_file = open(csv_file_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["x", "y"])
        for x, y in zip(history_x, history_y):
            csv_writer.writerow([str(x), y])
        csv_file.close()
        print(f"Data successfully written to {csv_file_path}")