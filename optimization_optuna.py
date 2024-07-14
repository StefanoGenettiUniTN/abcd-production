from typing import List, Dict, Tuple, Set
from alpypeopt import AnyLogicModel
from ast import literal_eval
import optuna
import matplotlib.pyplot as plt
import csv
import argparse

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

        # setup and execute black box optimization model
        def objective(trial):
            x = [trial.suggest_int(f'x{i}', 0, 1) for i in range(100)]
            y = simulation(x)
            history_x.append(x.copy())
            history_y.append(int(y))
            return y

        # run the optimizer
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args["n_trials"])

        # print result
        best = study.best_params
        print(f"Solution is {best} for a value of {study.best_value}")

        # run simulation with optimal result to use UI to explore results in AnyLogic
        best_x = list(best.values())
        simulation(best_x, reset=False)

        # close model
        abcd_model.close()

        # store fitness trend history in csv output file
        csv_file_path = "history_optuna.csv"
        csv_file = open(csv_file_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["x", "y"])
        for x, y in zip(history_x, history_y):
            csv_writer.writerow([str(x), y])
        csv_file.close()
        print(f"Data successfully written to history.csv")