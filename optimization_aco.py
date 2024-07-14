from typing import List, Dict, Tuple, Set, Callable
from alpypeopt import AnyLogicModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import random
import inspyred

from ea.observer import ea_observer # todo mettere anche observer nella classe

class ACO_ABCD(inspyred.benchmarks.Benchmark):
    def __init__(self, orders:List[int]):
        inspyred.benchmarks.Benchmark.__init__(self, len(orders))
        self.orders = orders
        self.components = [inspyred.swarm.TrailComponent((order), value=1) for order in orders]
        self.components.extend([inspyred.swarm.TrailComponent((order+len(orders)), value=1) for order in orders])   # order i+len(orders) represents the decision to not outsource order i
        self.capacity = len(orders)
        self.bias = 0.5
        self.bounder = inspyred.ec.DiscreteBounder([0,1])
        self.maximize = True
        self._use_ants = False
    
    def constructor(self, random, args):
        # return a candidate solution for an ant colony optimization
        self._use_ants = True
        candidate = []

        # for each order in self.orders decide whether to outsource the order or not
        while len(candidate) < self.capacity:
            # find feasible components
            feasible_components = []
            if len(candidate) == 0:
                feasible_components = self.components
            else:
                feasible_components = [c for c in self.components if c not in candidate and inspyred.swarm.TrailComponent((c.element+self.capacity),value=1) not in candidate and inspyred.swarm.TrailComponent((c.element-self.capacity),value=1) not in candidate]
            #print(f"({len(feasible_components)}) feasible_components={feasible_components}")
            if len(feasible_components)==0:
                break
            else:
                # choose a feasible component
                if random.random()<=self.bias:
                    next_component = random.choice(feasible_components)
                else:
                    next_component = inspyred.ec.selectors.fitness_proportionate_selection(random, feasible_components, {'num_selected': 1})[0]
                candidate.append(next_component)
            #print(f"next_components={next_component} - fitness={next_component.fitness}")
        #print(f"(len={len(candidate)}) candidate: {candidate}")
        return candidate

    def evaluator(self, candidates, args):
        fitness_function = args["fitness_function"]
        history_y = args["history_y"]

        # return the fitness values for the given candidate
        fitness = [None]*len(candidates)
        for index, a in tqdm(enumerate(candidates), total=len(candidates), desc=f"Candidates evaluation"):
            individual = [0]*100
            for item in a:
                if item.element<100:
                    individual[item.element] = 1
            fitness[index] = fitness_function(individual)
            history_y.append(int(fitness[index]))
        return fitness
    
    def terminator(self, population, num_generations, num_evaluations, args):
        if num_generations == args["max_generations"]:
            data = args["plot_data"]
            df = pd.DataFrame()
            df["generation"] = data[0]
            df["average_fitness"] = data[1]
            df["median_fitness"] = data[2]
            df["best_fitness"] = data[3]
            df["worst_fitness"] = data[4]
            df.to_csv("history_aco.csv", sep=",", index=False)
        return num_generations == args["max_generations"]


def read_arguments():
    parser = argparse.ArgumentParser(description="ABCD production optimizer.")
    
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--input_csv", type=str, help="File path of the CSV where the optimization output has been stored.")
    
    parser.add_argument('--population_size', type=int, default=50, help='EA population size.')
    parser.add_argument('--max_generations', type=int, default=50, help='Generational budget.')

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    rng = random.Random(args["random_seed"])
    
    if args["input_csv"]:
        csv_file_path = args["input_csv"]
        df = pd.read_csv(csv_file_path)
        plt.figure(figsize=(10, 6))
        plt.plot(df['generation'], df['average_fitness'], marker='o', linestyle='-', linewidth=1, markersize=4, label='Average Fitness')
        plt.plot(df['generation'], df['median_fitness'], marker='o', linestyle='-', linewidth=1, markersize=4, label='Median Fitness')
        plt.plot(df['generation'], df['best_fitness'], marker='o', linestyle='-', linewidth=1, markersize=4, label='Best Fitness')
        plt.plot(df['generation'], df['worst_fitness'], marker='o', linestyle='-', linewidth=1, markersize=4, label='Worst Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Revenue')
        plt.title('Fitness Trend')
        plt.legend()
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

        def simulation(x:List[int], reset=True):
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

        # init discrete optimization problem
        abcd_problem = ACO_ABCD(orders=[i for i in range(100)])

        # run the optimizer
        ac = inspyred.swarm.ACS(random=rng, components=abcd_problem.components)
        ac.observer = [ea_observer]
        ac.terminator = abcd_problem.terminator
        plot_data = [[], [], [], [], []]                                        # fitness trend to plot
        plot_data[0] = []                                                       # generation number
        plot_data[1] = []                                                       # average fitenss
        plot_data[2] = []                                                       # median fitness
        plot_data[3] = []                                                       # best fitness
        plot_data[4] = []                                                       # worst fitness
        final_pop = ac.evolve(  generator=abcd_problem.constructor,             # the function to be used to generate candidate solutions
                                evaluator=abcd_problem.evaluator,               # the function to be used to evaluate candidate solutions
                                pop_size=args["population_size"],               # the number of Individuals in the population 
                                max_generations=args["max_generations"],        # maximum generations
                                maximize=abcd_problem.maximize,                 # boolean value stating use of maximization
                                bounder=abcd_problem.bounder,                   # a function used to bound candidate solutions 
                                fitness_function=simulation,                    # fitness_function
                                history_y=[],                                   # keep track of individual fitness
                                plot_data = plot_data                           # data[0] generation number ; data[1] average fitenss ; data[2] median fitness ; data[3] best fitness ; data[4] worst fitness
                            )
        best_ACS = max(ac.archive)
        print(f"best_ACS={best_ACS}")
        """
        indices = []
        for order in best_ACS.candidate:
            # each item is (element, value)
            index = items.index((item.element, item.value))
            indices.append(index)
        solution_ACS = np.zeros(len(items),dtype=np.uint16)
        for i in indices:
            solution_ACS[i] += 1
        solution_ACS = solution_ACS.tolist()

        solution_EA = best_EA.candidate

        print("Best Solution ACS: {0} - Value: {1}".format(str(solution_ACS), best_ACS.fitness))
        print("Best Solution EA : {0} - Value: {1}".format(str(solution_EA), best_EA.fitness))
        """