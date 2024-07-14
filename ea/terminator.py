import pandas as pd

def ea_terminator(population, num_generations, num_evaluations, args):
    if num_generations == args["generations_budget"]:
       data = args["plot_data"]
       df = pd.DataFrame()
       df["generation"] = data[0]
       df["average_fitness"] = data[1]
       df["median_fitness"] = data[2]
       df["best_fitness"] = data[3]
       df["worst_fitness"] = data[4]
       df.to_csv("history_ga.csv", sep=",", index=False)
    return num_generations == args["generations_budget"]