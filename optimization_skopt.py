from alpypeopt import AnyLogicModel
from skopt import gp_minimize
from skopt.space import Integer

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
    
    # 'skopt' package only allows minimization problems
    # because of that, value must be negated
    return -model_output.getTotalRevenue()

# setup and execute black box optimmization model
res = gp_minimize(simulation,                                      # the function to minimize
                  [Integer(0, 1) for _ in range(100)],             # the bounds on each dimension of x
                  acq_func="EI",                                   # the acquisition function
                  n_calls=500,                                      # the number of evaluations of simulation
                  n_random_starts=5,                               # the number of random initialization points
                  random_state=1234)                               # the random seed

print(f"Solution is {res.x} for a value of {-res.fun}")

# run simulation with optimal result to use UI to explore results in AnyLogic
simulation(res.x, reset=False)

# close model
abcd_model.close()