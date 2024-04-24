import importlib
import numpy as np
import matplotlib.pyplot as plt

from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.examples.iea37 import get_iea37_initial, get_iea37_constraints, get_iea37_cost
from topfarm.plotting import NoPlot, XYPlotComp
from topfarm.cost_models.cost_model_wrappers import CostModelComponent


n_wt = 9          #valid number of turbines are: 9, 16, 36, 64
x, y = get_iea37_initial(n_wt).T

from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian   #wake model
from py_wake.examples.data.iea37 import IEA37_WindTurbines, IEA37Site       #wind turbines and site used

site = IEA37Site(n_wt)
wind_turbines = IEA37_WindTurbines()

wake_model = IEA37SimpleBastankhahGaussian(site, wind_turbines)
wd = np.linspace(0., 360., 16, endpoint=False)                              #number of wind directions to study

#objective function
def aep_func(x,y,wd=wd):
    sim_res = wake_model(x,y, wd=wd)
    aep = sim_res.aep().sum()
    return aep

def __init__(self, input_keys, n_wt, cost_function, cost_gradient_function=None,
             output_keys=["Cost"], output_unit="", additional_input=[], additional_output=[], max_eval=None,
             objective=True, maximize=False, output_vals=[0.0], input_units=[], step={}, use_penalty=True, **kwargs):

    from topfarm.cost_models.cost_model_wrappers import CostModelComponent

constraint = get_iea37_constraints(n_wt)

aep_comp = CostModelComponent(input_keys=['x','y'],
                              n_wt=n_wt,
                              cost_function=aep_func,
                              output_keys=[('AEP', 0)],
                              output_unit="GWh",
                              objective=True,
                              maximize=True
                             )
problem = TopFarmProblem(design_vars={'x': x, 'y': y},
                          n_wt=n_wt,
                          cost_comp=aep_comp,
                          constraints=constraint,
                          driver=EasyScipyOptimizeDriver(disp=False),
                          plot_comp=XYPlotComp()
                        )
_, state,_ = problem.optimize()
state
x_values = state['x']
y_values = state['y']

# Plot the design variables
plt.figure()
plt.plot(x_values, y_values, 'o')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Optimized Wind Farm Layout')
plt.grid(True)
plt.axis('equal')  # Set aspect ratio to be equal
plt.show()
