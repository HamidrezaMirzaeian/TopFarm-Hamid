import importlib
import importlib.util
import numpy as np
import matplotlib as plt

from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian     #wake model
from py_wake.examples.data.iea37 import IEA37_WindTurbines, IEA37Site         #wind turbines and site used
from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent   #cost model

from topfarm import TopFarmProblem
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.examples.iea37 import get_iea37_initial, get_iea37_constraints, get_iea37_cost
from topfarm.plotting import NoPlot, XYPlotComp

n_wt = 9
n_wd = 16

site = IEA37Site(9)
wind_turbines = IEA37_WindTurbines()
wd = np.linspace(0.,360.,n_wd, endpoint=False)
wfmodel = IEA37SimpleBastankhahGaussian(site, wind_turbines)   #PyWake's wind farm model

cost_comp = PyWakeAEPCostModelComponent(wfmodel, n_wt, wd=wd)

initial = get_iea37_initial(n_wt)
driver = EasyScipyOptimizeDriver()

{'x': [1, 2, 3], 'y':([3, 2, 1], 0, 1), 'z':([4, 5, 6],[4, 5, 4], [6, 7, 6])}
design_vars = dict(zip('xy', (initial[:, :2]).T))

tf_problem = TopFarmProblem(
            design_vars,
            cost_comp,
            constraints=get_iea37_constraints(n_wt),
            driver=driver,
            plot_comp=XYPlotComp())
_, state, _ = tf_problem.optimize()
