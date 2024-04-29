import importlib
import numpy as np
import matplotlib.pyplot as plt
import time
from py_wake.deficit_models.gaussian import BastankhahGaussian
from py_wake.examples.data.lillgrund import LillgrundSite
from py_wake.utils.gradients import autograd
from py_wake.examples.data.hornsrev1 import HornsrevV80

from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.easy_drivers import EasySGDDriver, EasyScipyOptimizeDriver
from topfarm.plotting import XYPlotComp
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm import TopFarmProblem
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.recorders import TopFarmListRecorder
from topfarm.constraint_components.constraint_aggregation import ConstraintAggregation
from topfarm.constraint_components.constraint_aggregation import DistanceConstraintAggregation

# defining the site, wind turbines and wake model
site = LillgrundSite()
site.interp_method = 'linear'
windTurbines = HornsrevV80()
wake_model = BastankhahGaussian(site, windTurbines)

#wind farm layout
x_rows = 3 # 5 # (commented for speeding up notebook tests)
y_rows = 3 # 5
sgd_iterations = 100 # 2000
spacing = 3
xu, yu = (x_rows * spacing * windTurbines.diameter(), y_rows * spacing * windTurbines.diameter())
np.random.seed(4)
x = np.random.uniform(0, xu, x_rows * y_rows)
y = np.random.uniform(0, yu, x_rows * y_rows)
x0, y0 = (x.copy(), y.copy())
n_wt = x.size

#wind resource
dirs = np.arange(0, 360, 1) #wind directions
ws = np.arange(3, 25, 1) # wind speeds
freqs = site.local_wind(x, y, wd=dirs, ws=ws).Sector_frequency_ilk[0, :, 0]     #sector frequency
As = site.local_wind(x, y, wd=dirs, ws=ws).Weibull_A_ilk[0, :, 0]               #weibull A
ks = site.local_wind(x, y, wd=dirs, ws=ws).Weibull_k_ilk[0, :, 0]               #weibull k

#boundaries
boundary = np.array([(0, 0), (xu, 0), (xu, yu), (0, yu)])

# objective function and gradient function
samps = 50    #number of samples

#function to create the random sampling of wind speed and wind directions
def sampling():
    idx = np.random.choice(np.arange(dirs.size), samps, p=freqs)
    wd = dirs[idx]
    A = As[idx]
    k = ks[idx]
    ws = A * np.random.weibull(k)
    return wd, ws

#aep function - SGD
def aep_func(x, y, full=False, **kwargs):
    wd, ws = sampling()
    aep_sgd = wake_model(x, y, wd=wd, ws=ws, time=True).aep().sum().values * 1e6
    return aep_sgd

#gradient function - SGD
def aep_jac(x, y, **kwargs):
    wd, ws = sampling()
    jx, jy = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=x, y=y, ws=ws, wd=wd, time=True)
    daep_sgd = np.array([np.atleast_2d(jx), np.atleast_2d(jy)]) * 1e6
    return daep_sgd

#aep function - SLSQP
def aep_func2(x, y, **kwargs):
    wd = np.arange(0, 360, 0.5)
    ws = np.arange(3, 25, 0.5)
    aep_slsqp = wake_model(x, y, wd=wd, ws=ws).aep().sum().values * 1e6
    return aep_slsqp

#gradient function - SLSQP
def aep_jac2(x, y, **kwargs):
    wd = np.arange(0, 360, 0.5)
    ws = np.arange(3, 25, 0.5)
    jx, jy = wake_model.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'], x=x, y=y, ws=ws, wd=wd, time=False)
    daep_slsqp = np.array([np.atleast_2d(jx), np.atleast_2d(jy)]) * 1e6
    return daep_slsqp

#aep component - SGD
aep_comp = CostModelComponent(input_keys=['x','y'], n_wt=n_wt, cost_function=aep_func, objective=True, cost_gradient_function=aep_jac, maximize=True)

#aep component - SLSQP
aep_comp2 = CostModelComponent(input_keys=['x','y'], n_wt=n_wt, cost_function=aep_func2, objective=True, cost_gradient_function=aep_jac2, maximize=True)

cost_comps = [aep_comp2, aep_comp]

min_spacing_m = 2 * windTurbines.diameter()  #minimum inter-turbine spacing in meters
constraint_comp = XYBoundaryConstraint(boundary, 'rectangle')

#constraints
constraints = [[SpacingConstraint(min_spacing_m), constraint_comp],
               DistanceConstraintAggregation([SpacingConstraint(min_spacing_m), constraint_comp],n_wt, min_spacing_m, windTurbines)]

#driver specs
driver_names = ['SLSQP', 'SGD']
drivers = [EasyScipyOptimizeDriver(maxiter=200, tol=1e-3),
           EasySGDDriver(maxiter=sgd_iterations, learning_rate=windTurbines.diameter()/5, max_time=18000, gamma_min_factor=0.1)]

driver_no = 1    #SGD driver
ec = [10,1]      #expected cost for SLSQP (10) and SGD (1) drivers

tf = TopFarmProblem(
        design_vars = {'x':x0, 'y':y0},
        cost_comp = cost_comps[driver_no],
        constraints = constraints[driver_no],
        driver = drivers[driver_no],
        plot_comp = XYPlotComp(),
        expected_cost = ec[driver_no]
        )
if 1:
    tic = time.time()
    cost, state, recorder = tf.optimize()
    toc = time.time()
