import importlib
import numpy as np
import matplotlib.pyplot as plt

from py_wake.examples.data.hornsrev1 import V80, Hornsrev1Site
from py_wake import BastankhahGaussian
from py_wake.utils.gradients import autograd

from topfarm import TopFarmProblem
from topfarm.plotting import XYPlotComp
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.utils import regular_generic_layout, regular_generic_layout_gradients

#specifying the site and wind turbines to use
site = Hornsrev1Site()
wt = V80()
D = wt.diameter()
windFarmModel = BastankhahGaussian(site, wt)
n_wt = 16
boundary = [[-800,-50], [1200, -50], [1200,2300], [-800, 2300]]
stagger = 1 * D   #to create a staggered layout

def reg_func(sx, sy, rotation, **kwargs):
    x, y = regular_generic_layout(n_wt, sx, sy, stagger, rotation)
    return [x, y]

def reg_grad(sx, sy, rotation, **kwargs):
    dx_dsx, dy_dsx, dx_dsy, dy_dsy, dx_dr, dy_dr = regular_generic_layout_gradients(n_wt, sx, sy, stagger, rotation)
    return [[dx_dsx, dy_dsx], [dx_dsy, dy_dsy], [dx_dr, dy_dr]]

reg_grid_comp = CostModelComponent(input_keys=[('sx', 0),
                                               ('sy', 0),
                                               ('rotation', 0)],
                              n_wt=n_wt,
                              cost_function=reg_func,
                              cost_gradient_function = reg_grad,
                              output_keys= [('x', np.zeros(n_wt)), ('y', np.zeros(n_wt))],
                              objective=False
                              )

def aep_fun(x, y):
    aep = windFarmModel(x, y).aep().sum()
    return aep

daep = windFarmModel.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'])

aep_comp = CostModelComponent(input_keys=['x', 'y'],
                              n_wt=n_wt,
                              cost_function=aep_fun,
                              cost_gradient_function = daep,
                              output_keys= ("aep", 0),
                              output_unit="GWh",
                              maximize=True,
                              objective=True)

problem = TopFarmProblem(design_vars={'sx': (3*D, 2*D, 15*D),
                                      'sy': (4*D, 2*D, 15*D),
                                       'rotation': (50, 0, 90)
                                      },
                         constraints=[XYBoundaryConstraint(boundary),
                                      SpacingConstraint(4*D)],
                        grid_layout_comp=reg_grid_comp,
                        n_wt = n_wt,
                        cost_comp=aep_comp,
                        driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=200),
                        plot_comp=XYPlotComp(),
                        expected_cost=0.1,
                        )

x, y = reg_func(3*D, 3*D, 50)
plt.figure()
plt.plot(x,y,'.')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Staggered wind farm layout with inter-turbine spacing of 3D')
cost, state, recorder = problem.optimize(disp=False)
problem.evaluate()

plt.show()

