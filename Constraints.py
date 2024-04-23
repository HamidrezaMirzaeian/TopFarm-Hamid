import importlib
import numpy as np
import matplotlib.pyplot as plt

from topfarm import TopFarmProblem
from topfarm.plotting import XYPlotComp
from topfarm.constraint_components.boundary import XYBoundaryConstraint, CircleBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.cost_models.cost_model_wrappers import CostModelComponent

# set up a "boundary" array with arbitrary points for use in the example
boundary = np.array([(0, 0), (1, 1), (3, 0), (3, 2), (0, 2)])

# set up dummy design variables and cost model component.
# This example includes 2 turbines (n_wt=2) located at x,y=0.5,0.5 and 1.5,1.5 respectively

x = [0.5,1.5]
y = [.5,1.5]
dummy_cost = CostModelComponent(input_keys=[],
                                n_wt=2,
                               cost_function=lambda : 1)

# We introduce a simple plotting function so we can quickly plot different types of site boundaries
def plot_boundary(name, constraint_comp):
    tf = TopFarmProblem(
        design_vars={'x':x, 'y':y}, # setting up the turbine positions as design variables
        cost_comp=dummy_cost, # using dummy cost model
        constraints=[constraint_comp], # constraint set up for the boundary type provided
        plot_comp=XYPlotComp()) # support plotting function

    plt.figure()
    plt.title(name)
    tf.plot_comp.plot_constraints() # plot constraints is a helper function in topfarm to plot constraints
    plt.plot(boundary[:,0], boundary[:,1],'.r', label='Boundary points') # plot the boundary points
    plt.axis('equal')
    plt.legend() # add the legend

plot_boundary('convex_hull', XYBoundaryConstraint(boundary, 'convex_hull'))
plt.show()
