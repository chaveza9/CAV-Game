import os
import sys
import numpy as np
import shutil
import os
import jax.numpy as jnp
sys.path.append(os.path.abspath('..'))

from cav_game.dynamics.car import DoubleIntegratorDynamics, BicycleDynamics
from cav_game.maneuvers.selfish import SelfishManeuver

import pyomo as pyo
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition

# Make sure that ipopt is installed
assert (shutil.which("ipopt") or os.path.isfile("ipopt"))

"""Define Vehicle Initial Conditions"""
# RIGHT LANE
# Vehicle C initial conditions
x0_c = 5  # Vehicle c initial x position [m]
v0_c = 24  # Vehicle c initial velocity [m/s]
X0_c = np.array([x0_c, v0_c])
# Obstacle U initial conditions
x0_u = 75  # Vehicle u initial x position [m]
v0_u = 17  # Vehicle u initial velocity [m/s]
X0_u = np.array([x0_u, v0_u])
# LEFT LANE
# Obstacle 1 initial conditions
x0_1 = 22  # Vehicle 1 initial x position [m]
v0_1 = 29  # Vehicle 1 initial velocity [m/s]
X0_1 = np.array([x0_1, v0_1])
# Obstacle 2 initial conditions
x0_2 = 0  # Vehicle 2 initial x position [m]
v0_2 = 29  # Vehicle 2 initial velocity [m/s]
X0_2 = np.array([x0_2, v0_2])


"""Initialize Vehicle Parameters"""
# Obstacle location descriptor
obstacles_c = {"front": X0_u, "front_left": X0_1, "rear_left": X0_2}
obstacles_1 = {"front_right": X0_u, "rear_right": X0_c, "back": X0_2}
obstacles_2 = {"front_right": X0_c, "front": X0_1}
# Construct optimization parameters
maneuver_params = {"cav_type": "CAVC",
                   "alpha_time": 0.2,
                   "alpha_control": 0.1,
                   "alpha_speed": 0.7,
                   "n": 300,
                   "diff_method": 'dae.finite_difference',
                   "alpha_control": 0.1,
                   "alpha_time": 0.05,
                   "alpha_speed": 0.8}
# Construct vehicle models
veh_params = {}
veh_c = BicycleDynamics(veh_params)
veh_1 = BicycleDynamics(veh_params)
veh_2 = BicycleDynamics(veh_params)
veh_u = BicycleDynamics(veh_params)
# Construct maneuver
long_maneuver_c = SelfishManeuver(veh_c, x0=X0_c, x0_obst=obstacles_c, params=maneuver_params)

"""Compute Trajectory Maneuver"""
feasible, trajectory_c = long_maneuver_c.compute_longitudinal_trajectory(obstacle=True)






"""Relax Maneuver"""
# Extract previous terminal time
tf = trajectory_c['t'][-1]
feasible, trajectory_c = long_maneuver_c.relax_terminal_time(time=tf * 10, obstacle=True)

# Extract previous terminal time
tf = trajectory_c['t'][-1]
# Extract terminal position
xf = trajectory_c['x'][-1]

"""Compute Trajectory Maneuver"""



