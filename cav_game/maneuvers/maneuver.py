from curses.has_key import has_key
import matplotlib.pyplot as plt
import shutil
import os 
import abc
import jax.numpy as jnp
from typing import List, Tuple, Dict
from cav_game.dynamics.car import BicycleDynamics

import pyomo as pyo

# Make sure that ipopt is installed
assert(shutil.which("ipopt") or os.path.isfile("ipopt"))

# Definition of maneuver class
class Maneuver(metaclass=abc.ABCMeta):
    """Class for defining a maneuver. This class is inherited by the other maneuvers and implements the 
    common functions with respect to the optimization problem using bicycle dynamics."""
    def __init__(self,vehicle: BicycleDynamics, x0: jnp.ndarray, params: dict, **kwargs):
        # Create vehicle object
        self.vehicle = vehicle
        # Define Mandatory Parameters
        self.cav_type = params["cav_type"] # CAV type can be "CAV1", "CAV2", or "CAVC"
        # Define initial state
        self._initial_state = x0     
        ## Define Optional Parameters
        # Optimization Parameters
        self.n = 250 if not params.has_key("n") else params["n"] # Number of time steps
        
        # Check that weights add up to 1
        assert(self.alpha_time + self.alpha_control + self.alpha_speed == 1, "Weights must add up to 1")
        opt_optiopns = {"acceptable_tol": 1e-8, 
                    "acceptable_obj_change_tol" : 1e-8,  
                    "max_iter" : 10000, 
                    "print_level" : 3}
        self.opt_options = opt_optiopns if not params.has_key("opt_options") else params["opt_options"] # Optimization options 
        
        ## Extract vehicle parameters
        self.v_bounds = self.vehicle.v_bounds # bounds on the velocity
        self.u_bounds = self.vehicle.u_bounds # bounds on the acceleration
        
        ## Define private variables
        # Flags
        self._is_feasible = False # Flag to indicate if the optimization problem is feasible
        # Terminal States
        self._terminal_time = 0 # Terminal time
        self._terminal_state = x0 # Terminal state
        # Model placeholder
        self._model = None # Placeholder for the model
        self._model_instance = None # Placeholder for the model instance
        self._results = None # Placeholder for the optimization results
        
        # Initialize solver
        self._opt = pyo.solverFactory('ipopt')
        self._discretizer = pyo.dae.TransformationFactory('dae.collocation')
    
    # Abstract methods
    
    @abc.abstractmethod
    def _define_model(self) -> None:
        "Function to define the optimization model with its variables"
        pass
    
    @abc.abstractmethod
    def _define_dae_constraints(self) -> None:
        "Function to define the DAE constraints"
        pass
    
    @abc.abstractmethod
    def _define_constraints(self) -> None:
        "Function to define the constraints"
        pass
    
    @abc.abstractmethod
    def _define_objective(self) -> None:
        "Function to define the objective"
        pass
    
    @abc.abstractmethod
    def _define_initial_conditions(self) -> None:
        "Function to define the initial conditions"
        pass
    
    @abc.abstractmethod
    def _solve(self) -> None:
        "Function to solve the optimization problem"
        pass
    

# Longitudinal Maneuver
class LongitudinalManeuver(Maneuver):
    def __init__(self,vehicle: BicycleDynamics, x0: jnp.ndarray, x0_obst: Dict[str, jnp.array], params: dict, **kwargs):
        """
        Class for defining a longitudinal maneuver. This class is inherited by the other maneuvers and 
        implements the common functions with respect to the optimization problem using bicycle dynamics.
        Inputs:
        vehicle: BicycleDynamics object
        x0: Initial state of the CAV
        x0_obst: Initial state of the obstacle vehicle, dictionary [obstacle location: initial state]   
                initial stat-> [x, y, v, theta] for each obstacle 
                obstacle_location-> location of the obstacle vehicle, e.g. "front", "back", "front left", "front right","rear left", "rear right", 
        """
        super().__init__(vehicle, x0, params, **kwargs)
        # Desired Terminal State
        self.v_des = 30 if not params.has_key("v_des") else params["v_des"] # Desired velocity [m/s]
        # Safety Parameters
        self.reaction_time = 0.6 if not params.has_key("reaction_time") else params["reaction_time"] # Vehicle Reaction time [s]
        self.min_safe_distance = 2 if not params.has_key("min_safe_distance") else params["min_safe_distance"] # Minimum inter-vehicle safety distance [m]
        # Long Tunning Parameters
        self.alpha_time = 0.1 if not params.has_key("alpha_time") else params["alpha_time"] # Time penalty weight
        self.alpha_control = 0.1 if not params.has_key("alpha_control") else params["alpha_control"] # Control penalty weight
        self.alpha_speed = 0.1 if not params.has_key("alpha_speed") else params["alpha_speed"] # Speed penalty weight
        self.t_max = 15 if not params.has_key("t_max") else params["t_max"] # Maximum time [s]
        self._initial_state_obst = x0_obst # List of initial states of the obstacle vehicles
        
        # Normalize the weights
        self._beta_u = self.alpha_control/(jnp.max(jnp.array(self.u_bounds)**2)) # Acceleration cost weight
        self._beta_t = self.alpha_time                 # Minimum Time weight
        self._beta_v = self.alpha_speed     # Desired velocity weight
        
        
        
        
        
        
    
              
        
        
        