import matplotlib.pyplot as plt
import shutil
import os
import abc
import jax.numpy as jnp
from typing import List, Tuple, Dict
from cav_game.dynamics.dynamics import ControlAffineDynamics
from warnings import warn
import pyomo as pyo
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import pao

# Make sure that ipopt is installed
assert (shutil.which("ipopt") or os.path.isfile("ipopt"))


# Definition of maneuver class
class Maneuver(metaclass=abc.ABCMeta):
    """Class for defining a maneuver. This class is inherited by the other maneuvers and implements the 
    common functions with respect to the optimization problem using bicycle dynamics."""

    def __init__(self, vehicle: ControlAffineDynamics, x0: jnp.ndarray, params: dict, **kwargs):
        # Create vehicle object
        self.vehicle = vehicle
        # Define Mandatory Parameters
        self.cav_type = params["cav_type"]  # CAV type can be "CAV1", "CAV2", or "CAVC"
        # Define initial state
        self._initial_state = x0
        ## Define Optional Parameters
        # Optimization Parameters
        self.n = params.get("n", 250)  # Number of time steps
        self.display_solver_output = params.get("display_solver_output", False)  # Display solver output
        self.diff_method = params.get("diff_method", "dae.collocation")  # Finite difference method
        opt_optiopns = {"acceptable_tol": 1e-8,
                        "acceptable_obj_change_tol": 1e-8,
                        "max_iter": 10000,
                        "print_level": 3,
                        "halt_on_ampl_error": "yes"}
        self.opt_options = params.get("opt_options", opt_optiopns)  # Optimization options

        ## Extract vehicle parameters
        self.v_bounds = self.vehicle.v_bounds  # bounds on the velocity
        self.u_bounds = self.vehicle.u_bounds  # bounds on the acceleration

        ## Define private variables
        # Flags
        self._is_feasible = False  # Flag to indicate if the optimization problem is feasible
        # Terminal States
        self._terminal_time = 0  # Terminal time
        self._terminal_state = x0  # Terminal state
        # Model placeholder
        self._model = None  # Placeholder for the model
        self._model_instance = None  # Placeholder for the model instance
        self._results = None  # Placeholder for the optimization results

    # Abstract methods
    @abc.abstractmethod
    def _define_solver(self) -> None:
        """Initialize the solver."""
        # Initialize solver
        pass
    @staticmethod
    def _rk4( f, x0, u, dt):
        """Runge-Kutta 4th order integration."""
        k1 = f(x0, u)
        k2 = f(x0 + 0.5 * dt * k1, u)
        k3 = f(x0 + 0.5 * dt * k2, u)
        k4 = f(x0 + dt * k3, u)
        return x0 + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @abc.abstractmethod
    def _define_model(self) -> None:
        """Function to define the optimization model with its variables"""
        pass

    @abc.abstractmethod
    def _define_dae_constraints(self) -> None:
        """Function to define the DAE constraints"""
        pass

    @abc.abstractmethod
    def _define_constraints(self) -> None:
        """Function to define the constraints"""
        pass

    @abc.abstractmethod
    def _define_objective(self) -> None:
        """Function to define the objective"""
        pass

    @abc.abstractmethod
    def _define_initial_conditions(self) -> None:
        """Function to define the initial conditions"""
        pass

    @abc.abstractmethod
    def _solve(self) -> bool:
        """Function to solve the optimization problem"""
        pass

    @abc.abstractmethod
    def _extract_results(self) -> Dict[str, jnp.ndarray]:
        """Function to extract the results of the optimization problem"""
        pass


# Longitudinal Maneuver
class LongitudinalManeuver(Maneuver):
    def __init__(self, vehicle: ControlAffineDynamics, x0: jnp.ndarray, x0_obst: Dict[str, jnp.array], params: dict,
                 **kwargs):  # TODO: change control affine dynamics to just vehicle
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
        self.v_des = params.get("v_des", 30)  # Desired velocity [m/s]
        # Safety Parameters
        self.reaction_time = params.get("reaction_time", 0.6)  # Vehicle Reaction time [s]
        self.min_safe_distance = params.get("min_safe_distance", 2)  # Minimum inter-vehicle safety distance [m]
        # Long Tunning Parameters
        self.alpha_time = params.get("alpha_time", 0.1)  # Time penalty weight
        self.alpha_control = params.get("alpha_control", 0.3)  # Control penalty weight
        self.alpha_speed = params.get("alpha_speed", 0.6)  # Speed penalty weight

        self.t_max = params.get("t_max", 15)  # Maximum time [s]
        self._initial_state_obst = x0_obst  # List of initial states of the obstacle vehicles

        # Normalize the weights
        max_u = max([(self.u_bounds[0]) ** 2, (self.u_bounds[1]) ** 2])
        max_delta_v = max([(self.v_bounds[0] - self.v_des) ** 2, (self.v_bounds[1] - self.v_des) ** 2])
        self._beta_u = self.alpha_control / max_u  # Acceleration cost weight
        self._beta_t = self.alpha_time  # Time cost weight
        self._beta_v = self.alpha_speed  # Desired velocity weight


    def compute_longitudinal_trajectory(self, plot: bool = True, save_path: str = "", obstacle: bool = False,
                                        show: bool = True, **kwargs) -> Tuple[bool, Dict[str, jnp.ndarray]]:
        """Function to compute the longitudinal trajectory of the CAV"""
        feasible = self._solve()
        # Extract the results
        trajectory = self._extract_results()
        # Generate the plots
        if plot:
            self._generate_trajectory_plots(trajectory, save_path, obstacle, show, **kwargs)

        return feasible, trajectory

    def _extract_results(self) ->  Dict[str, jnp.ndarray]:
        raise NotImplementedError("Define _extract_results in subclass")

    @abc.abstractmethod
    def _generate_trajectory_plots(self, trajectory, save_path: str = "", obstacle: bool = False, show: bool = False,
                                   **kwargs) -> Tuple[plt.figure, List[plt.axes]]:
        """Function to generate the trajectory plots"""
        pass

    def _solve(self) -> bool:
        """Function to solve the optimization problem"""
        # Solve the optimization problem
        results = self._opt.solve(self._model_instance, tee=self.display_solver_output)
        # Check the results status       
        if not ((results.solver.status == SolverStatus.ok) or not (
                results.solver.termination_condition == TerminationCondition.optimal)):
            warn("Optimal solution not found", results.solver.status)
            feasible = False
        else:
            print("Optimal solution found")
            feasible = True

        self._results = results

        return feasible

    def _define_solver(self) -> None:
        """Initialize the solver."""
        # Initialize solver
        self._opt = pao.Solver('ipopt', **self.opt_options)
        self._discretizer = None
