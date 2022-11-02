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
                        "print_level": 3}
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
        # Check that weights add up to 1
        # assert (self.alpha_time + self.alpha_control + self.alpha_speed == 1, "Weights must add up to 1")

        self.t_max = params.get("t_max", 15)  # Maximum time [s]
        self._initial_state_obst = x0_obst  # List of initial states of the obstacle vehicles

        # Normalize the weights
        self._beta_u = self.alpha_control / (jnp.max(jnp.array(self.u_bounds) ** 2))  # Acceleration cost weight
        self._beta_t = self.alpha_time  # Minimum Time weight
        self._beta_v = self.alpha_speed  # Desired velocity weight

        # Define the optimization model
        self._define_model()
        self._define_constraints()
        self._define_dae_constraints()
        self._define_objective()
        self._define_initial_conditions()
        self._define_solver()
        # create model instance
        self._model_instance = self._model.create_instance()

    def compute_longitudinal_trajectory(self, plot: bool = True, save_path: str = "", obstacle: bool = False,
                                        show: bool = True) -> Tuple[bool, Dict[str, jnp.ndarray]]:
        """Function to compute the longitudinal trajectory of the CAV"""
        feasible = self._solve()
        # Extract the results
        trajectory = self._extract_results()
        # Generate the plots
        if plot:
            self._generate_trajectory_plots(trajectory, save_path, obstacle, show)

        return feasible, trajectory

    def _extract_results(self) ->  Dict[str, jnp.ndarray]:
        raise NotImplementedError("Define _extract_results in subclass")

    def _generate_trajectory_plots(self, traj: Dict, save_path: str = "", obstacle: bool = False, show: bool = False) -> Tuple[
        plt.figure, List[plt.axes]]:
        """Function to generate the plots of the trajectory and the control inputs"""
        model = self._model_instance

        # Extract the results
        tsim = traj["t"]
        xsim = traj["x"]
        vsim = traj["v"]
        usim = traj["u"]

        if obstacle:
            xsim_obst = traj["x_obst"]
            safety_distance = traj["safety_distance"]

        # Plot the trajectory

        fig = plt.figure(figsize=(10, 5))
        (ax1, ax2, ax3) = fig.subplots(3, sharex=True)
        fig.suptitle('Longitudinal Trajectory ' + self.cav_type)
        # Position vs Time
        ax1.plot(tsim, xsim, label='Ego Vehicle')
        if obstacle:
            ax1.plot(tsim, xsim_obst, label='Obstacle Vehicle', color='red')
            ax1.plot(tsim, safety_distance, label='Safety Distance', color='green', linestyle='-.')
            ax1.legend()
        ax1.grid(True, which='both')
        ax1.set_ylabel('Position [m]')
        # Velocity vs Time
        ax2.plot(tsim, vsim, label='Ego Vehicle')
        ax2.grid(True, which='both')
        ax2.set_ylabel('Velocity [m/s]')
        # Acceleration vs Time
        ax3.plot(tsim, usim, label='Ego Vehicle')
        ax3.set_ylabel('Acceleration [m/s^2]')
        ax3.set_xlabel('Time [s]')
        ax3.ylimits = self.u_bounds
        ax3.grid(True, which='both')

        if save_path:
            fig.savefig(save_path)

        if show:
            fig.tight_layout()
            fig.show()

        return (fig, [ax1, ax2, ax3])

    def _solve(self) -> bool:
        """Function to solve the optimization problem"""
        # Solve the optimization problem
        results = self._opt.solve(self._model_instance, tee=self.display_solver_output)
        # Check the results status       
        if not ((results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition == TerminationCondition.optimal)):
            warn("Optimal solution not found", results.solver.status)
            feasible = False
        else:
            print("Optimal solution found")
            feasible = True

        self._results = results

        return feasible
