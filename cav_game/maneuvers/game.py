import jax.numpy as jnp
import pyomo as pyo
from pyomo.core.expr import differentiate
import pyomo.environ as pe
import pao
import matplotlib.pyplot as plt

from .maneuver import LongitudinalManeuver
from cav_game.dynamics.car import ControlAffineDynamics
from typing import List, Tuple, Dict
from warnings import warn


class DualLagrangianGame(LongitudinalManeuver):
    def __init__(self, vehicle: ControlAffineDynamics, time: float, xf_c: float,
                 x0: jnp.ndarray, x0_obst: Dict[str, jnp.array], params: dict,
                 **kwargs):
        self.terminal_time = time
        self.terminal_position = xf_c
        #  initialize the parent class
        super().__init__(vehicle, x0, x0_obst, params, **kwargs)

    def _define_solver(self) -> None:
        """Initialize the solver."""
        # Initialize solver
        nlp = self._opt = pao.Solver('ipopt', **self.opt_options)
        self._opt = pao.Solver('pao.pyomo.REG', nlp_solver=pao.Solver('ipopt'))
        self._discretizer = None

    def _define_model(self) -> None:
        # ----------------- Main Model -----------------
        # Initialize a parametric model
        model = pe.ConcreteModel()

        # Define model parameters
        model.t = pe.RangeSet(0, self.n)  # Time Discretization
        model.dt = pe.Param(initialize=self.terminal_time / self.n, mutable=True)  # Time Step
        model.x0 = pe.Param(initialize=self._initial_state[0], mutable=True)  # Initial position
        model.v0 = pe.Param(initialize=self._initial_state[1], mutable=True)  # Initial speed
        # Define vehicle variables
        model.x = pe.Var(model.t)  # Position [m]
        model.v = pe.Var(model.t, bounds=(self.v_bounds[0], self.v_bounds[1]))  # Velocity [m/s]
        model.u = pe.Var(model.t, bounds=(self.u_bounds[0], self.u_bounds[1]),
                         initialize=3)  # Acceleration [m/s^2]
        # ------------------ Obstacle Model ------------------
        # Define sub-model parameters
        model.x0_obst = pe.Param(initialize=self._initial_state_obst['rear'][0])  # Initial position
        model.v0_obst = pe.Param(initialize=self._initial_state_obst['rear'][1])  # Initial speed

        # Define obstacle variables
        model.x_obst = pe.Var(model.t)  # Position [m]
        model.v_obst = pe.Var(model.t, bounds=(self.v_bounds[0], self.v_bounds[1]))  # Velocity [m/s]
        model.u_obst = pe.Var(model.t, bounds=(self.u_bounds[0], self.u_bounds[1]),
                              initialize=3)  # Acceleration [m/s^2]

        # Define dual variables
        model.lamda_safety = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for safety constraint
        model.lambda_v_max = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for velocity constraint
        model.lambda_v_min = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for velocity constraint
        model.lambda_u_max = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for acceleration constraint
        model.lambda_u_min = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for acceleration constraint
        model.mu_x = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for position equality constraint
        model.mu_v = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for velocity equality constraint

        self._model = model

    def _define_dae_constraints(self) -> None:
        """Define the differential algebraic constraints."""
        model = self._model

        # ----------------- Main Model -----------------
        # Define differential algebraic equations
        def ode_x(m, k):
            if k < self.n:
                return m.x[k + 1] == m.x[k] + m.v[k] * m.dt
            else:
                return pe.Constraint.Skip

        model.ode_x = pe.Constraint(model.t, rule=ode_x)

        def ode_v(m, k):
            if k < self.n:
                return m.v[k + 1] == m.v[k] + m.u[k] * m.dt
            else:
                return pe.Constraint.Skip

        model.ode_v = pe.Constraint(model.t, rule=ode_v)

        # ----------------- Obstacle Model -----------------
        # Define differential algebraic equations
        def ode_x_obst(m, k):
            if k < self.n:
                return m.x_obst[k + 1] == m.x_obst[k] + m.v_obst[k] * m.dt
            else:
                return pe.Constraint.Skip

        model.ode_x_obst = pe.Constraint(model.t, rule=ode_x_obst)

        def ode_v_obst(m, t):
            if t < self.n:
                return m.v_obst[t + 1] == m.v_obst[t] + m.u_obst[t] * m.dt
            else:
                return pe.Constraint.Skip

        model.ode_v_obst = pe.Constraint(model.t, rule=ode_v_obst)

    def _define_constraints(self) -> None:
        """Create model constraints """
        model = self._model

        # Define safety constraints
        def safety_distance(m, t):
            return m.x[t] - m.x_obst[t] >= self.reaction_time * m.sub.v_obst[t] + self.min_safe_distance

        model.safety = pe.Constraint(model.t, rule=safety_distance)

        # Define terminal constraints
        model.xf = pe.Constraint(expr=model.x[self.n] >= self.terminal_position + 15.0)

    def _define_initial_conditions(self) -> None:
        """Define the initial conditions."""
        model = self._model
        # Main Model
        # Define initial conditions
        model.x[0].fix(model.x0)
        model.v[0].fix(model.v0)

        # Sub Model
        # Define initial conditions
        model.x_obst[0].fix(model.x0_obst)
        model.v_obst[0].fix(model.v0_obst)

    def _define_objective(self) -> None:
        """Define the objective function."""
        model = self._model
        # ----------------- Main Model -----------------
        # Define objective function expressions
        model.speed_objective = sum(float(self._beta_v) * (model.v[t] - self.v_des) ** 2 for t in model.t)
        model.acceleration_objective = sum(model.dt * float(self._beta_u) * (model.u[t]) ** 2 for t in model.t)
        # Influence Objective
        model.opponent_objective_pos = sum(model.dt * (model.x_obst[t] - self.terminal_position -
                                                       model.v_obst[
                                                           t] * self.reaction_time - self.min_safe_distance) ** 2
                                           for t in model.t)
        model.opponent_objective_vel = sum(model.dt * (model.v_obst[t] - self.v_des) ** 2 for t in model.t)
        # Define objective function ego vehicle expression
        model.ego_objective = model.speed_objective + model.acceleration_objective + \
                              model.opponent_objective_pos + model.opponent_objective_vel

        # ----------------- Obstacle Model -----------------
        # Define objective function expressions
        model.speed_objective_obst = sum(self._beta_v * (model.v_obst[t] - self.v_des) ** 2 for t in model.t)
        model.acceleration_objective_obst = sum(model.dt * float(self._beta_u) * (model.u_obst[t]) ** 2
                                                for t in model.t)
        # Define objective function ego vehicle expression
        model.obst_objective = model.speed_objective_obst + model.acceleration_objective_obst

        model.obj = pe.Objective(rule=model.ego_objective + model.obst_objective, sense=pe.minimize)

    def _define_lagrangian(self) -> None:
        model = self._model
        discrete_position = lambda m, t: m.x[t - 1] + m.v[t] * m.dt + 0.5 * m.u[t] * m.dt ** 2
        discrete_position_obst = lambda m, t: m.x_obst[t - 1] + m.v_obst[t] * m.dt + 0.5 * m.u_obst[t] * m.dt ** 2
        discrete_velocity_obst = lambda m, t: m.v_obst[t - 1] + m.u_obst[t] * m.dt
        # Lagrangian Objective
        model.lagrangian_objective = model.obst_objective
        # Lagrangian Equality Constraints
        model.lagrangian_position_constraint = sum(model.mu_x[t] * (discrete_position_obst(model, t))
                                                   for t in model.t if t > 1)
        model.lagrangian_velocity_constraint = sum(model.mu_v[t] * (discrete_velocity_obst(model, t))
                                                   for t in model.t if t > 1)
        # Lagrangian Inequality Constraints
        safety_function = lambda m, t: -(discrete_position(m, t) - discrete_position_obst(m, t)) + \
                                       self.reaction_time * discrete_velocity_obst(m, t) + self.min_safe_distance
        model.lagrangian_safety_constraint = sum(model.mu_s[t] * safety_function(model, t) for t in model.t)
        # Lagrangian's actuation constraints
        model.lagrangian_v_min = sum(model.mu_v_min[t] * (-discrete_velocity_obst(model, t) + self.v_bounds[0])
                                     for t in model.t)
        model.lagrangian_v_max = sum(model.mu_v_max[t] * (discrete_velocity_obst(model, t) - self.v_bounds[1])
                                     for t in model.t)
        model.lagrangian_u_min = sum(model.mu_u_min[t] * (-model.u_obst + self.u_bounds[0]) for t in model.t)
        model.lagrangian_u_max = sum(model.mu_u_max[t] * (model.u_obst - self.u_bounds[1]) for t in model.t)
        # Add lagrangian terms
        model.lagrangian = model.lagrangian_objective + model.lagrangian_position_constraint + \
                           model.lagrangian_velocity_constraint + model.lagrangian_safety_constraint + \
                           model.lagrangian_v_min + model.lagrangian_v_max + model.lagrangian_u_min + \
                           model.lagrangian_u_max

    def _add_kkt_constraints(self) -> None:
        model = self._model
        # KKT Conditions
        # Add stationary conditions
        model.kkt_stationary = pe.ConstraintList()
        for k in model.t:
            model.kkt_stationary.add(expr=differentiate(model.u_obst[k], wrt=model.u_obst[k]) == 0)

        # Add complementary slackness conditions
        model.kkt_complementary_slackness = pe.ConstraintList()

        model.kkt_position = pe.Constraint(expr=model.lagrangian_position_constraint == 0)
        model.kkt_velocity = pe.Constraint(expr=model.lagrangian_velocity_constraint == 0)
        model.kkt_safety = pe.Constraint(expr=model.lagrangian_safety_constraint == 0)
        model.kkt_v_min = pe.Constraint(expr=model.lagrangian_v_min == 0)
        model.kkt_v_max = pe.Constraint(expr=model.lagrangian_v_max == 0)
        model.kkt_u_min = pe.Constraint(expr=model.lagrangian_u_min == 0)
        model.kkt_u_max = pe.Constraint(expr=model.lagrangian_u_max == 0)

    def _extract_results(self) -> Dict[str, jnp.ndarray]:
        """Extract the solution from the solver."""
        model = self._model_instance

        trajectory = dict()
        # Extract solution
        trajectory['t'] = [t * model.tf for t in model.t]
        # Ego Vehicle
        trajectory['x_ego'] = [model.x[i]() for i in model.t]
        trajectory['v_ego'] = [model.v[i]() for i in model.t]
        trajectory['u_ego'] = [model.u[i]() for i in model.t]
        # Opponent Vehicle
        trajectory['x_opponent'] = [model.x_obst[i]() for i in model.t]
        trajectory['v_opponent'] = [model.v_obst[i]() for i in model.t]
        trajectory['u_opponent'] = [model.u_obst[i]() for i in model.t]

        return trajectory

    def _generate_trajectory_plots(self, save_path: str = "", obstacle: bool = False, show: bool = False) \
            -> Tuple[plt.figure, List[plt.axes]]:
        """Function to generate the plots of the trajectory and the control inputs"""
        model = self._model_instance

        # Extract the results
        tsim = self.trajectory['t']
        xsim = self.trajectory['x_ego']
        vsim = self.trajectory['v_ego']
        usim = self.trajectory['u_ego']

        if obstacle:
            xsim_opponent = self.trajectory['x_opponent']
            vsim_opponent = self.trajectory['v_opponent']
            usim_opponent = self.trajectory['u_opponent']
            safety_distance = xsim_opponent - self.reaction_time * vsim_opponent - self.min_safe_distance
        # Plot the trajectory
        plt.rcParams[r'text.usetex'] = True
        fig = plt.figure(figsize=(10, 5))
        (ax1, ax2, ax3) = fig.subplots(3, sharex=True)
        fig.suptitle('Longitudinal Trajectory ' + self.cav_type)
        # Position vs Time
        ax1.plot(tsim, xsim, label='Ego Vehicle')
        ax1.grid(True, which='both')
        ax1.set_ylabel(r'Position $displaystyle{\m\]}$')
        # Velocity vs Time
        ax2.plot(tsim, vsim, label='Ego Vehicle')
        ax2.grid(True, which='both')
        ax2.set_ylabel(r'Velocity $displaystyle{\[\frac{m}{s}\]}$')
        # Acceleration vs Time
        ax3.plot(tsim, usim, label='Ego Vehicle')
        ax3.set_ylabel(r'Acceleration $displaystyle{\[\frac{m}{s^2}\]}$')
        ax3.set_xlabel('Time [s]')
        ax3.ylimits = self.u_bounds
        ax3.grid(True, which='both')

        if obstacle:
            ax1.plot(tsim, xsim_opponent, label='Opponent Vehicle', color='red')
            ax1.plot(tsim, safety_distance, label='Safety Distance', color='green', linestyle='-.')
            ax1.legend()
            ax2.plot(tsim, vsim_opponent, label='Opponent Vehicle', color='red')
            ax3.plot(tsim, usim_opponent, label='Opponent Vehicle', color='red')

        if save_path:
            fig.savefig(save_path)

        if show:
            fig.tight_layout()
            fig.show()

        return fig, [ax1, ax2, ax3]
