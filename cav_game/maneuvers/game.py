import jax.numpy as jnp
import numpy as np
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
        model.jerk = pe.Var(model.t, bounds=(-0.8, 0.8))  # Jerk [m/s^3]
        # ------------------ Obstacle Model ------------------
        # Define sub-model parameters
        model.x0_obst = pe.Param(initialize=self._initial_state_obst['rear'][0])  # Initial position
        model.v0_obst = pe.Param(initialize=self._initial_state_obst['rear'][1])  # Initial speed

        # Define obstacle variables
        model.x_obst = pe.Var(model.t)  # Position [m]
        model.v_obst = pe.Var(model.t, bounds=(self.v_bounds[0], self.v_bounds[1]))  # Velocity [m/s]
        model.u_obst = pe.Var(model.t, bounds=(self.u_bounds[0], self.u_bounds[1]),
                              initialize=3)  # Acceleration [m/s^2]
        model.jerk_obst = pe.Var(model.t, bounds=(-0.8, 0.8))  # Jerk [m/s^3]

        # Define dual variables
        model.lamda_safety = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for safety constraint
        model.lambda_v_max = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for velocity constraint
        model.lambda_v_min = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for velocity constraint
        model.lambda_u_max = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for acceleration constraint
        model.lambda_u_min = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for acceleration constraint
        model.mu_x = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for position equality constraint
        model.mu_v = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for velocity equality constraint
        model.mu_jerk = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for jerk equality constraint

        # Define Relaxation Variables
        model.epsilon = pe.Param(initialize=1)  # Relaxation variable lagrangian multiplier
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

        def ode_jerk(m, k):
            if k < self.n:
                return m.jerk[k + 1] == m.u[k + 1] - m.u[k]
            else:
                return pe.Constraint.Skip

        model.ode_jerk = pe.Constraint(model.t, rule=ode_jerk)

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

        def ode_jerk_obst(m, k):
            if k < self.n:
                return m.jerk_obst[k + 1] == m.u_obst[k + 1] - m.u_obst[k]
            else:
                return pe.Constraint.Skip

        model.ode_u_obst = pe.Constraint(model.t, rule=ode_jerk_obst)

    def _define_constraints(self) -> None:
        """Create model constraints """
        model = self._model

        # Define safety constraints
        def safety_distance(m, t):
            return m.x[t] - m.x_obst[t] >= self.reaction_time * m.v_obst[t] + self.min_safe_distance

        model.safety = pe.Constraint(model.t, rule=safety_distance)

        # Define terminal constraints
        model.xf = pe.Constraint(expr=model.x[self.n] >= self.terminal_position + 15.0)

        # Add Lagrangian constraints
        self._define_lagrangian()

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
        model.influence_objective_pos = sum(model.dt * -(self.terminal_position - 15 - model.x_obst[t]) ** 2
                                            for t in model.t)
        model.influence_objective_vel = sum(
            model.dt * float(self._beta_v) * (model.v_obst[t] - self.v_des) ** 2 for t in model.t)
        # Define objective function ego vehicle expression
        model.ego_objective = model.speed_objective + model.acceleration_objective + \
                              model.influence_objective_pos + model.influence_objective_vel

        # ----------------- Obstacle Model -----------------
        # Define objective function expressions
        model.speed_objective_obst = sum(float(self._beta_v) * (model.v_obst[t] - self.v_des) ** 2 for t in model.t)
        model.acceleration_objective_obst = sum(model.dt * float(self._beta_u) * (model.u_obst[t]) ** 2
                                                for t in model.t)
        # Define objective function ego vehicle expression
        model.obst_objective = model.speed_objective_obst + model.acceleration_objective_obst

        model.obj = pe.Objective(rule=model.ego_objective + model.obst_objective, sense=pe.minimize)

    def _define_lagrangian(self) -> None:
        model = self._model
        discrete_position = lambda m, t: m.x[t - 1] + m.v[t] * m.dt + 0.5 * m.u[t] * m.dt ** 2 if t > 0 else m.x[t]
        discrete_position_obst = lambda m, t: m.x_obst[t - 1] + m.v_obst[t] * m.dt + 0.5 * m.u_obst[t] * m.dt ** 2 \
            if t > 0 else m.x_obst[t]
        discrete_velocity_obst = lambda m, t: m.v_obst[t - 1] + m.u_obst[t] * m.dt if t > 0 else m.v_obst[t]
        # Lagrangian Inequality Constraints
        safety_function = lambda m, t: -(discrete_position(m, t) - discrete_position_obst(m, t)) + \
                                       self.reaction_time * discrete_velocity_obst(m, t) + self.min_safe_distance

        # KKT Conditions
        # Objective terms
        accel_obj_lambda = lambda m, t: m.u_obst[t] * m.dt
        speed_obj_lambda = lambda m, t: (m.v_obst[t - 1] - self.v_des) * m.dt + m.dt ** 2 if t > 0 else 0
        # Equality terms
        position_constraint_lambda = lambda m, t: m.mu_x[t] * 0.5 * m.dt ** 2
        velocity_constraint_lambda = lambda m, t: m.mu_v[t] * m.dt
        jerk_constraint_lambda = lambda m, t: m.mu_jerk[t] * 1/m.dt
        # Inequality terms
        velocity_lim_lambda = lambda m, t: -m.lambda_v_min[t] * m.dt + m.lambda_v_max[t] * m.dt
        acceleration_lim_lambda = lambda m, t: -m.lambda_u_min[t] * m.dt + m.lambda_u_max[t] * m.dt
        safety_lambda = lambda m, t: m.lamda_safety[t] * (0.5 * m.dt ** 2 + self.reaction_time * m.dt)
        lagrangian_dot = lambda m, t: accel_obj_lambda(m, t) + speed_obj_lambda(m, t) + jerk_constraint_lambda(m, t) +\
                                      position_constraint_lambda(m, t) + velocity_constraint_lambda(m, t) + \
                                      velocity_lim_lambda(m, t) + acceleration_lim_lambda(m, t) + \
                                      safety_lambda(m, t) == 0

        model.kkt_stationary = pe.Constraint(model.t, rule=lagrangian_dot)
        # Add complementary slackness conditions
        # Safety Constraint
        model.kkt_safety = pe.ConstraintList()
        for k in model.t:
            model.kkt_safety.add(expr=model.lamda_safety[k] * safety_function(model, k) >= -model.epsilon)

        # Velocity Bounds
        model.kkt_v_min = pe.ConstraintList()
        for k in model.t:
            model.kkt_v_min.add(expr=model.lambda_v_min[k] * (-discrete_velocity_obst(model, k) + self.v_bounds[0])
                                     >= -model.epsilon)

        model.kkt_v_max = pe.ConstraintList()
        for k in model.t:
            model.kkt_v_max.add(expr=model.lambda_v_max[k] * (discrete_velocity_obst(model, k) - self.v_bounds[1])
                                     >= -model.epsilon)

        # Actuation Bounds
        model.kkt_u_min = pe.ConstraintList()
        for k in model.t:
            model.kkt_u_min.add(expr=model.lambda_u_min[k] * (-model.u_obst[k] + self.u_bounds[0])
                                     >= -model.epsilon)

        model.kkt_u_max = pe.ConstraintList()
        for k in model.t:
            model.kkt_u_max.add(expr=model.lambda_u_max[k] * (model.u_obst[k] - self.u_bounds[1])
                                     >= -model.epsilon)

    def _extract_results(self) -> Dict[str, jnp.ndarray]:
        """Extract the solution from the solver."""
        model = self._model_instance

        trajectory = dict()
        # Extract solution
        trajectory['t'] = [t * model.dt() for t in model.t]
        # Ego Vehicle
        trajectory['x_ego'] = [model.x[i]() for i in model.t]
        trajectory['v_ego'] = [model.v[i]() for i in model.t]
        trajectory['u_ego'] = [model.u[i]() for i in model.t]
        # Opponent Vehicle
        trajectory['x_opponent'] = [model.x_obst[i]() for i in model.t]
        trajectory['v_opponent'] = [model.v_obst[i]() for i in model.t]
        trajectory['u_opponent'] = [model.u_obst[i]() for i in model.t]

        return trajectory

    def _generate_trajectory_plots(self, trajectory, save_path: str = "", obstacle: bool = False, show: bool = False) \
            -> Tuple[plt.figure, List[plt.axes]]:
        """Function to generate the plots of the trajectory and the control inputs"""
        model = self._model_instance

        # Extract the results
        tsim = trajectory['t']
        xsim = trajectory['x_ego']
        vsim = trajectory['v_ego']
        usim = trajectory['u_ego']

        if obstacle:
            xsim_opponent = trajectory['x_opponent']
            vsim_opponent = trajectory['v_opponent']
            usim_opponent = trajectory['u_opponent']
            safety_distance = jnp.array(xsim) - \
                              self.reaction_time * jnp.array(vsim_opponent) - self.min_safe_distance
        # Plot the trajectory
        plt.rcParams[r'text.usetex'] = True
        fig = plt.figure(figsize=(10, 5))
        (ax1, ax2, ax3) = fig.subplots(3, sharex=True)
        fig.suptitle('Longitudinal Trajectory ' + self.cav_type)
        # Position vs Time
        ax1.plot(tsim, xsim, label='Ego Vehicle')
        ax1.grid(True, which='both')
        ax1.set_ylabel(r'Position $[m]$')
        # Velocity vs Time
        ax2.plot(tsim, vsim, label='Ego Vehicle')
        ax2.grid(True, which='both')
        ax2.set_ylabel(r'Velocity $[m/s]$')
        # Acceleration vs Time
        ax3.plot(tsim, usim, label='Ego Vehicle')
        ax3.set_ylabel(r'Acceleration $ [ m/s^2 ]$')
        ax3.set_xlabel(r'Time s')
        ax3.set_ylim([min(usim) - 0.5, max(usim) + 0.5])
        ax3.grid(True, which='both')

        if obstacle:
            ax1.plot(tsim, xsim_opponent, label='Opponent Vehicle', color='red')
            ax1.plot(tsim, safety_distance, label='Safety Distance', color='green', linestyle='-.')
            ax1.legend()
            ax2.plot(tsim, vsim_opponent, label='Opponent Vehicle', color='red')
            ax3.plot(tsim, usim_opponent, label='Opponent Vehicle', color='red')
            ax3.set_ylim([min(usim + usim_opponent) - 0.5, max(usim + usim_opponent) + 0.5])

        if save_path:
            fig.savefig(save_path)

        if show:
            fig.tight_layout()
            fig.show()

        return fig, [ax1, ax2, ax3]
