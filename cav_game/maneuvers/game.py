import jax.numpy as jnp
import pyomo as pyo
import pyomo.environ as pe
import pao
import matplotlib.pyplot as plt

from .maneuver import LongitudinalManeuver
from cav_game.dynamics.car import ControlAffineDynamics
from typing import List, Tuple, Dict


class DualGame(LongitudinalManeuver):
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
        # Initialize a parametric sub-model
        model.sub = pao.pyomo.SubModel(fixed=[model.x, model.v, model.u])
        model.sub.dt = pe.Param(initialize=self.terminal_time / self.n, mutable=True)
        # Define sub-model parameters
        model.sub.x0_obst = pe.Param(initialize=self._initial_state_obst['rear'][0])  # Initial position
        model.sub.v0_obst = pe.Param(initialize=self._initial_state_obst['rear'][1])  # Initial speed

        # Define obstacle variables
        model.sub.x_obst = pe.Var(model.t)  # Position [m]
        model.sub.v_obst = pe.Var(model.t, bounds=(self.v_bounds[0], self.v_bounds[1]))  # Velocity [m/s]
        model.sub.u_obst = pe.Var(model.t, bounds=(self.u_bounds[0], self.u_bounds[1]),
                                  initialize=3)  # Acceleration [m/s^2]

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

        model.sub.ode_x = pe.Constraint(model.t, rule=ode_x_obst)

        def ode_v_obst(m, t):
            if t < self.n:
                return m.v_obst[t + 1] == m.v_obst[t] + m.u_obst[t] * m.dt
            else:
                return pe.Constraint.Skip

        model.sub.ode_v = pe.Constraint(model.t, rule=ode_v_obst)

    def _define_constraints(self) -> None:
        """Create model constraints """
        model = self._model

        # Define safety constraints
        def safety_distance(m, t):
            return m.x[t] - m.sub.x_obst[t] >= self.reaction_time * m.sub.v_obst[t] + self.min_safe_distance

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
        model.sub.x_obst[0].fix(model.sub.x0_obst)
        model.sub.v_obst[0].fix(model.sub.v0_obst)

    def _define_objective(self) -> None:
        """Define the objective function."""
        model = self._model
        # ----------------- Main Model -----------------
        # Define objective function expressions
        model.speed_objective = sum(float(self._beta_v) * (model.v[t] - self.v_des) ** 2 for t in model.t)
        model.acceleration_objective = sum(model.dt * float(self._beta_u) * (model.u[t]) ** 2 for t in model.t)
        # Influence Objective
        model.opponent_objective_pos = sum(model.dt * (model.sub.x_obst[t] - self.terminal_position -
                                                       model.sub.v_obst[
                                                           t] * self.reaction_time - self.min_safe_distance) ** 2
                                           for t in model.t)
        model.opponent_objective_vel = sum(model.dt * (model.sub.v_obst[t] - self.v_des) ** 2 for t in model.t)
        # Define objective function
        model.obj = pe.Objective(rule=model.speed_objective + model.acceleration_objective +
                                      model.opponent_objective_pos + model.opponent_objective_vel, sense=pe.minimize)

        # ----------------- Obstacle Model -----------------
        # Define objective function expressions
        model.sub.speed_objective = sum(self._beta_v * (model.sub.v_obst[t] - self.v_des) ** 2 for t in model.t)
        model.sub.acceleration_objective = sum(model.dt*float(self._beta_u) * (model.sub.u_obst[t]) ** 2
                                               for t in model.t)
        # Define objective function
        model.sub.obj = pe.Objective(rule=model.sub.speed_objective + model.sub.acceleration_objective,
                                     sense=pe.minimize)

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
        trajectory['x_opponent'] = [model.sub.x_obst[i]() for i in model.t]
        trajectory['v_opponent'] = [model.sub.v_obst[i]() for i in model.t]
        trajectory['u_opponent'] = [model.sub.u_obst[i]() for i in model.t]

        return trajectory


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
        # Initialize a parametric sub-model
        model.sub = pao.pyomo.SubModel(fixed=[model.x, model.v, model.u])
        model.sub.dt = pe.Param(initialize=self.terminal_time / self.n, mutable=True)
        # Define sub-model parameters
        model.sub.x0_obst = pe.Param(initialize=self._initial_state_obst['rear'][0])  # Initial position
        model.sub.v0_obst = pe.Param(initialize=self._initial_state_obst['rear'][1])  # Initial speed

        # Define obstacle variables
        model.sub.x_obst = pe.Var(model.t)  # Position [m]
        model.sub.v_obst = pe.Var(model.t, bounds=(self.v_bounds[0], self.v_bounds[1]))  # Velocity [m/s]
        model.sub.u_obst = pe.Var(model.t, bounds=(self.u_bounds[0], self.u_bounds[1]),
                                  initialize=3)  # Acceleration [m/s^2]

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

        model.sub.ode_x = pe.Constraint(model.t, rule=ode_x_obst)

        def ode_v_obst(m, t):
            if t < self.n:
                return m.v_obst[t + 1] == m.v_obst[t] + m.u_obst[t] * m.dt
            else:
                return pe.Constraint.Skip

        model.sub.ode_v = pe.Constraint(model.t, rule=ode_v_obst)

    def _define_constraints(self) -> None:
        """Create model constraints """
        model = self._model

        # Define safety constraints
        def safety_distance(m, t):
            return m.x[t] - m.sub.x_obst[t] >= self.reaction_time * m.sub.v_obst[t] + self.min_safe_distance

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
        model.sub.x_obst[0].fix(model.sub.x0_obst)
        model.sub.v_obst[0].fix(model.sub.v0_obst)

    def _define_objective(self) -> None:
        """Define the objective function."""
        model = self._model
        # ----------------- Main Model -----------------
        # Define objective function expressions
        model.speed_objective = sum(float(self._beta_v) * (model.v[t] - self.v_des) ** 2 for t in model.t)
        model.acceleration_objective = sum(model.dt * float(self._beta_u) * (model.u[t]) ** 2 for t in model.t)
        # Influence Objective
        model.opponent_objective_pos = sum(model.dt * (model.sub.x_obst[t] - self.terminal_position -
                                                       model.sub.v_obst[
                                                           t] * self.reaction_time - self.min_safe_distance) ** 2
                                           for t in model.t)
        model.opponent_objective_vel = sum(model.dt * (model.sub.v_obst[t] - self.v_des) ** 2 for t in model.t)
        # Define objective function
        model.obj = pe.Objective(rule=model.speed_objective + model.acceleration_objective +
                                      model.opponent_objective_pos + model.opponent_objective_vel, sense=pe.minimize)

        # ----------------- Obstacle Model -----------------
        # Define objective function expressions
        model.sub.speed_objective = sum(self._beta_v * (model.sub.v_obst[t] - self.v_des) ** 2 for t in model.t)
        model.sub.acceleration_objective = sum(model.dt*float(self._beta_u) * (model.sub.u_obst[t]) ** 2
                                               for t in model.t)
        # Define objective function
        model.sub.obj = pe.Objective(rule=model.sub.speed_objective + model.sub.acceleration_objective,
                                     sense=pe.minimize)

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
        trajectory['x_opponent'] = [model.sub.x_obst[i]() for i in model.t]
        trajectory['v_opponent'] = [model.sub.v_obst[i]() for i in model.t]
        trajectory['u_opponent'] = [model.sub.u_obst[i]() for i in model.t]

        return trajectory