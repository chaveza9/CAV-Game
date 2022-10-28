import jax.numpy as jnp
import pyomo as pyo
import pao
import matplotlib.pyplot as plt

from .maneuver import LongitudinalManeuver
from cav_game.dynamics.car import ControlAffineDynamics
from typing import List, Tuple, Dict


class DualGame(LongitudinalManeuver):
    def __init__(self, vehicle: ControlAffineDynamics, time: float, xf_c: float,
                 x0: jnp.ndarray, x0_obst: Dict[str, jnp.array], params: dict,
                 **kwargs):
        #  initialize the parent class
        super().__init__(vehicle, x0, x0_obst, params, **kwargs)
        self.terminal_time = time
        self.terminal_position = xf_c

    def _define_model(self) -> None:
        # ----------------- Main Model -----------------
        # Initialize a parametric model
        model = pyo.environ.ConcreteModel()

        # Define model parameters
        model.t = pyo.dae.ContinuousSet(bounds=(0, self.terminal_time))
        model.x0 = pyo.environ.Param(initialize=self._initial_state[0], mutable=True)  # Initial position
        model.v0 = pyo.environ.Param(initialize=self._initial_state[1], mutable=True)  # Initial speed
        # Define vehicle variables
        model.x = pyo.environ.Var(model.t)  # Position [m]
        model.v = pyo.environ.Var(model.t, bounds=(self.v_bounds[0], self.v_bounds[1]))  # Velocity [m/s]
        model.u = pyo.environ.Var(model.t, bounds=(self.u_bounds[0], self.u_bounds[1]),
                                  initialize=3)  # Acceleration [m/s^2]

        # Define Derivatives
        model.x_dot = pyo.dae.DerivativeVar(model.x, wrt=model.t)
        model.v_dot = pyo.dae.DerivativeVar(model.v, wrt=model.t)
        model.u_dot = pyo.dae.DerivativeVar(model.u, wrt=model.t)

        # ------------------ Obstacle Model ------------------
        # Initialize a parametric sub-model
        model.sub = pao.pyomo.SubModel(fixed=[model.x, model.v, model.u])
        # Define sub-model parameters
        model.sub.x0_obst = pyo.environ.Param(initialize=self._initial_state_obst['rear'][0],
                                              mutable=True)  # Initial position
        model.sub.x0_obst = pyo.environ.Param(initialize=self._initial_state_obst['rear'][1],
                                              mutable=True)  # Initial speed

        # Define obstacle variables
        model.sub.x_obst = pyo.environ.Var(model.t)  # Position [m]
        model.sub.v_obst = pyo.environ.Var(model.t, bounds=(self.v_bounds[0], self.v_bounds[1]))  # Velocity [m/s]
        model.sub.u_obst = pyo.environ.Var(model.t, bounds=(self.u_bounds[0], self.u_bounds[1]),
                                           initialize=3)  # Acceleration [m/s^2]

        # Define obstacle Derivatives
        model.sub.x_obst_dot = pyo.dae.DerivativeVar(model.sub.x_obst, wrt=model.t)
        model.sub.v_obst_dot = pyo.dae.DerivativeVar(model.sub.v_obst, wrt=model.t)
        model.sub.u_obst_dot = pyo.dae.DerivativeVar(model.sub.u_obst, wrt=model.t)

        self._model = model

    def _define_dae_constraints(self) -> None:
        """Define the differential algebraic constraints."""
        model = self._model

        # ----------------- Main Model -----------------
        # Define differential algebraic equations
        def ode_x(m, t):
            return m.x_dot[t] == m.v[t]

        model.ode_x = pyo.environ.Constraint(model.t, rule=ode_x)

        def ode_v(m, t):
            return m.v_dot[t] == m.u[t]

        model.ode_v = pyo.environ.Constraint(model.t, rule=ode_v)

        # ----------------- Obstacle Model -----------------
        # Define differential algebraic equations
        def ode_x_obst(m, t):
            return m.sub.x_obst_dot[t] == m.sub.v_obst[t]

        model.sub.ode_x = pyo.environ.Constraint(model.t, rule=ode_x_obst)

        def ode_v_obst(m, t):
            return m.sub.v_obst_dot[t] == m.sub.u_obst[t]

        model.sub.ode_v = pyo.environ.Constraint(model.t, rule=ode_v_obst)

    def _define_constraints(self) -> None:
        """Create model constraints """
        model = self._model

        # Define safety constraints
        def safety_distance(m, t):
            return m.x[t] - m.sub.x_obst[t] >= self.reaction_time * m.sub.v_obst[t] + self.min_safe_distance

        model.sub.safety = pyo.environ.Constraint(model.t, rule=safety_distance)

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
        model.speed_objective = self._beta_v * (model.v[1] - self.v_des) ** 2
        model.acceleration_objective = pyo.dae.Integral(model.t, wrt=model.t,
                                                        rule=lambda m, t: float(self._beta_u) * (m.u[t]) ** 2)
        # Influence Objective
        model.opponent_objective_pos = pyo.dae.Integral(model.t, wrt=model.t, rule=lambda m, t:
        (m.sub.x_obst[t] - self.terminal_position - m.sub.v_obst[t] * self.reaction_time
         - self.min_safe_distance) ** 2)
        model.opponent_objective_vel = pyo.dae.Integral(model.t, wrt=model.t, rule=lambda m, t:
        (m.sub.v_obst[t] - self.v_des) ** 2)
        # Define objective function
        model.obj = pyo.environ.Objective(
            rule=model.speed_objective + model.acceleration_objective +
                    model.opponent_objective_pos + model.opponent_objective_vel, sense=pyo.environ.minimize)

        # ----------------- Obstacle Model -----------------
        # Define objective function expressions
        model.sub.speed_objective = self._beta_v * (model.sub.v_obst[1] - self.v_des) ** 2
        model.sub.acceleration_objective = pyo.dae.Integral(model.t, wrt=model.t,
                                                        rule=lambda m, t: float(self._beta_u) * (m.sub.u_obst[t]) ** 2)
        # Define objective function
        model.sub.obj = pyo.environ.Objective(rule=model.sub.speed_objective + model.sub.acceleration_objective,
                                              sense=pyo.environ.minimize)


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
