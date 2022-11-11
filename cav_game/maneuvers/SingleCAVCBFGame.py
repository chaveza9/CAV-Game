import jax.numpy as jnp
from pyomo.core.expr import differentiate
import pao
import pyomo as pyo
import matplotlib.pyplot as plt

import pyomo.environ as pe
from .maneuver import LongitudinalManeuver
from cav_game.dynamics.car import ControlAffineDynamics
from typing import List, Tuple, Dict
from warnings import warn





class SingleCAVCBFGame(LongitudinalManeuver):
    def __init__(self, vehicle: ControlAffineDynamics, time: float, xf_c: float,
                 x0: jnp.ndarray, x0_obst: Dict[str, jnp.array], params: dict,
                 **kwargs):
        self.terminal_time = time
        self.terminal_position = xf_c
        #  initialize the parent class
        super().__init__(vehicle, x0, x0_obst, params, **kwargs)

        # Redefine normalization variables
        max_u = max([(self.u_bounds[0]) ** 2, (self.u_bounds[1]) ** 2])
        max_delta_v = max([(self.v_bounds[0] - self.v_des) ** 2, (self.v_bounds[1] - self.v_des) ** 2])
        self._beta_u = 0.1 / max_u  # Acceleration cost weight
        self._beta_v = 0.9 / max_delta_v  # Speed cost weight
        self._beta_x = 0.85 * self.terminal_time / \
                       max((self.terminal_time * self.v_bounds[0] + self._initial_state_obst['rear'][0] -
                            self.terminal_position - 15) ** 2,
                           (self.terminal_time * self.v_bounds[1] + self._initial_state_obst['rear'][0] -
                            self.terminal_position - 15) ** 2)
        self._beta_v_obst = 0.9 / max_delta_v  # Speed cost weight
        self._beta_u_obst = 0.1 / max_u  # Acceleration cost weight
        self.ego_payout = 0.7
        self.obs_payout = 0.3

        # Define optimization problem
        self._define_model()
        self._define_initial_conditions()
        self._define_objective()
        self._define_constraints()
        self._define_lagrangian()
        self._define_dae_constraints()
        self._define_solver()
        # create model instance
        self._model_instance = self._model.create_instance()

    def _define_solver(self) -> None:
        """Initialize the solver."""
        # Initialize solver
        self._opt = pao.Solver('ipopt', **self.opt_options)
        self._discretizer = pe.TransformationFactory(self.diff_method)

        if self.diff_method == 'dae.collocation':
            self._discretizer.apply_to(self._model, nfe=self.n, ncp=6, scheme='LAGRANGE-RADAU')
        elif self.diff_method == 'dae.finite_difference':
            self._discretizer.apply_to(self._model, nfe=self.n, scheme='BACKWARD')
        else:
            raise ValueError('Invalid discretization method.')

    def _define_model(self) -> None:
        # ----------------- Main Model -----------------
        # Initialize a parametric model
        model = pe.ConcreteModel()

        # Define model parameters
        model.t = pyo.dae.ContinuousSet(bounds=(0, self.terminal_time))  # Time Discretization
        model.dt = pe.Param(initialize=self.terminal_time / self.n, mutable=True)  # Time Step
        model.x0 = pe.Param(initialize=self._initial_state[0], mutable=True)  # Initial position
        model.v0 = pe.Param(initialize=self._initial_state[1], mutable=True)  # Initial speed
        # Define vehicle variables
        model.x = pe.Var(model.t)  # Position [m]
        model.v = pe.Var(model.t, bounds=(self.v_bounds[0], self.v_bounds[1]))  # Velocity [m/s]
        model.u = pe.Var(model.t, bounds=(self.u_bounds[0], self.u_bounds[1]),
                         initialize=3)  # Acceleration [m/s^2]
        # model.jerk = pe.Var(model.t, bounds=(-0.5, 0.5))  # Jerk [m/s^3]
        # Define Derivatives
        model.x_dot = pyo.dae.DerivativeVar(model.x, wrt=model.t)
        model.v_dot = pyo.dae.DerivativeVar(model.v, wrt=model.t)
        # model.u_dot = pyo.dae.DerivativeVar(model.u, wrt=model.t)

        # ------------------ Obstacle Model ------------------
        # Define sub-model parameters
        model.x0_obst = pe.Param(initialize=self._initial_state_obst['rear'][0])  # Initial position
        model.v0_obst = pe.Param(initialize=self._initial_state_obst['rear'][1])  # Initial speed

        # Define obstacle variables
        model.x_obst = pe.Var(model.t)  # Position [m]
        model.v_obst = pe.Var(model.t, bounds=(self.v_bounds[0], self.v_bounds[1]))  # Velocity [m/s]
        model.u_obst = pe.Var(model.t, bounds=(self.u_bounds[0], self.u_bounds[1]),
                              initialize=3)  # Acceleration [m/s^2]
        # model.jerk_obst = pe.Var(model.t, bounds=(-0.5, 0.5))  # Jerk [m/s^3]
        # Define Derivatives
        model.x_obst_dot = pyo.dae.DerivativeVar(model.x_obst, wrt=model.t)
        model.v_obst_dot = pyo.dae.DerivativeVar(model.v_obst, wrt=model.t)
        # model.u_obst_dot = pyo.dae.DerivativeVar(model.u_obst, wrt=model.t)

        # Define dual variables
        model.lamda_safety = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for safety constraint
        model.lambda_v_max = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for velocity constraint
        model.lambda_v_min = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for velocity constraint
        model.lambda_u_max = pe.Var(model.t,
                                    domain=pe.NonNegativeReals)  # Dual variable for acceleration constraint
        model.lambda_u_min = pe.Var(model.t,
                                    domain=pe.NonNegativeReals)  # Dual variable for acceleration constraint
        model.lambda_v_des = pe.Var(model.t, domain=pe.NonNegativeReals)  # Dual variable for velocity constraint

        model.mu_x = pe.Var(model.t, bounds=(0, None),
                            initialize=0)  # Dual variable for position equality constraint
        model.mu_v = pe.Var(model.t, bounds=(0, None),
                            initialize=0)  # Dual variable for velocity equality constraint
        # model.mu_jerk = pe.Var(model.t, bounds=(0, None),
        #                        initialize=0)  # Dual variable for jerk equality constraint
        # Lyapunov Barrier Function Constraint
        model.epsilon = pe.Param(initialize=0.5, mutable=True)  # Relaxation parameter for Lyapunov Constraint
        model.gamma = pe.Param(initialize=1, mutable=True)  # Linear Class K Function Constant
        model.lamb = pe.Param(initialize=3, mutable=True)  # Relaxation parameter for Lyapunov Constraint
        model.e = pe.Var(model.t, bounds=(None, None))  # Slack variable for QP
        model.slack = pe.Var(bounds=(0, None), initialize=0.1)  # Slack variable for QP

        self._model = model

    def _define_objective(self) -> None:
        """Define the objective function."""
        model = self._model
        # ----------------- Main Model -----------------
        # Define objective function expressions
        model.speed_objective = pyo.dae.Integral(model.t, wrt=model.t,
                                                 rule=lambda m, t: 0.5 * float(self._beta_v) * (
                                                         m.v[t] - self.v_des) ** 2)
        model.acceleration_objective = pyo.dae.Integral(model.t, wrt=model.t,
                                                        rule=lambda m, t: 0.5 * float(self._beta_u) * (m.u[t]) ** 2)
        model.terminal_speed = float(self._beta_v) * self.terminal_time * (
                model.v[self.terminal_time] - self.v_des) ** 2
        # Influence Objective
        smooth_max = lambda x, y: 0.5 * (x + y + pe.sqrt((x - y) ** 2 + 0.01))
        model.influence_objective_pos = self._beta_x * smooth_max(-(self.terminal_position - self.min_safe_distance) +
                                                                  (model.x_obst[self.terminal_time] +
                                                                   model.v_obst[
                                                                       self.terminal_time] * self.reaction_time),
                                                                  0) ** 2

        model.influence_objective_vel = pyo.dae.Integral(model.t, wrt=model.t,
                                                         rule=lambda m, t: 0.5 * float(self._beta_v) * (
                                                                 m.v_obst[t] - self.v_des) ** 2)
        # Define objective function ego vehicle expression
        model.ego_objective = model.speed_objective + model.acceleration_objective + model.terminal_speed + \
                              model.influence_objective_vel + 50*model.influence_objective_pos

        # ----------------- Obstacle Model -----------------
        # Define objective function expressions
        model.speed_objective_obst = pyo.dae.Integral(model.t, wrt=model.t,
                                                      rule=lambda m, t: 0.5 * float(self._beta_v_obst) * (
                                                              m.v_obst[t] - self.v_des) ** 2)
        model.obst_cbf_objective = pyo.dae.Integral(model.t, wrt=model.t, rule=lambda m, t: 0.5 * m.u_obst[t] ** 2 +
                                                                                            m.lamb * m.e[t] ** 2)
        # Define objective function ego vehicle expression
        model.obst_objective = model.speed_objective_obst + model.obst_cbf_objective

        model.obj = pe.Objective(rule=self.ego_payout * model.ego_objective + self.obs_payout * model.obst_objective \
                                 + 6*model.slack**2,
                                 sense=pe.minimize)

    def _define_dae_constraints(self) -> None:
        """Define the differential algebraic constraints."""
        model = self._model

        # ----------------- Main Model -----------------
        # Define differential algebraic equations
        def ode_x(m, k):
            return m.x_dot[k] == m.v[k]

        model.ode_x = pe.Constraint(model.t, rule=ode_x)

        def ode_v(m, k):
            return m.v_dot[k] == m.u[k]

        model.ode_v = pe.Constraint(model.t, rule=ode_v)

        # def ode_u(m, k):
        #     return m.u_dot[k] == m.jerk[k]
        #
        # model.ode_jerk = pe.Constraint(model.t, rule=ode_u)

        # ----------------- Obstacle Model -----------------
        # Define differential algebraic equations
        def ode_x_obst(m, k):
            return m.x_obst_dot[k] == m.v_obst[k]

        model.ode_x_obst = pe.Constraint(model.t, rule=ode_x_obst)

        def ode_v_obst(m, t):
            return m.v_obst_dot[t] == m.u_obst[t]

        model.ode_v_obst = pe.Constraint(model.t, rule=ode_v_obst)

        # def ode_u_obst(m, t):
        #     return m.u_obst_dot[t] == m.jerk_obst[t]
        #
        # model.ode_u_obst = pe.Constraint(model.t, rule=ode_u_obst)

    def _define_constraints(self) -> None:
        """Create model constraints """
        model = self._model

        # Define safety constraints
        def safety_distance(m, t):
            return m.x[t] - m.x_obst[t] >= self.reaction_time * m.v_obst[t] + self.min_safe_distance

        model.safety = pe.Constraint(model.t, rule=safety_distance)

        # Define terminal constraints
        model.xf = pe.Constraint(expr=model.x[self.terminal_time] >= self.terminal_position + 15.0)

        # model.xf_obst = pe.Constraint(expr=model.x_obst[self.terminal_time] <= self.terminal_position - 15.0)

        # Define CBF constraint
        def cbf_safety(m, t):
            return m.v[t] - m.v_obst[t] - float(self.reaction_time) * m.u_obst[t] + \
                   m.gamma * (m.x[t] - self.reaction_time * m.v_obst[t] - self.min_safe_distance) >= 0

        def cbf_v_max(m, t):
            return -m.u_obst[t] + m.gamma * (self.v_bounds[1] - m.v_obst[t]) >= 0

        def cbf_v_min(m, t):
            return m.u_obst[t] + m.gamma * (m.v_obst[t] - self.v_bounds[0]) >= 0

        model.cbf_safety = pe.Constraint(model.t, rule=cbf_safety)
        model.cbf_v_max = pe.Constraint(model.t, rule=cbf_v_max)
        model.cbf_v_min = pe.Constraint(model.t, rule=cbf_v_min)

        # Define Lyapunov constraint
        def clf_v_des(m, t):
            return 2 * (m.v_obst[t] - self.v_des) * m.u_obst[t] + m.epsilon * (m.v_obst[t] - self.v_des) ** 2 <= m.e[t]

        model.clf_constraint = pe.Constraint(model.t, rule=clf_v_des)

    def _define_initial_conditions(self) -> None:
        """Define the initial conditions."""
        model = self._model
        # Main Model
        # Define initial conditions
        model.x[0].fix(model.x0)
        model.v[0].fix(model.v0)
        model.u[0].fix(0)
        # Sub Model
        # Define initial conditions
        model.x_obst[0].fix(model.x0_obst)
        model.v_obst[0].fix(model.v0_obst)
        model.u_obst[0].fix(0)

    def _define_lagrangian(self) -> None:
        model = self._model
        # Constraints for the Lagrangian
        # Inequality constraints
        cbf_safety = lambda m, t: -(m.v[t] - m.v_obst[t] - self.reaction_time * m.u_obst[t] +
                                    m.gamma * (m.x[t] - self.reaction_time * m.v_obst[t] - self.min_safe_distance))
        cbf_v_max = lambda m, t: m.u_obst[t] - m.gamma * (self.v_bounds[1] - m.v_obst[t])
        cbf_v_min = lambda m, t: -m.u_obst[t] - m.gamma * (m.v_obst[t] - self.v_bounds[0])
        clf_v_des = lambda m, t: 2 * (m.v_obst[t] - self.v_des) * m.u_obst[t] + \
                                 m.epsilon * (m.v_obst[t] - self.v_des) ** 2 - m.e[t]

        # KKT Conditions
        # Objective terms
        acc_obj_dot = lambda m, t: m.u_obst[t] * m.dt
        # Equality terms
        pos_eq_dot = lambda m, t: m.mu_x[t] * 0.5 * m.dt ** 2
        vel_eq_dot = lambda m, t: m.mu_v[t] * m.dt
        # jerk_eq_dot = lambda m, t: m.mu_jerk[t] * 1 / m.dt
        # Inequality terms
        acc_lim_dot = lambda m, t: -m.lambda_u_min[t] * m.dt + m.lambda_u_max[t] * m.dt
        cbf_safety = lambda m, t: (m.dt + self.reaction_time + m.gamma / 2 * m.dt ** 2 +
                                   self.reaction_time * m.gamma * m.dt) * m.lamda_safety[t]
        cbf_v_max = lambda m, t: m.dt * m.lambda_v_max[t] * (1 + m.gamma * m.dt)
        cbf_v_min = lambda m, t: m.dt * m.lambda_v_min[t] * (1 + m.gamma * m.dt)
        clf_v_des = lambda m, t: m.dt * m.lambda_v_des[t] * (2 * (m.v_obst[t] + m.u_obst[t] * m.dt - self.v_des)
                                                             - 2 * m.u_obst[t] * m.dt + +2 * (
                                                                     m.v_obst[t] + m.u_obst[t] * m.dt
                                                                     - self.v_des) * m.gamma * m.dt)

        # Define the Lagrangian gradient constraint (Stationary conditions)
        lagrangian_dot = lambda m, t: acc_obj_dot(m, t) + \
                                      pos_eq_dot(m, t) + vel_eq_dot(m, t) + \
                                      acc_lim_dot(m, t) + cbf_safety(m, t) + cbf_v_max(m, t) + cbf_v_min(m, t) + \
                                      clf_v_des(m, t) == 0 if t > 0 else pe.Constraint.Skip

        model.kkt_stationary = pe.Constraint(model.t, rule=lagrangian_dot)
        # Add complementary slackness conditions
        # Safety Constraint
        kkt_safety = lambda m, t: m.lamda_safety[t] * cbf_safety(m, t) >= -m.slack
        model.kkt_safety = pe.Constraint(model.t, rule=kkt_safety)
        # Velocity Bounds
        kkt_v_min = lambda m, k: m.lambda_v_min[k] * cbf_v_min(m, k) >= -m.slack
        model.kkt_v_min = pe.Constraint(model.t, rule=kkt_v_min)

        kkt_v_max = lambda m, k: m.lambda_v_max[k] * cbf_v_max(m, k) >= -m.slack
        model.kkt_v_max = pe.Constraint(model.t, rule=kkt_v_max)

        # Actuation Bounds
        kkt_u_min = lambda m, k: m.lambda_u_min[k] * (-m.u_obst[k] + self.u_bounds[0]) >= -m.slack
        model.kkt_u_min = pe.Constraint(model.t, rule=kkt_u_min)

        kkt_u_max = lambda m, k: m.lambda_u_max[k] * (m.u_obst[k] - self.u_bounds[1]) >= -m.slack
        model.kkt_u_max = pe.Constraint(model.t, rule=kkt_u_max)

        # Velocity Desired
        kkt_v_des = lambda m, k: m.lambda_v_des[k] * clf_v_des(m, k) >= -m.slack
        model.kkt_v_des = pe.Constraint(model.t, rule=kkt_v_des)

    def _extract_results(self) -> Dict[str, jnp.ndarray]:
        """Extract the solution from the solver."""
        model = self._model_instance

        trajectory = dict()
        # Extract solution
        trajectory['t'] = [t for t in model.t]
        # Ego Vehicle
        trajectory['x_ego'] = [model.x[i]() for i in model.t]
        trajectory['v_ego'] = [model.v[i]() for i in model.t]
        trajectory['u_ego'] = [model.u[i]() for i in model.t]
        # Opponent Vehicle
        trajectory['x_opponent'] = [model.x_obst[i]() for i in model.t]
        trajectory['v_opponent'] = [model.v_obst[i]() for i in model.t]
        trajectory['u_opponent'] = [model.u_obst[i]() for i in model.t]

        return trajectory

    def _generate_trajectory_plots(self, trajectory, save_path: str = "", obstacle: bool = False, show: bool = False,
                                   **kwargs) -> Tuple[plt.figure, List[plt.axes]]:
        """Function to generate the plots of the trajectory and the control inputs"""
        # Extract the results
        tsim = trajectory['t']
        xsim = trajectory['x_ego']
        vsim = trajectory['v_ego']
        usim = trajectory['u_ego']

        # Check if reference vehicle is present
        if kwargs.get('ref_trajectory') is not None and kwargs.get('ref_name') is not None:
            ref_trajectory = kwargs.get('ref_trajectory')
            ref = True
            x_ref = ref_trajectory['x']
            v_ref = ref_trajectory['v']
            u_ref = ref_trajectory['u']
            t_ref = ref_trajectory['t']
        else:
            ref = False
        if obstacle:
            xsim_opponent = trajectory['x_opponent']
            vsim_opponent = trajectory['v_opponent']
            usim_opponent = trajectory['u_opponent']
            safety_distance = jnp.array(xsim_opponent) + \
                              self.reaction_time * jnp.array(vsim_opponent) + self.min_safe_distance
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
            ax1.legend(prop={'size': 6})
            ax2.plot(tsim, vsim_opponent, label='Opponent Vehicle', color='red')
            ax3.plot(tsim, usim_opponent, label='Opponent Vehicle', color='red')
            ax3.set_ylim([min(usim + usim_opponent) - 0.5, max(usim + usim_opponent) + 0.5])

        if ref:
            ax1.plot(t_ref, x_ref, label=kwargs.get('ref_name'), color='blue', linestyle='dashed')
            ax1.legend(prop={'size': 6})
            ax2.plot(t_ref, v_ref, label=kwargs.get('ref_name'), color='blue', linestyle='dashed')
            ax3.plot(t_ref, u_ref, label=kwargs.get('ref_name'), color='blue', linestyle='dashed')

        if save_path:
            fig.savefig(save_path)

        if show:
            fig.tight_layout()
            fig.show()

        return fig, [ax1, ax2, ax3]


class DualCAVCBFGame(LongitudinalManeuver):

