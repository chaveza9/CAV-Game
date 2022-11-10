import jax.numpy as jnp
import pyomo as pyo
import pyomo.environ as pe
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

import pao

from .maneuver import LongitudinalManeuver
from cav_game.dynamics.car import ControlAffineDynamics
from typing import List, Tuple, Dict


class FollowingManeuver(LongitudinalManeuver):
    def __init__(self, vehicle: ControlAffineDynamics, tf: float, xf: float, x0: jnp.ndarray,
                 x0_obst: Dict[str, jnp.array], params: dict,
                 **kwargs):
        self.terminal_time = tf
        self.terminal_position = xf
        #  initialize the parent class
        super().__init__(vehicle, x0, x0_obst, params, **kwargs)
        max_u = max([(self.u_bounds[0]) ** 2, (self.u_bounds[1]) ** 2])
        max_delta_v = max([(self.v_bounds[0] - self.v_des) ** 2, (self.v_bounds[1] - self.v_des) ** 2])
        self._beta_u = 0.3 / max_u  # Acceleration cost weight
        self._beta_v = 0.7 / max_delta_v  # Speed cost weight
        self._define_model()
        self._define_initial_conditions()
        self._define_objective()
        self._define_constraints()
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
        # Initialize a parametric model
        model = pe.ConcreteModel()
        # Define model parameters
        model.tf = pe.Param(initialize=self.terminal_time)  # Terminal time [s]
        model.xf_ref = pe.Param(initialize=self.terminal_position)  # Terminal reference position [m]
        model.t = pyo.dae.ContinuousSet(bounds=(0, model.tf))
        model.x0 = pe.Param(initialize=self._initial_state[0], mutable=True)  # Initial position
        model.v0 = pe.Param(initialize=self._initial_state[1], mutable=True)  # Initial speed

        # Define vehicle variables
        model.x = pe.Var(model.t)  # Position [m]
        model.v = pe.Var(model.t, bounds=(self.v_bounds[0], self.v_bounds[1]))  # Velocity [m/s]
        model.u = pe.Var(model.t, bounds=(self.u_bounds[0], self.u_bounds[1]), initialize=3)  # Acceleration [m/s^2]
        model.jerk = pe.Var(model.t, bounds=(-0.8, 0.8))  # Jerk [m/s^3]

        # Define Derivatives
        model.x_dot = pyo.dae.DerivativeVar(model.x, wrt=model.t)
        model.v_dot = pyo.dae.DerivativeVar(model.v, wrt=model.t)
        model.u_dot = pyo.dae.DerivativeVar(model.u, wrt=model.t)

        self._model = model

    def relax_terminal_time(self, time, xf_ref, plot: bool = True, save_path: str = "",
                            obstacle: bool = False, show: bool = True) \
            -> Tuple[bool, Dict[str, jnp.ndarray]]:
        """Relax the terminal time constraint to an specified value."""
        # Check if the solver has been initialized
        if self._model_instance is None or self._model_instance.tf is None:
            raise ValueError('Solver has not been initialized.')

        self._model_instance.tf.reconstruct(time)
        self._model_instance.xf_ref.reconstruct(xf_ref)

        # Get the solution
        return self.compute_longitudinal_trajectory(plot, save_path, obstacle, show)

    def _define_dae_constraints(self) -> None:
        """Define the differential algebraic constraints."""
        model = self._model

        # Define differential algebraic equations
        def ode_x(m, t):
            return m.x_dot[t] == m.v[t]  # Position

        model.ode_x = pe.Constraint(model.t, rule=ode_x)

        def ode_v(m, t):
            return m.v_dot[t] == m.u[t]  # Velocity

        model.ode_v = pe.Constraint(model.t, rule=ode_v)

        def ode_u(m, t):
            return m.u_dot[t] == m.jerk[t]  # Acceleration

        model.ode_u = pe.Constraint(model.t, rule=ode_u)

    def _define_constraints(self) -> None:
        """Create model constraints """
        model = self._model

        model.safety = pe.Constraint(rule=model.x[model.tf] + self.reaction_time * model.v[model.tf] +
                                          self.min_safe_distance <= model.xf_ref)

    def _define_initial_conditions(self) -> None:
        """Define the initial conditions."""
        model = self._model
        # Define initial conditions
        model.x[0].fix(model.x0)
        model.v[0].fix(model.v0)
        model.u[0].fix(0)

    def _define_objective(self) -> None:
        """Define the objective function."""
        model = self._model
        # Define objective function
        model.speed_objective = pyo.dae.Integral(model.t, wrt=model.t,
                                                 rule=lambda m, t: 0.5*float(self._beta_v) * (m.v[t] - self.v_des) ** 2)
        model.terminal_speed = 6*float(self._beta_v)*model.tf*(model.v[model.tf] - self.v_des)**2
        model.acceleration_objective = pyo.dae.Integral(model.t, wrt=model.t,
                                                        rule=lambda m, t: 0.5 * float(self._beta_u) * (m.u[t]) ** 2)
        model.obj = pe.Objective(
            rule=model.speed_objective + model.acceleration_objective + model.terminal_speed,
            sense=pe.minimize)

    def _extract_results(self) -> Dict[str, jnp.ndarray]:
        """Extract the solution from the solver."""
        model = self._model_instance

        trajectory = dict()
        # Extract solution
        trajectory['t'] = [t for t in model.t]
        trajectory['x'] = [model.x[i]() for i in model.t]
        trajectory['v'] = [model.v[i]() for i in model.t]
        trajectory['u'] = [model.u[i]() for i in model.t]
        trajectory['safety_distance'] = [model.x[i]() + model.v[i]() * self.reaction_time + self.reaction_time
                                         for i in model.t]

        return trajectory

    def _generate_trajectory_plots(self, trajectory, save_path: str = "", obstacle: bool = False, show: bool = False,
                                   **kwargs) -> Tuple[plt.figure, List[plt.axes]]:
        """Function to generate the plots of the trajectory and the control inputs"""
        model = self._model_instance

        # Extract the results
        tsim = trajectory['t']
        xsim = trajectory['x']
        vsim = trajectory['v']
        usim = trajectory['u']

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

        # Plot the trajectory
        plt.style.use(['science', 'ieee'])
        cm = 1 / 2.54  # centimeters in inches
        fig = plt.figure(figsize=(8 * cm, 8 * cm))
        (ax1, ax2, ax3) = fig.subplots(3, sharex=True)

        fig.suptitle("Longitudinal Trajectory " + self.cav_type)
        # Position vs Time
        ax1.plot(tsim, xsim, label="Ego Vehicle", color='blue')
        ax1.grid(True, which="both")
        ax1.set_ylabel(r'Position $[m]$')
        # Velocity vs Time
        ax2.plot(tsim, vsim, label="Ego Vehicle", color='blue')
        ax2.grid(True, which='both')
        ax2.set_ylabel(r'Velocity $[m/s]$')
        # Acceleration vs Time
        ax3.plot(tsim, usim, label="Ego Vehicle", color='blue')
        ax3.set_ylabel(r'Acceleration $ [ m/s^2 ]$')
        ax3.set_xlabel(r'Time s')
        ax3.set_ylim([min(usim) - 0.5, max(usim) + 0.5])
        ax3.grid(True, which='both')

        if ref:
            ax1.plot(t_ref, x_ref, label=kwargs.get('ref_name'), color='orange', linestyle='solid')
            ax2.plot(t_ref, v_ref, label=kwargs.get('ref_name'), color='orange', linestyle='solid')
            ax3.plot(t_ref, u_ref, label=kwargs.get('ref_name'), color='orange', linestyle='solid')
            ax3.legend(prop={'size': 6})

        fig.align_ylabels()
        if save_path:
            fig.savefig(save_path)

        if show:
            fig.tight_layout()
            fig.show()

        return fig, [ax1, ax2, ax3]
