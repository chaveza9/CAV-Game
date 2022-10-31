import jax.numpy as jnp
import pyomo as pyo
import pyomo.environ as pe
import matplotlib.pyplot as plt
import pao

from .maneuver import LongitudinalManeuver
from cav_game.dynamics.car import ControlAffineDynamics
from typing import List, Tuple, Dict

class SelfishManeuver(LongitudinalManeuver):  
    def __init__(self,vehicle: ControlAffineDynamics, x0: jnp.ndarray, x0_obst: Dict[str, jnp.array], params: dict, **kwargs):
        #  initialize the parent class
        super().__init__(vehicle, x0, x0_obst, params, **kwargs)

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
        model.t = pyo.dae.ContinuousSet(bounds=(0, 1))
        model.x0 = pe.Param(initialize=self._initial_state[0], mutable = True) # Initial position
        model.v0 = pe.Param(initialize=self._initial_state[1], mutable = True) # Initial speed
        # Front obstacle parameters
        model.x0_obst = pe.Param(initialize=self._initial_state_obst['front'][0], mutable = True) # Initial position
        model.v0_obst = pe.Param(initialize=self._initial_state_obst['front'][1], mutable = True) # Initial speed
        # Define vehicle variables
        model.tf = pe.Var(bounds=(0, self.t_max), domain=pe.NonNegativeReals, initialize=3) # Terminal time [s]
        model.x = pe.Var(model.t) # Position [m]
        model.v = pe.Var(model.t, bounds=(self.v_bounds[0],self.v_bounds[1])) # Velocity [m/s]
        model.u = pe.Var(model.t, bounds=(self.u_bounds[0],self.u_bounds[1]), initialize=3) # Acceleration [m/s^2]
        model.jerk = pe.Var(model.t, bounds=(-0.8,0.8))  # Jerk [m/s^3]
        # Define obstacle variables
        model.x_obst = pe.Var(model.t) # Position
        
        # Define Derivatives
        model.x_dot = pyo.dae.DerivativeVar(model.x, wrt=model.t)
        model.v_dot = pyo.dae.DerivativeVar(model.v, wrt=model.t)
        model.u_dot = pyo.dae.DerivativeVar(model.u, wrt=model.t)
        model.x_obst_dot = pyo.dae.DerivativeVar(model.x_obst, wrt=model.t)  # Position
        
        self._model = model


    def relax_terminal_time(self, time, plot: bool = True, save_path: str = "",
                                               obstacle: bool = False,  show: bool = True) \
            -> Tuple[bool ,Dict[str, jnp.ndarray]]:
        """Relax the terminal time constraint to an specified value."""
        # Check if the solver has been initialized
        if self._model_instance is None or self._model_instance.tf is None:
            raise ValueError('Solver has not been initialized.')

        self._model_instance.tf.fix(time)

        # Get the solution
        return self.compute_longitudinal_trajectory(plot, save_path, obstacle, show)
        
        
    def _define_dae_constraints(self) -> None:
        """Define the differential algebraic constraints."""
        model = self._model
        # Define differential algebraic equations
        def ode_x(m, t):
            return m.x_dot[t] == m.v[t]*m.tf
        model.ode_x = pe.Constraint(model.t, rule=ode_x)
        def ode_v(m, t):
            return m.v_dot[t] == m.u[t]*m.tf
        model.ode_v = pe.Constraint(model.t, rule=ode_v)

        def ode_u(m, t):
            return m.u_dot[t] == m.jerk[t]*m.tf
        model.ode_u = pe.Constraint(model.t, rule=ode_u)

        # Define Obstalce Model
        def obstacle_model(m, t):
            return m.x_obst_dot[t] == m.v0_obst* m.tf
        model.obs_ode = pe.Constraint(model.t, rule=obstacle_model)
    
    def _define_constraints(self) -> None:
        """Create model constraints """
        model = self._model
        # Define safety constraints
        def safety_distance(m, t):
            return m.x_obst[t] - m.x[t] >= self.reaction_time*m.v[t] + self.min_safe_distance
        model.safety = pe.Constraint(model.t, rule = safety_distance)
        
    def _define_initial_conditions(self) -> None:
        """Define the initial conditions."""
        model = self._model
        # Define initial conditions
        model.x[0].fix(model.x0)
        model.v[0].fix(model.v0)
        model.u[0].fix(0)
        model.x_obst[0].fix(model.x0_obst)
        # model.u[0].fix(0) # Initial acceleration must be zero to avoid numerical issues and simulate beggining of the maneuver
        
    def _define_objective(self) -> None:
        """Define the objective function."""
        model = self._model
        # Define objective function
        model.jerk_objective = pyo.dae.Integral(model.t, wrt=model.t,
                                                        rule=lambda m, t: 0.1*(m.jerk[t])**2)
        model.time_objective =  model.tf*self.alpha_time
        model.speed_objective =  self._beta_v*(model.v[1]-self.v_des)**2
        # model.speed_objective = pyo.dae.Integral(model.t, wrt=model.t,
                         # rule=lambda m, t:  float(self._beta_v) * (m.v[t]-self.v_des) ** 2)
        model.acceleration_objective = pyo.dae.Integral(model.t, wrt=model.t,
                                                        rule=lambda m, t: float(self._beta_u)*(m.u[t])**2)
        
        model.obj = pe.Objective(
            rule=model.time_objective+model.speed_objective+(model.acceleration_objective+model.jerk_objective),
            sense=pe.minimize)

        
    def _extract_results(self) -> Dict[str, jnp.ndarray]:
        """Extract the solution from the solver."""
        model = self._model_instance
        
        trajectory = dict()
        # Extract solution
        trajectory['t'] = [t*model.tf.value for t in model.t]
        trajectory['x'] = [model.x[i]() for i in model.t]
        trajectory['v'] = [model.v[i]() for i in model.t]
        trajectory['u'] = [model.u[i]() for i in model.t]
        trajectory['x_obst'] = [model.x_obst[i]() for i in model.t]
        trajectory['safety_distance'] = [model.x_obst[i]() - model.v[i]() * self.reaction_time - self.reaction_time
                                         for i in model.t]
        
        return trajectory


class SelfishManeuverRK4(LongitudinalManeuver):
    def __init__(self, vehicle: ControlAffineDynamics, x0: jnp.ndarray, x0_obst: Dict[str, jnp.array], params: dict,
                 **kwargs):
        #  initialize the parent class
        super().__init__(vehicle, x0, x0_obst, params, **kwargs)

    def _define_solver(self) -> None:
        """Initialize the solver."""
        # Initialize solver
        self._opt = pao.Solver('ipopt', **self.opt_options)
        self._discretizer = None
    def _define_model(self) -> None:
        # Initialize a parametric model
        model = pe.ConcreteModel()
        # Define model parameters
        model.t = pe.RangeSet(0, self.n) # Time Discretization
        model.x0 = pe.Param(initialize=self._initial_state[0], mutable=True)  # Initial position
        model.v0 = pe.Param(initialize=self._initial_state[1], mutable=True)  # Initial speed
        # Front obstacle parameters
        model.x0_obst = pe.Param(initialize=self._initial_state_obst['front'][0], mutable=True)  # Initial position
        model.v0_obst = pe.Param(initialize=self._initial_state_obst['front'][1], mutable=True)  # Initial speed
        # Define vehicle variables
        model.tf = pe.Var(bounds=(0, self.t_max), domain=pe.NonNegativeReals, initialize=3)  # Terminal time [s]
        model.x = pe.Var(model.t)  # Position [m]
        model.v = pe.Var(model.t, bounds=(self.v_bounds[0], self.v_bounds[1]))  # Velocity [m/s]
        model.u = pe.Var(model.t, bounds=(self.u_bounds[0], self.u_bounds[1]))  # Acceleration [m/s^2]
        model.jerk = pe.Var(model.t, bounds=(-0.8, 0.8))  # Jerk [m/s^3]
        # Define obstacle variables
        model.x_obst = pe.Var(model.t)  # Position [m]
        # Define time interval
        model.dt = model.tf / self.n # Time step [s]

        self._model = model

    def relax_terminal_time(self, time, plot: bool = True, save_path: str = "",
                            obstacle: bool = False, show: bool = True) \
            -> Tuple[bool, Dict[str, jnp.ndarray]]:
        """Relax the terminal time constraint to an specified value."""
        # Check if the solver has been initialized
        if self._model_instance is None or self._model_instance.tf is None:
            raise ValueError('Solver has not been initialized.')

        self._model_instance.tf.fix(float(time))

        # Get the solution
        return self.compute_longitudinal_trajectory(plot, save_path, obstacle, show)

    def _define_dae_constraints(self) -> None:
        """Define the differential algebraic constraints."""
        model = self._model

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

        def ode_u(m, k):
            if k < self.n:
                return m.u[k + 1] == m.u[k] + m.jerk[k] * m.dt
            else:
                return pe.Constraint.Skip
        model.ode_u = pe.Constraint(model.t, rule=ode_u)

        # Define Obstalce Model
        def obstacle_model(m, k):
            if k < self.n:
                return m.x_obst[k + 1] == m.x_obst[k] + m.v0_obst * m.dt
            else:
                return pe.Constraint.Skip
        model.obs_ode = pe.Constraint(model.t, rule=obstacle_model)

    def _define_constraints(self) -> None:
        """Create model constraints """
        model = self._model
        # Define safety constraints
        def safety_distance(m, t):
            return m.x_obst[t] - m.x[t] >= self.reaction_time * m.v[t] + self.min_safe_distance
        model.safety = pe.Constraint(model.t, rule=safety_distance)

    def _define_initial_conditions(self) -> None:
        """Define the initial conditions."""
        model = self._model
        # Define initial conditions
        model.x[0].fix(model.x0)
        model.v[0].fix(model.v0)
        model.u[0].fix(0)
        model.x_obst[0].fix(model.x0_obst)

    def _define_objective(self) -> None:
        """Define the objective function."""
        model = self._model
        # Define objective function

        model.jerk_objective = sum(model.dt * 0.1* (model.jerk[k]) ** 2 for k in model.t)
        model.time_objective = model.tf * self.alpha_time
        model.speed_objective = self._beta_v * (model.v[self.n] - self.v_des) ** 2
        model.acceleration_objective = sum(model.dt * float(self._beta_u)* (model.u[k]) ** 2 for k in model.t)
        model.obj = pe.Objective(
            rule=model.time_objective + model.speed_objective + (model.acceleration_objective + model.jerk_objective),
            sense=pe.minimize)

    def _extract_results(self) -> Dict[str, jnp.ndarray]:
        """Extract the solution from the solver."""
        model = self._model_instance

        trajectory = dict()
        # Extract solution
        trajectory['t'] = jnp.linspace(0, model.tf.value, self.n + 1)
        trajectory['x'] = [model.x[i]() for i in model.t]
        trajectory['v'] = [model.v[i]() for i in model.t]
        trajectory['u'] = [model.u[i]() for i in model.t]
        trajectory['x_obst'] = [model.x_obst[i]() for i in model.t]
        trajectory['safety_distance'] = [model.x_obst[i]() - model.v[i]()*self.reaction_time-self.reaction_time
                                         for i in model.t]

        return trajectory
        
        
        
        
        
        

        
            
    