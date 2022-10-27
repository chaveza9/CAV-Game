import jax.numpy as jnp
import pyomo as pyo
import matplotlib.pyplot as plt

from .maneuver import LongitudinalManeuver
from cav_game.dynamics.car import ControlAffineDynamics
from typing import List, Tuple, Dict

class SelfishManeuver(LongitudinalManeuver):  
    def __init__(self,vehicle: ControlAffineDynamics, x0: jnp.ndarray, x0_obst: Dict[str, jnp.array], params: dict, **kwargs):
        #  initialize the parent class
        super().__init__(vehicle, x0, x0_obst, params, **kwargs)   
    
    def _define_model(self) -> None:
        # Initialize a parametric model
        model = pyo.environ.ConcreteModel()
        # Define model parameters
        model.t = pyo.dae.ContinuousSet(bounds=(0, 1))
        model.x0 = pyo.environ.Param(initialize=self._initial_state[0], mutable = True) # Initial position
        model.v0 = pyo.environ.Param(initialize=self._initial_state[1], mutable = True) # Initial speed
        # Front obstacle parameters
        model.x0_obst = pyo.environ.Param(initialize=self._initial_state_obst['front'][0], mutable = True) # Initial position
        model.v0_obst = pyo.environ.Param(initialize=self._initial_state_obst['front'][1], mutable = True) # Initial speed
        # Define vehicle variables
        model.tf = pyo.environ.Var(bounds=(0, self.t_max)) # Terminal time
        model.x = pyo.environ.Var(model.t) # Position
        model.v = pyo.environ.Var(model.t, bounds=(self.v_bounds[0],self.v_bounds[1])) # Velocity
        model.u = pyo.environ.Var(model.t, bounds=(self.u_bounds[0],self.u_bounds[1])) # Acceleration
        # Define obstacle variables
        model.x_obst = pyo.environ.Var(model.t) # Position
        
        # Define Derivatives
        model.x_dot = pyo.dae.DerivativeVar(model.x, wrt=model.t)
        model.v_dot = pyo.dae.DerivativeVar(model.v, wrt=model.t)
        
        self._model = model
        
        
    def _define_dae_constraints(self) -> None:
        """Define the differential algebraic constraints."""
        model = self._model
        # Define differential algebraic equations
        def ode_x(m, t):
            return m.x_dot[t] == m.v[t]*m.tf
        model.ode_x = pyo.environ.Constraint(model.t, rule=ode_x)
        def ode_v(m, t):
            return m.v_dot[t] == m.u[t]*m.tf
        model.ode_v = pyo.environ.Constraint(model.t, rule=ode_v)
    
    def _define_constraints(self) -> None:
        """Create model constraints """
        model = self._model
        # Define Obstalce Model
        def obstacle_model(m, t):
            return m.x_obst[t] == (m.x0_obst + m.v0_obst*t)*m.tf
        model.obs_ode = pyo.environ.Constraint(model.t, rule = obstacle_model)
        # Define safety constraints
        def safety_distance(m, t):
            return m.x_obst[t] - m.x[t] >= self.reaction_time*m.v[t] + self.min_safe_distance
        model.safety = pyo.environ.Constraint(model.t, rule = safety_distance)
        
    def _define_initial_conditions(self) -> None:
        """Define the initial conditions."""
        model = self._model
        # Define initial conditions
        model.x[0].fix(model.x0)
        model.v[0].fix(model.v0)
        
    def _define_objective(self) -> None:
        """Define the objective function."""
        model = self._model
        # Define objective function
        model.time_objective =  model.tf*self.alpha_time*self.n
        model.speed_objective =  self._beta_v*model.tf*(model.v[1]-self.v_des)**2
        model.acceleration_objective = pyo.dae.Integral(model.t, wrt=model.t,
                                                        rule=lambda m, t: 0.5*float(self._beta_u)*m.tf*(m.u[t])**2)
        
        model.obj = pyo.environ.Objective(rule=model.time_objective+model.speed_objective+model.acceleration_objective,
                                  sense=pyo.environ.minimize)
        
    def _define_solver(self) -> None:
        """Initialize the solver."""
        self._discretizer.apply_to(self._model, nfe=self.n, ncp=3, scheme='LAGRANGE-RADAU')
        # create model instance
        self._model_instance = self._model.create_instance()
        
    def _extract_results(self) -> Dict[str, jnp.ndarray]:
        """Extract the solution from the solver."""
        model = self._model_instance
        
        trajectory = dict()
        # Extract solution
        trajectory['t'] = [t*model.tf for t in model.t]
        trajectory['x'] = [model.x[i]() for i in model.t]
        trajectory['v'] = [model.v[i]() for i in model.t]
        trajectory['u'] = [model.u[i]() for i in model.t]
        
        return trajectory
        
        
        
        
        
        
        
        
        

        
            
    