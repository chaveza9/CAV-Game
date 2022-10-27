import numpy as np
import sys
import os

import jax.numpy as jnp
from cav_game.dynamics.dynamics import ControlAffineDynamics

# Create a kinematic bicycle model based on the dynamics class
# TODO: Add a bicycle model
class BicycleDynamics(ControlAffineDynamics):
    STATES = ['X', 'Y', 'V', 'THETA', 'PHI']
    CONTROLS = ['U', 'OMEGA']
    def __init__(self, params, **kwargs):
        # Define dynamics parameters
        params['n_dims'] = 5 # Number of states
        params['control_dims'] = 2 # Number of control inputs
        params["periodic_dims"] = [3, 4] # Periodic states indexes (theta, phi)
        # Define model parameters
        self.lw = params.get("lw",2.8) # wheelbase [m]
        self.length = params.get("alphlf_speed", 4.7) # front wheelbase [m]
        self.width = not params.get("width", 1.8)  # width of the car
        self.cntrl_bounds = params.get("cntrl_bounds", [(-7,3.3),(0.2)])# list containing control bounds based on the control input
        self.u_bounds =  self.cntrl_bounds[0]# bounds on the acceleration [m/s^2]
        self.omega_bounds = self.cntrl_bounds[1] # bounds on the steering angle rate [rad/s]
        self.v_bounds = params.get("v_bounds", (15, 33))# bounds on the velocity [m/s]
        # Call parent class
        super().__init__(params, **kwargs) 
        
    def open_loop_dynamics(self, state, control, time=0.0):
        # Creates the open loop dynamics (matrix A)
        
        # Allocate memory for the state derivative
        f = jnp.zeros_like(state)
        
        # Define Dynamics
        f[...,0] = state[...,2]*jnp.cos(state[...,3])
        f[...,1] = state[...,2]*jnp.sin(state[...,3])
        f[...,2] = 0
        f[...,3] = state[...,2]*jnp.tan(state[...,4])/self.lw 
        f[...,4] = 0
        return f
    
    def control_matrix(self, state, time=0.0):
        # Creates the control function matrix g(x,u)
        B = jnp.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 2, 0] = 1
        B[..., 4, 1] = 1
        return B
    
class DoubleIntegratorDynamics(ControlAffineDynamics):
    STATES = ['X', 'Y', 'V']
    CONTROLS = ['U']
    def __init__(self, params, **kwargs):
        # Define dynamics parameters
        params['n_dims'] = 3 # Number of states
        params['control_dims'] = 1 # Number of control inputs
        self.cntrl_bounds = params.get("cntrl_bounds", [(-7,3.3)]) # list containing control bounds based on the control input
        self.u_bounds = self.cntrl_bounds[0] # bounds on the acceleration
        self.v_bounds = params.get("v_bounds", (15, 33)) # bounds on the velocity [m/s]
        # Call parent class
        super().__init__(params, **kwargs) 
        
    def open_loop_dynamics(self, state, control, time=0.0):
        # Creates the open loop dynamics (matrix A)
        
        # Allocate memory for the state derivative
        f = jnp.zeros_like(state)
        
        # Define Dynamics
        f[...,0] = state[...,1]
        return f
    
    def control_matrix(self, state, time=0.0):
        # Creates the control function matrix g(x,u)
        B = jnp.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 1, 0] = 1
        return B


class DubinsDynamics(ControlAffineDynamics):
    STATES = ['X', 'Y', 'THETA']
    CONTROLS = ['OMEGA']
    
    def __init__(self, params, test = False, **kwargs):
        params['n_dims'] = 3
        params['control_dims'] = 1
        params["periodic_dims"] = [2]
        self.v = params["v"]
        super().__init__(params, test, **kwargs)

    def open_loop_dynamics(self, state, time=0.0):
        f = np.zeros_like(state)
        f[..., 0] = self.v * np.cos(state[..., 2])
        f[..., 1] = self.v * np.sin(state[..., 2])
        return f

    def control_matrix(self, state, time=0.0):
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 2, 0] = 1
        return B

    def disturbance_jacobian(self, state, time=0.0):
        return np.repeat(np.zeros_like(state)[..., None], 1, axis=-1)

    def state_jacobian(self, state, control, time=0.0):
        J = np.repeat(np.zeros_like(state)[..., None], self.n_dims, axis=-1)
        J[..., 0, 2] = -self.v * np.sin(state[..., 2])
        J[..., 1, 2] = self.v * np.cos(state[..., 2])
        return J

