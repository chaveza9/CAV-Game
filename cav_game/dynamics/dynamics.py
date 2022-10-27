import abc
import jax
import jax.numpy as jnp
from typing import Tuple
from cav_game.dynamics.tests import test_dynamics


# Create parent class for dynamics
class Dynamics(metaclass=abc.ABCMeta):
    def __init__(self, params: dict, test: bool = True, **kwargs):
        self.n_dims = len(self.STATES) if hasattr(self, "STATES") else params["n_dims"]
        self.control_dims = len(self.CONTROLS) if hasattr(self, "CONTROLS") else params["control_dims"]
        self.disturbance_dims = params.get("disturbance_dims", 1)
        self.dt = 0.1 if "dt" not in params else params["dt"]
        self.periodic_dims = self.PERIODIC_DIMS if hasattr(self, "PERIODIC_DIMS") else []
        if test:
            test_dynamics.test_dynamics(self)

    @abc.abstractmethod
    def __call__(self, state: jnp.ndarray, control: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
        """Implements the continuous-time dynamics ODE \dot{x} = f(x, u, t)"""

    def wrap_dynamics(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Periodic dimensions are wrapped to [-pi, pi)
        Args:
            state (jnp.ndarray): Unwrapped state
        Returns:
            state (jnp.ndarray): Wrapped state
        """
        for periodic_dim in self.periodic_dims:
            try:
                state[..., periodic_dim] = (state[..., periodic_dim] + jnp.pi) % (2 * jnp.pi) - jnp.pi
            except TypeError:  # FIXME: Clunky at best, how to deal with jnp and jnp mix
                state = state.at[periodic_dim].set((state[periodic_dim] + jnp.pi) % (2 * jnp.pi) - jnp.pi)

        return state

    def state_jacobian(self, state: jnp.ndarray, control: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
        raise NotImplementedError("Define state_jacobian in subclass")

    def control_jacobian(self, state: jnp.ndarray, control: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
        raise NotImplementedError("Define control_jacobian in subclass")

    def linearized_ct_dynamics(
        self, state: jnp.ndarray, control: jnp.ndarray, time: float = 0.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Computes the linearized state and control matrices, A and B, of the continuous-time dynamics"""
        return self.state_jacobian(state, control, time), self.control_jacobian(state, control, time)

    def linearized_dt_dynamics(self, state: jnp.ndarray, control: jnp.ndarray, time: float = 0.0):
        """Implements the linearized discrete-time dynamics"""
        A, B = self.linearized_ct_dynamics(state, control, time)
        A_d = jnp.eye(self.n_dims) + self.dt * A
        B_d = self.dt * B
        return A_d, B_d

    def step(self, state: jnp.ndarray, control: jnp.ndarray, time: float = 0.0, scheme: str = "fe") -> jnp.ndarray:
        """Implements the discrete-time dynamics ODE
        scheme in {fe, rk4}"""
        if scheme == "fe":
            n_state = state + self(state, control, time) * self.dt
        elif scheme == "rk4":
            # TODO: Figure out how to do RK4 with periodic dimensions (aka angle normalization)
            # Assumes zoh on control
            k1 = self(state, control, time)
            k2 = self(state + k1 * self.dt / 2, control, time + self.dt / 2)
            k3 = self(state + k2 * self.dt / 2, control, time + self.dt / 2)
            k4 = self(state + k3 * self.dt, control, time + self.dt)
            n_state = state + (k1 + 2 * k2 + 2 * k3 + k4) * self.dt / 6
        else:
            raise ValueError("scheme must be either 'fe' or 'rk4'")
        return self.wrap_dynamics(n_state)


# Create a control affine dynamics class
class ControlAffineDynamics(Dynamics):
    def __init__(self, params: dict, test: bool = False, **kwargs):
        super().__init__(params, test, **kwargs)
        if test:
            test_dynamics.test_control_affine_dynamics(self)

    def __call__(self, state: jnp.ndarray, control: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
        """Implements the continuous-time dynamics ODE: dx_dt = f(x, t) + g(x, t) @ u"""
        return (
            self.open_loop_dynamics(state, time) + (self.control_matrix(state, time) @ jnp.atleast_3d(control)).squeeze()
        )

    @abc.abstractmethod
    def open_loop_dynamics(self, state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
        """Implements the open loop dynamics f(x,t)"""

    def control_jacobian(self, state: jnp.ndarray, control: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
        """For control affine systems the control_jacobian is equivalent to the control matrix"""
        return self.control_matrix(state, time)
    
    def state_jacobian(self, state, control, time=0.0):
        # Creates the state jacobian dF/dx
        return jax.jacfwd(self.open_loop_dynamics, argnums=1)(state, control, time)

    @abc.abstractmethod
    def control_matrix(self, state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
        """Implements the control Jacobian g(x,u) f(x,u,t)"""

    def disturbance_jacobian(self, state: jnp.ndarray, time: float = 0.0) -> jnp.ndarray:
        # TODO: Is this required?
        return jnp.atleast_2d(jnp.zeros((self.n_dims, self.disturbance_dims)))
