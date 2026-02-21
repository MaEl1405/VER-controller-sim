from .state_derivatives import state_derivatives
def rk4_step(state, dt, robot, controllers, phase_offsets):
    k1 = state_derivatives(state, robot, controllers, phase_offsets)
    k2 = state_derivatives(state + 0.5 * dt * k1, robot, controllers, phase_offsets)
    k3 = state_derivatives(state + 0.5 * dt * k2, robot, controllers, phase_offsets)
    k4 = state_derivatives(state + dt * k3, robot, controllers, phase_offsets)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
