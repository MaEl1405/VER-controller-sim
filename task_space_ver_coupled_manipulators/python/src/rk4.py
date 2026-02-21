from .state_derivatives import state_derivatives

# Single RK4 integration step for robot state update
def rk4_step(state, dt, robots, controllers, phase_offsets):
    k1, dbg1 = state_derivatives(state, robots, controllers, phase_offsets, dt, 0)
    k2, dbg = state_derivatives(state + 0.5 * dt * k1, robots, controllers, phase_offsets, dt, 0.5*dt)
    k3, dbg = state_derivatives(state + 0.5 * dt * k2, robots, controllers, phase_offsets, dt, 0.5*dt)
    k4, dbg = state_derivatives(state + dt * k3, robots, controllers, phase_offsets, dt, dt)

    new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return new_state, dbg1
