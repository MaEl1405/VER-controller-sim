# Virtual Energy Regulator (VER) Simulation

This repository contains a Python simulation of the **Virtual Energy Regulator (VER)**, a time-independent controller designed for generating stable, synchronized cyclic motions in robotic systems. The simulation is implemented for a 2-link planar manipulator, demonstrating how to generate complex limit cycles from a simple task-space trajectory (e.g., a circle).

## Core Concepts

The VER controller replaces both the traditional pattern generator and the tracking controller. It operates directly in the phase space (position vs. velocity) and is composed of three main components. The output of the VER, `u_ver`, represents the desired joint acceleration.

### Control Law: `u_ver = α + β + γ`

1.  **Attractor (α):** This component acts like a "virtual spring-damper" in the phase space. It generates a force to pull the system's state onto the desired limit cycle trajectory. The force is proportional to the "virtual energy" error (the difference between the current and desired squared radius in the phase plane).

    $ \alpha = P \cdot \tanh(\dot{q}) \cdot (r_d^2 - r^2) $

    -   `P`: Attractor gain.
    -   `r`: Current radius from the limit cycle center.
    -   `r_d`: Desired radius for the current phase.

2.  **Accelerator (β):** This is a feed-forward term that provides the necessary acceleration to move along the limit cycle at the desired speed. It is pre-computed and stored in a lookup table based on the desired trajectory.

3.  **Synchronizer (γ):** This component couples multiple joints together, forcing them to maintain a specific phase relationship. It calculates a corrective force based on the difference in the "pitch" (the fundamental harmonic phase) of the coupled joints.

    $ \gamma_i = K \cdot (\sin(\rho_j + \phi) - \sin(\rho_i)) $

    -   `K`: Synchronizer gain.
    -   `ρ`: The pitch of each joint.
    -   `φ`: The desired phase offset.

### Dynamics and Control Implementation

To apply this to a real robot, we use **Feedback Linearization**. The controller calculates the desired acceleration (`u_ver`), and then a dynamic model of the robot is used to compute the required torques (`τ`) to achieve that acceleration.

$ \tau = M(q) \cdot u_{ver} + H(q, \dot{q}) $

-   `M(q)`: The mass matrix of the robot.
-   `H(q, \dot{q})`: The vector of Coriolis, centrifugal, and gravity forces (`C(q, \dot{q}) + G(q)`).

---

## Project Structure

The project is organized into modular Python files:

-   `main.py`: The main script to configure and run the simulation, and to generate plots.
-   `robot.py`: Defines the `TwoLinkArm` class, which includes the full dynamic model (M, C, G matrices).
-   `ver_controller.py`: Implements the `VER_Controller` class, which calculates the desired acceleration `u_ver`.
-   `create_limit_cycle.py`: The "Toolbox" module. It takes a task-space path, performs inverse kinematics, and generates all necessary lookup tables for the VER controller. It also automatically calculates the required phase offset between the joints.
-   `state_derivatives.py`: Defines the state-space representation of the robot's dynamics for the integrator.
-   `rk4.py`: Implements the 4th-order Runge-Kutta (RK4) numerical integrator for accurate simulation.
-   `inverse_kinematics.py`: A helper function for the toolbox.

---

## How to Run

1.  Ensure you have `numpy`, `matplotlib`, and `scipy` installed.
2.  Configure the simulation parameters in `main.py`:
    -   `P_GAIN`: The gain for the attractor component.
    -   `K_GAIN`: The gain for the synchronizer component.
    -   `DT`: The simulation time step.
    -   `SIM_TIME`: The total duration of the simulation.
3.  Run the main script from your terminal:
    ```bash
    python main.py
    ```

The script will run the simulation and display several figures showing the phase space trajectories, joint positions, control torques, phase synchronization performance, end effector tracking performnace and final animation.

