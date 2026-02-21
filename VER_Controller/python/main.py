import numpy as np
from src.robot import TwoLinkArm
from src.ver_controller import VER_Controller
from src.create_limit_cycle import create_limit_cycles_from_path
import matplotlib.pyplot as plt
from src.rk4 import rk4_step
from src.visualizer import animate_robot

# Define parameters
P_GAIN = 40
K_GAIN = 1.0
DT = 0.05
SIM_TIME = 40.0
time_steps = np.arange(0, SIM_TIME, DT)

# Create the robot and prepare the desired limit cycles
robot = TwoLinkArm(method='point')
tables1, tables2, PHASE_OFFSET_RAD = create_limit_cycles_from_path(l1=robot.L1, l2=robot.L2)
PHASE_OFFSET_DEG = np.rad2deg(PHASE_OFFSET_RAD)
print(f"Calculated Desired Phase Offset: {PHASE_OFFSET_DEG:.2f} degrees")

# Initialize robot state and controllers
robot_state = np.array([-1.0, 2.5, -0.5, 0.5])
controller1 = VER_Controller(tables1, P_GAIN, K_GAIN)
controller2 = VER_Controller(tables2, P_GAIN, K_GAIN)
controllers = [controller1, controller2]

# Define phase offsets
phase_offsets = np.array([PHASE_OFFSET_RAD, -PHASE_OFFSET_RAD])

# Data logging
history = {'q1':[], 'q2':[], 'q1_dot':[], 'q2_dot':[], 'tau1':[], 'tau2':[], 
           'pitch1':[], 'pitch2':[]}

# Main Simulation Loop
for t in time_steps:
    q1, q2, q1_dot, q2_dot = robot_state
    
    _, pitch1 = controller1.get_desired_acceleration([q1, q1_dot], history['pitch2'][-1] if t > 0 else 0, phase_offsets[0])
    _, pitch2 = controller2.get_desired_acceleration([q2, q2_dot], history['pitch1'][-1] if t > 0 else 0, phase_offsets[1])
    
    u1, _ = controller1.get_desired_acceleration([q1, q1_dot], pitch2, phase_offsets[0])
    u2, _ = controller2.get_desired_acceleration([q2, q2_dot], pitch1, phase_offsets[1])

    M,CG,_ = robot.forward_dynamics(robot_state[:2], robot_state[2:], tau = 0.0)

    tau = M @ np.array([u1, u2]) + CG
    
    history['q1'].append(q1); history['q2'].append(q2)
    history['q1_dot'].append(q1_dot); history['q2_dot'].append(q2_dot)
    history['tau1'].append(tau[0]); history['tau2'].append(tau[1])
    history['pitch1'].append(pitch1); history['pitch2'].append(pitch2)

    robot_state = rk4_step(robot_state, DT, robot, controllers, phase_offsets)

# visualization

fig, axs = plt.subplots(3, 2, figsize=(20, 30))
fig.suptitle(f'Phase offset:({PHASE_OFFSET_DEG:.2f}°)', fontsize=16)

# Desired LC1 and Actual LC1
x_d = tables1['center'] + tables1['r_d'] * np.cos(tables1['theta']) 
y_d = tables1['r_d'] * np.sin(tables1['theta'])
axs[0, 0].plot(history['q1'], history['q1_dot'], label='Joint 1 Trajectory')
axs[0, 0].plot(x_d, y_d,'k--', label='Desired LC 1')
axs[0, 0].set_title('Phase Space: Joint 1'); axs[0, 0].legend(); axs[0, 0].axis('equal')
axs[0,0].grid(True)
# Desired LC2 and Actual LC2
axs[0, 1].plot(history['q2'], history['q2_dot'], 'g-', label='Joint 2 Trajectory')
axs[0, 1].plot(tables2['center'] + tables2['r_d'] * np.cos(tables2['theta']), tables2['r_d'] * np.sin(tables2['theta']), 'k--', label='Desired LC 2')
axs[0, 1].set_title('Phase Space: Joint 2'); axs[0, 1].legend(); axs[0, 1].axis('equal')
axs[1, 0].plot(time_steps, history['q1'], label='Joint 1 Position')
axs[1, 0].plot(time_steps, history['q2'], 'g-', label='Joint 2 Position')
axs[1, 0].set_title('Position vs. Time'); axs[1, 0].legend()

# Pitch difference plot
unwrapped_pitch1 = np.unwrap(history['pitch1']); unwrapped_pitch2 = np.unwrap(history['pitch2'])
pitch_diff = (np.rad2deg(unwrapped_pitch1 - unwrapped_pitch2) + 180) % 360 - 180
axs[1, 1].plot(time_steps, pitch_diff, label='Actual Pitch Difference')
axs[1, 1].axhline(y=PHASE_OFFSET_DEG, color='r', linestyle='--', label='Desired Offset')
axs[1, 1].set_title('Pitch Difference (Joint 1 - Joint 2)'); axs[1, 1].legend(); axs[1, 1].grid(True)

# control signal plot
axs[2, 0].plot(time_steps, history['tau1'], label='Joint 1 Torque (τ1)')
axs[2, 0].plot(time_steps, history['tau2'], 'g-', label='Joint 2 Torque (τ2)')
axs[2, 0].set_title('Control Torques vs. Time'); axs[2, 0].legend()

# Wrapped Pitch vs. Time
wrapped_pitch1 = (np.array(history['pitch1']) + np.pi) % (2 * np.pi) - np.pi
wrapped_pitch2 = (np.array(history['pitch2']) + np.pi) % (2 * np.pi) - np.pi
axs[2, 1].plot(time_steps, wrapped_pitch1, label='Joint 1 Pitch')
axs[2, 1].plot(time_steps, wrapped_pitch2, 'g-', label='Joint 2 Pitch')
axs[2, 1].set_title('Wrapped Pitch vs. Time'); axs[2, 1].legend()

## Add grid
for ax_row in axs:
    for ax in ax_row:
        ax.grid(True)


fig.subplots_adjust(hspace=0.4)
plt.grid()
plt.show()

# plot end efector trajectory
q1 = np.array(history['q1'])
q2 = np.array(history['q2'])
x_e = robot.L1*np.cos(q1) + robot.L2 * np.cos(q1+q2)
y_e = robot.L1*np.sin(q1) + robot.L2*np.sin(q1+q2)
# Desired joint angles from tables
q1_d = tables1['center'] + tables1['r_d'] * np.cos(tables1['theta'])
q2_d = tables2['center'] + tables2['r_d'] * np.cos(tables2['theta'])
# Desired end-effector trajectory (forward kinematics)
x_e_d = robot.L1 * np.cos(q1_d) + robot.L2 * np.cos(q1_d + q2_d)
y_e_d = robot.L1 * np.sin(q1_d) + robot.L2 * np.sin(q1_d + q2_d)
plt.figure(figsize=(10,5))
plt.plot(x_e, y_e)
plt.plot(x_e_d, y_e_d, '--k')
plt.legend(['End effector trajectory', ' Desired trajectory'])
plt.grid()
plt.show()

# animate two link manipulator
animate_robot(history, time_steps, robot.L1, robot.L2, tables1, tables2)