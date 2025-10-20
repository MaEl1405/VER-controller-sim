import numpy as np
from src.limit_cycle import table_rho
import matplotlib.pyplot as plt
import pdb
import numpy as np
from src.robot import TwoLinkArm
import numpy as np
from src.robot import TwoLinkArm
from src.ver_controller import VER_Controller
import matplotlib.pyplot as plt
from src.rk4 import rk4_step
from src.visualizer import animate_robot
from src.forward_kinematic import forward_kinematics

##############################################################################

w = 1.0
N = 1000
T = 2*np.pi
t = np.linspace(0, T, N)
circle_center = [1 ,-1]
circle_rad = 0.5
P_GAIN = 80.0
K_GAIN = 10.0
DT = 0.005
SIM_TIME = 50.0
time_steps = np.arange(0, SIM_TIME, DT)

#############################################################################

tables1, x_rho = table_rho("x", w=w, t=t, circle_center=circle_center, circle_rad=circle_rad)
tables2, y_rho = table_rho("y", w=w, t=t, circle_center=circle_center, circle_rad=circle_rad)
PHASE_OFFSET_RAD = x_rho-y_rho


######################################################################

robot = TwoLinkArm(method='distributed')
PHASE_OFFSET_DEG = np.rad2deg(PHASE_OFFSET_RAD[0])
print(f"Calculated Desired Phase Offset: {PHASE_OFFSET_DEG} degrees")

# Initialize robot state and controllers
robot_state = np.array([0.0, -np.pi/4, 0.0, 0.0])
controller1 = VER_Controller(tables1, P_GAIN, K_GAIN)
controller2 = VER_Controller(tables2, P_GAIN, K_GAIN)
controllers = [controller1, controller2]

# Define phase offsets
phase_offsets = np.array([PHASE_OFFSET_RAD[0], -PHASE_OFFSET_RAD[0]])

#######################################################################

# Data logging
history = {'beta1':[], 'beta2':[], 'q1':[], 'q2':[], 'q1_dot':[], 'q2_dot':[], 'x':[], 'xdot':[], 'y':[], 'ydot':[],
           'pitch1':[], 'pitch2':[]}

# Main Simulation Loop
for t in time_steps:
    q1, q2, q1_dot, q2_dot = robot_state
    
    x,y = forward_kinematics(q1, q2)
    xdot, ydot = robot.get_jacob(q1, q2) @ [q1_dot, q2_dot]

    pitch1 = controller1.get_pitch([x, xdot])
    pitch2 = controller2.get_pitch([y, ydot])
    

    u1, beta1 = controller1.get_desired_acceleration([x, xdot], pitch2, phase_offsets[0])
    u2, beta2 = controller2.get_desired_acceleration([y, ydot], pitch1, phase_offsets[1])

    history['beta1'].append(beta1)
    history['beta2'].append(beta2)




    history['x'].append(x); history['xdot'].append(xdot)
    history['y'].append(y); history['ydot'].append(ydot)

    history['q1'].append(q1); history['q2'].append(q2)
    history['q1_dot'].append(q1_dot); history['q2_dot'].append(q2_dot)
    # history['tau1'].append(tau[0]); history['tau2'].append(tau[1])
    history['pitch1'].append(pitch1); history['pitch2'].append(pitch2)

    robot_state = rk4_step(robot_state, DT, robot, controllers, phase_offsets)

# visualization
fig, axs = plt.subplots(3, 2, figsize=(20, 30))
fig.suptitle(f'Phase offset:({PHASE_OFFSET_DEG:.2f}°)', fontsize=16)

# Desired LC1 and Actual LC1
x_d = tables1['center'] + tables1['r_d'] * np.cos(tables1['theta']) 
y_d = tables1['r_d'] * np.sin(tables1['theta'])
axs[0, 0].plot(history['x'], history['xdot'], label='x axis LC')
axs[0, 0].plot(x_d, y_d,'k--', label='Desired LC 1')
axs[0, 0].set_title('Task Space: X-axis'); axs[0, 0].legend(); axs[0, 0].axis('equal')
axs[0,0].grid(True)
# Desired LC2 and Actual LC2
axs[0, 1].plot(history['y'], history['ydot'], 'g-', label='y-axis LC')
axs[0, 1].plot(tables2['center'] + tables2['r_d'] * np.cos(tables2['theta']), tables2['r_d'] * np.sin(tables2['theta']), 'k--', label='Desired LC 2')
axs[0, 1].set_title('Task Space:  y-axis'); axs[0, 1].legend(); axs[0, 1].axis('equal')
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
# axs[2, 0].plot(time_steps, history['tau1'], label='Joint 1 Torque (τ1)')
# axs[2, 0].plot(time_steps, history['tau2'], 'g-', label='Joint 2 Torque (τ2)')
# axs[2, 0].set_title('Control Torques vs. Time'); axs[2, 0].legend()

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
t = np.linspace(0, T, N)
x_e_d = circle_center[0] + circle_rad*np.sin(w*t)
y_e_d = circle_center[1] + circle_rad*np.cos(w*t)
plt.figure()
plt.plot(x_e, y_e)
plt.plot(x_e_d, y_e_d, '--r')
plt.legend([' Desired trajectory'])
plt.grid()
plt.axis("equal")
plt.show()

# animate two link manipulator
animate_robot(history, time_steps, robot.L1, robot.L2, tables1, tables2)