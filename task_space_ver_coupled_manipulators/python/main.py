import sys
import os
import pdb 
import numpy as np
import matplotlib.pyplot as plt
from VER_task_space_Controller.src.robot import TwoLinkArm
from coupled_manipulator_sim.src.rk4 import rk4_step
from VER_task_space_Controller.src.limit_cycle import table_rho
from coupled_manipulator_sim.src.visualizer import animate_robot
from VER_task_space_Controller.src.ver_controller import VER_Controller

## Define end effector desired path parameters
w = 1.0      #Frequency
N = 1000     #Num points
T = 2*np.pi  #Period
t = np.linspace(0, T, N)
circle_center = [0.5 ,-1]
circle_rad = 0.5  #circle radius

## Define main simulation time
SIM_TIME = 40  # Final simulation time
DT = 0.01      # Step size
time_steps = np.arange(0, SIM_TIME, DT)

# Define controller parameters
P_GAIN = 200.0  # Attractor gain
K_GAIN = 1.0   # Synchronizer gain

# Limit cycles parameters
tables1, x_rho = table_rho("x", w=w, t=t, circle_center=circle_center, circle_rad=circle_rad) # x-axis limit cycle
tables2, y_rho = table_rho("y", w=w, t=t, circle_center=circle_center, circle_rad=circle_rad) # y-axis limit cycle

PHASE_OFFSET_RAD = x_rho-y_rho  # Phase offset in radians
PHASE_OFFSET_DEG = np.rad2deg(PHASE_OFFSET_RAD[0]) # Phase offset in deg
phase_offsets = np.array([PHASE_OFFSET_RAD[0], -PHASE_OFFSET_RAD[0]])
print(f"Calculated Desired Phase Offset: {PHASE_OFFSET_DEG} degrees")

# Initialize master and slave two-link robots
master_base_point = np.array([0.0, 0.0])
master_robot = TwoLinkArm(method='distributed', base_point=master_base_point)

slave_base_point = np.array([1.0, 0.0])
slave_robot = TwoLinkArm(method='distributed', base_point=slave_base_point)

robots = [master_robot, slave_robot]

# Initialize robots state 
q_master_init = np.array([-0.5, -np.pi/1.7])
qdot_master_init = np.array([0.0, 0.0])
P_master_init = master_robot.forward_kinematics(q_master_init[0], q_master_init[1])

q_slave_init = np.array([-1.41498642, -1.52073533])
qdot_slave_init = np.array([0.0, 0.0])

robot_state = np.concatenate([q_master_init, q_slave_init, qdot_master_init, qdot_slave_init])


# Initialize Controllers
x_axis_controller = VER_Controller(tables1, P_GAIN, K_GAIN) 
y_axis_controller = VER_Controller(tables2, P_GAIN, K_GAIN)
controllers = [x_axis_controller, y_axis_controller]

# Data logging
history = {'q1_master':[], 'q2_master':[], 'q1dot_master':[], 'q2dot_master':[],
           'q1_slave':[], 'q2_slave':[], 'q1dot_slave':[], 'q2dot_slave':[],
            'x_master':[], 'xdot_master':[], 'y_master':[], 'ydot_master':[],
            'tau1_master':[], 'tau2_master':[],'lambda_x' :[], 'lambda_y':[],
            'x_axis_pitch':[], 'y_axis_pitch':[]
            }

# Main simulation Loop
for t in time_steps:
    q_master = robot_state[0:2]; q_slave = robot_state[2:4]
    qdot_master = robot_state[4:6]; qdot_slave = robot_state[6:8]
    
    # End effector position and velocity
    x_master, y_master = master_robot.forward_kinematics(q_master[0], q_master[1])
    J_master = master_robot.get_jacob(q_master[0], q_master[1]) 
    xdot_master, ydot_master = J_master @ qdot_master

    # Limit cycles pitch
    x_axis_pitch = x_axis_controller.get_pitch([x_master, xdot_master])
    y_axis_pitch = y_axis_controller.get_pitch([y_master, ydot_master])
    

    # Store data
    history['q1_master'].append(q_master[0]);       history['q2_master'].append(q_master[1])
    history['q1_slave'].append(q_slave[0]);         history['q2_slave'].append(q_slave[1])
    history['q1dot_master'].append(qdot_master[0]); history['q2dot_master'].append(qdot_master[1])
    history['q1dot_slave'].append(qdot_slave[0]);   history['q2dot_slave'].append(qdot_slave[1])
    history['x_master'].append(x_master);           history['xdot_master'].append(xdot_master)
    history['y_master'].append(y_master);           history['ydot_master'].append(ydot_master)
    # history['tau1_master'].append();                history['tau2_master'].append()
    history['lambda_x'].append(x_master);           history['lambda_y'].append(xdot_master)
    history['x_axis_pitch'].append(x_axis_pitch);   history['y_axis_pitch'].append(y_axis_pitch)

    # Apdate states using Runge–Kutta fourth-order
    robot_state,dbg = rk4_step(robot_state, DT, robots, controllers, phase_offsets)



# Visualize outputs 
fig, axs = plt.subplots(3, 2, figsize=(20, 30))
fig.suptitle(f'Phase offset:({PHASE_OFFSET_DEG:.2f}°)', fontsize=16)

# Desired LC1 and Actual LC1
x_d = tables1['center'] + tables1['r_d'] * np.cos(tables1['theta']) #
y_d = tables1['r_d'] * np.sin(tables1['theta'])
axs[0, 0].plot(history['x_master'], history['xdot_master'], label='x axis LC')
axs[0, 0].plot(x_d, y_d,'k--', label='Desired LC 1')
axs[0, 0].set_title('Task Space: X-axis'); axs[0, 0].legend(); axs[0, 0].axis('equal')
axs[0,0].grid(True)

# Desired LC2 and Actual LC2
axs[0, 1].plot(history['y_master'], history['ydot_master'], 'g-', label='y-axis LC')
axs[0, 1].plot(tables2['center'] + tables2['r_d'] * np.cos(tables2['theta']), tables2['r_d'] * np.sin(tables2['theta']), 'k--', label='Desired LC 2')
axs[0, 1].set_title('Task Space:  y-axis'); axs[0, 1].legend(); axs[0, 1].axis('equal')
axs[1, 0].plot(time_steps, history['q1_master'], label='Joint 1 Position')
axs[1, 0].plot(time_steps, history['q2_master'], 'g-', label='Joint 2 Position')
axs[1, 0].set_title('Position vs. Time'); axs[1, 0].legend()

# Pitch difference plot
unwrapped_x_axis_pitch = np.unwrap(history['x_axis_pitch']); unwrapped_y_axis_pitch = np.unwrap(history['y_axis_pitch'])
pitch_diff = (np.rad2deg(unwrapped_x_axis_pitch - unwrapped_y_axis_pitch) + 180) % 360 - 180
axs[1, 1].plot(time_steps, pitch_diff, label='Actual Pitch Difference')
axs[1, 1].axhline(y=PHASE_OFFSET_DEG, color='r', linestyle='--', label='Desired Offset')
axs[1, 1].set_title('Pitch Difference (Joint 1 - Joint 2)'); axs[1, 1].legend(); axs[1, 1].grid(True)

# Wrapped Pitch vs. Time
wrapped_x_axis_pitch = (np.array(history['x_axis_pitch']) + np.pi) % (2 * np.pi) - np.pi
wrapped_y_axis_pitch = (np.array(history['y_axis_pitch']) + np.pi) % (2 * np.pi) - np.pi
axs[2, 1].plot(time_steps, wrapped_x_axis_pitch, label='Joint 1 Pitch')
axs[2, 1].plot(time_steps, wrapped_y_axis_pitch, 'g-', label='Joint 2 Pitch')
axs[2, 1].set_title('Wrapped Pitch vs. Time'); axs[2, 1].legend()

## Add grid for plots
for ax_row in axs:
    for ax in ax_row:
        ax.grid(True)
        
fig.subplots_adjust(hspace=0.4)
plt.grid()
plt.show()

# plot end efector trajectory
q1 = np.array(history['q1_master'])
q2 = np.array(history['q2_master'])
x_e = master_robot.L1*np.cos(q1) + master_robot.L2 * np.cos(q1+q2)
y_e = master_robot.L1*np.sin(q1) + master_robot.L2*np.sin(q1+q2)
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
animate_robot(history, time_steps, master_robot.L1, master_robot.L2, tables1, tables2)


