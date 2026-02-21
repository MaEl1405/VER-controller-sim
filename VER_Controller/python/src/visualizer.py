import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_robot(history_states, time_steps, L1, L2, tables1=None, tables2=None):
    """
    Animates the motion of the two-link robot from simulation history.
    """
    # Calculate Cartesian coordinates from the loaded history
    theta1_rad = np.array(history_states['q1'])
    theta2_rad = np.array(history_states['q2'])
    x1 = L1 * np.cos(theta1_rad)
    y1 = L1 * np.sin(theta1_rad)
    x2 = x1 + L2 * np.cos(theta1_rad + theta2_rad)
    y2 = y1 + L2 * np.sin(theta1_rad + theta2_rad)

    if tables1 is not None and tables2 is not None:
        theta1_d = tables1['center'] + tables1['r_d'] * np.cos(tables1['theta'])
        theta2_d = tables2['center'] + tables2['r_d'] * np.cos(tables2['theta'])
        x1_d = L1 * np.cos(theta1_d)
        y1_d = L1 * np.sin(theta1_d)
        x2_d = x1_d + L2 * np.cos(theta1_d + theta2_d)
        y2_d = y1_d + L2 * np.sin(theta1_d + theta2_d)
    else:
        x2_d, y2_d = [], []


    # animation figure setup
    fig_anim = plt.figure(figsize=(8, 8))
    ax_anim = plt.axes(xlim=(-(L1+L2)*1.1, (L1+L2)*1.1), ylim=(-(L1+L2)*1.1, (L1+L2)*1.1))
    ax_anim.set_aspect('equal')
    ax_anim.set_title("Two-Link Robot Animation")
    ax_anim.grid()

    line, = ax_anim.plot([], [], 'o-', lw=3, markersize=8, color='#d62728')
    target_line, = ax_anim.plot([], [], 'o--', lw=2, markersize=8, color='#1f77b4', alpha=0.5)
    time_text = ax_anim.text(0.05, 0.95, '', transform=ax_anim.transAxes)

    # plot desired trajectory if exists
    desired_traj_line = None
    if len(x2_d) > 0:
        desired_traj_line, = ax_anim.plot(x2_d, y2_d, 'k--', lw=1.5, label="Desired Trajectory", alpha=0.6)
        ax_anim.legend()

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        """
        Animation function for matplotlib's FuncAnimation.
        Updates the line and time text at each frame of the animation.
        """
        line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        time_text.set_text(f'Time = {time_steps[i]:.2f}s')
        return line, time_text

    # time step
    dt = time_steps[1] - time_steps[0]
    
    # create the animation 
    ani = FuncAnimation(fig_anim, animate, frames=len(time_steps),
                        interval=dt*400, blit=True, init_func=init, repeat=True)

    plt.show()