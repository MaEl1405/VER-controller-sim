import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_robot(history_states, time_steps, L1, L2, tables1=None, tables2=None):
    """Animate two-link master & slave robots together."""
    # Desired trajectory params
    w = 1
    P = 2 * np.pi / w
    N = 2000
    circle_rad = 0.5
    t = np.linspace(0, P, N)

    # Master manipulator
    q1m = np.array(history_states['q1_master'])
    q2m = np.array(history_states['q2_master'])
    x1m = L1 * np.cos(q1m)
    y1m = L1 * np.sin(q1m)
    x2m = x1m + L2 * np.cos(q1m + q2m)
    y2m = y1m + L2 * np.sin(q1m + q2m)

    # Slave manipulator 
    offset_x = 1.0 # base offset
    q1s = np.array(history_states['q1_slave'])
    q2s = np.array(history_states['q2_slave'])
    x1s = offset_x + L1 * np.cos(q1s)
    y1s = L1 * np.sin(q1s)
    x2s = offset_x + (L1 * np.cos(q1s) + L2 * np.cos(q1s + q2s))
    y2s = L1 * np.sin(q1s) + L2 * np.sin(q1s + q2s)

    # Desired trajectory circle (for reference)
    x_des = 0.5 + circle_rad * np.sin(2 * np.pi * t)
    y_des = -1 + circle_rad * np.cos(2 * np.pi * t)

    # Figure setup
    fig_anim = plt.figure(figsize=(8, 8))
    ax = plt.axes(xlim=(-(L1+L2)*1.3, (L1+L2)*2),
                  ylim=(-(L1+L2)*1.3, (L1+L2)*1.3))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Two-Link Robots (Master & Slave)")
    ax.grid(True)

    # Lines
    line_master, = ax.plot([], [], 'o-', lw=3, color='#d62728', label='Master')
    line_slave, = ax.plot([], [], 'o-', lw=3, color='#2ca02c', alpha=0.7, label='Slave')
    traj_desired, = ax.plot(x_des, y_des, 'k--', lw=1.2, alpha=0.6, label='Desired Trajectory')
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    ax.legend()

    # Init
    def init():
        line_master.set_data([], [])
        line_slave.set_data([], [])
        time_text.set_text('')
        return line_master, line_slave, time_text

    # Animate each frame
    def animate(i):
        line_master.set_data([0, x1m[i], x2m[i]], [0, y1m[i], y2m[i]])
        line_slave.set_data([offset_x, x1s[i], x2s[i]],
                            [0, y1s[i], y2s[i]])
        time_text.set_text(f'Time = {time_steps[i]:.2f}s')
        return line_master, line_slave, time_text

    # Animation setup
    dt = time_steps[1] - time_steps[0]
    ani = FuncAnimation(fig_anim, animate,
                        frames=len(time_steps),
                        interval=dt*400,
                        blit=True, init_func=init,
                        repeat=True)

    plt.show()
