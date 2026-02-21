clc; close all; 

% Reference parameters
w = 1.0;      
N = 1000;     
t = linspace(0, 2*pi, N);

circle_center = [0.0; -1]; 
circle_rad = 0.5;

% Desired trajectory
x_d = circle_center(1) + circle_rad * sin(w*t);
y_d = circle_center(2) + circle_rad * cos(w*t);

% Actual trajectory 
slave_ee_x = out.slave_end_effector_pos.Data(:, 1);
slave_ee_y = out.slave_end_effector_pos.Data(:, 2);

% Trajectory Plot
figure;
plot(slave_ee_x, slave_ee_y, 'b', 'LineWidth', 1.5);
hold on;
plot(x_d, y_d, 'r', 'LineWidth', 1.5);
grid on; 
axis equal;
legend('Actual', 'Desired'); 

% Lambda Force Plot
figure;
plot(out.lambda_force.Data, 'LineWidth', 1.5);
grid minor;
legend('X', 'Y');

% World Force Plot
figure;
F_world = squeeze(out.F_world.Data)';
plot(-F_world(:, 1:2), 'LineWidth', 1.5);
grid minor;
legend('X', 'Y');

% Slave Torque Plot
figure;
tau_slave = squeeze(out.tau_slave.Data)';
plot(tau_slave(:, 1:2), 'LineWidth', 1.5);
grid minor;
legend('tau1', 'tau2');