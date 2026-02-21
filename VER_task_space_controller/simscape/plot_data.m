clc; close all;

%% Parameters
w = 1.0;      % Frequency
N = 1000;     % Num points
T = 2*pi;     % Period
t = linspace(0, T, N);
circle_center = [1.0, -1];
circle_rad = 0.5;  % Circle radius

x = circle_center(1) + circle_rad*sin(w*t);
y = circle_center(2) + circle_rad*cos(w*t);

%% Visualization
% Plot actual and desired trajectory
plot(out.ee.Data(:,1), out.ee.Data(:,2), 'LineWidth',2)
hold on 
plot(x,y, 'LineWidth', 2)
legend('Actual', 'Desired')
grid minor 
axis equal