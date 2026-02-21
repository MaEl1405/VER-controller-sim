clear; clc; close all;

%% Define end effector desired path parameters 
w = 1.0;      % Frequency
N = 1000;     % Num points
T = 2*pi;     % Period
t = linspace(0, T, N);
circle_center = [1, -1];
circle_rad = 0.5;  % Circle radius

%% Generate Limit Cycle Tables

% X-axis limit cycle 
[x_pos, x_vel, x_acc] = calc_limit_cycle("x", w, t, circle_center, circle_rad);
[tables1, x_rho] = prepare_joint_tables(x_pos, x_vel, x_acc, "x");


% Y-axis limit cycle 
[y_pos, y_vel, y_acc] = calc_limit_cycle("y", w, t, circle_center, circle_rad);
[tables2, y_rho] = prepare_joint_tables(y_pos, y_vel, y_acc, "y");
tables2.r_d = circle_rad * ones(length(tables1.r_d), 1);


%% Define Controller & Sim Parameters
P_GAIN = 50.0;  % Attractor gain
K_GAIN = 1.0;   % Synchronizer gain

% Calculate phase offset
PHASE_OFFSET_RAD = x_rho - y_rho;
PHASE_OFFSET_DEG = rad2deg(PHASE_OFFSET_RAD(1));
phase_offsets = [PHASE_OFFSET_RAD(1), -PHASE_OFFSET_RAD(1)];
fprintf('Calculated Desired Phase Offset: %.2f degrees\n', PHASE_OFFSET_DEG);

% Initial robot state [q1, q2, q1_dot, q2_dot]
robot_state_initial = [-0.5, -pi/1.7, 0.0, 0.0];

% Robot parameters 
L1 = 1.0;
L2 = 1.0;

% Simulation time
SIM_TIME = 40;
DT = 0.01;


%% Create robot parameters struct

robot_params = struct();
robot_params.m1 = 1.0;
robot_params.m2 = 1.0;
robot_params.L1 = 1.0;
robot_params.L2 = 1.0;
robot_params.g = 9.81;
robot_params.link_width = 0.1; % Used for distributed mass inertia

robot_params.I1 = (robot_params.m1 * (robot_params.L1^2 + robot_params.link_width^2)) / 12;
robot_params.I2 = (robot_params.m2 * (robot_params.L2^2 + robot_params.link_width^2)) / 12;


% Create BusElement objects
elems(1) = Simulink.BusElement;
elems(1).Name = 'm1';
elems(1).DataType = 'double';

elems(2) = Simulink.BusElement;
elems(2).Name = 'm2';
elems(2).DataType = 'double';

elems(3) = Simulink.BusElement;
elems(3).Name = 'L1';
elems(3).DataType = 'double';

elems(4) = Simulink.BusElement;
elems(4).Name = 'L2';
elems(4).DataType = 'double';

elems(5) = Simulink.BusElement;
elems(5).Name = 'g';
elems(5).DataType = 'double';

elems(6) = Simulink.BusElement;
elems(6).Name = 'link_width';
elems(6).DataType = 'double';

elems(7) = Simulink.BusElement;
elems(7).Name = 'I1';
elems(7).DataType = 'double';

elems(8) = Simulink.BusElement;
elems(8).Name = 'I2';
elems(8).DataType = 'double';

% Create the main Bus object
bus_RobotParams = Simulink.Bus;
bus_RobotParams.Elements = elems;

% Clean up the temporary variable
clear elems;

[dim_theta, ~] = size(tables1.theta);
[dim_rd, ~]    = size(tables1.r_d);
[dim_beta, ~]  = size(tables1.beta);

% Create BusElement objects
elems_tbl(1) = Simulink.BusElement;
elems_tbl(1).Name = 'center';
elems_tbl(1).DataType = 'double';
elems_tbl(1).Dimensions = 1; % It's a scalar

elems_tbl(2) = Simulink.BusElement;
elems_tbl(2).Name = 'theta';
elems_tbl(2).DataType = 'double';
elems_tbl(2).Dimensions = [dim_theta, 1]; % Dynamic size from created table

elems_tbl(3) = Simulink.BusElement;
elems_tbl(3).Name = 'r_d';
elems_tbl(3).DataType = 'double';
elems_tbl(3).Dimensions = [dim_rd, 1]; 

elems_tbl(4) = Simulink.BusElement;
elems_tbl(4).Name = 'beta';
elems_tbl(4).DataType = 'double';
elems_tbl(4).Dimensions = [dim_beta, 1]; 

% Create the main Bus object
bus_Table = Simulink.Bus;
bus_Table.Elements = elems_tbl;

% Clean up the temporary variable
clear elems_tbl dim_theta dim_rd dim_beta;


%% Helper functions
function [table, rho] = prepare_joint_tables(q, q_dot, q_ddot, axis_name)

    q = q(:);
    q_dot = q_dot(:);
    q_ddot = q_ddot(:);
        
    center = mean(q);
    
    
    q_centered = q - center;                      
    theta_unwrapped = unwrap(atan2(q_dot, q_centered)); 
    r_d = sqrt(q_centered.^2 + q_dot.^2);               
    [sorted_theta_unwrapped, sort_idx] = sort(theta_unwrapped);
    
    sorted_r_d = r_d(sort_idx);
    sorted_beta = q_ddot(sort_idx);
    
    table = struct();
    table.center = center;
    table.theta = sorted_theta_unwrapped; 
    table.r_d = sorted_r_d;
    table.beta = sorted_beta;
    
    rho = linspace(theta_unwrapped(1), theta_unwrapped(end), length(q));
end

function [pos, vel, acc] = calc_limit_cycle(axis_name, w, t, circle_center, circle_rad)
    % Compute position, velocity, and acceleration for circular motion
    if axis_name == "x"
        % X-axis: sine-based motion
        pos = circle_center(1) + circle_rad * sin(w * t);
        vel = circle_rad * w * cos(w * t);
        acc = -circle_rad * (w ^ 2) * sin(w * t);

    elseif axis_name == "y"
        % Y-axis: cosine-based motion
        pos = circle_center(2) + circle_rad * cos(w * t);
        vel = -circle_rad * w * sin(w * t);
        acc = -circle_rad * (w ^ 2) * cos(w * t);
    else
        error('Axis name must be "x" or "y"');
    end
end
