clear; clc; close all;

% Limit cycle and trajectory parameters
w = 1.0;      
N = 1000;     
T = 2*pi;     
t = linspace(0, T, N);
circle_center = [0.0; -1]; 
circle_rad = 0.5;

% Generate limit cycle tables
[x_pos, x_vel, x_acc] = calc_limit_cycle("x", w, t, circle_center, circle_rad);
[tables1, x_rho] = prepare_joint_tables(x_pos, x_vel, x_acc, circle_center(1));

[y_pos, y_vel, y_acc] = calc_limit_cycle("y", w, t, circle_center, circle_rad);
[tables2, y_rho] = prepare_joint_tables(y_pos, y_vel, y_acc, circle_center(2));

% Controller parameters
P_GAIN = 200.0; 
K_GAIN = 1.0;   
alpha_stabilize = 100.0; 
beta_stabilize  = 20.0;  

% Phase offset calculation
PHASE_OFFSET_RAD = x_rho(1) - y_rho(1);
phase_offsets = [PHASE_OFFSET_RAD, -PHASE_OFFSET_RAD];

% Robot parameters
robot_params = struct();
robot_params.m1 = 1.0;
robot_params.m2 = 1.0;
robot_params.L1 = 1.0;
robot_params.L2 = 1.0;
robot_params.g = 9.81;
robot_params.link_width = 0.1;
robot_params.I1 = (robot_params.m1 * (robot_params.L1^2 + robot_params.link_width^2)) / 12;
robot_params.I2 = (robot_params.m2 * (robot_params.L2^2 + robot_params.link_width^2)) / 12;

robot_params_master = robot_params;
robot_params_slave  = robot_params;

% Initial conditions 
base_1 = [0, 0];
base_2 = [0, 0];
x_target = x_pos(1) + 0.3;
y_target = y_pos(1);

[q1_m, q2_m] = Inv_kinematics(x_target, y_target, base_1, robot_params_master.L1, robot_params_master.L2, -1); 
[q1_s, q2_s] = Inv_kinematics(x_target, y_target, base_2, robot_params_slave.L1, robot_params_slave.L2, 1); 

fprintf('Master Init: q1=%.4f, q2=%.4f\n', q1_m, q2_m);
fprintf('Slave Init : q1=%.4f, q2=%.4f\n', q1_s, q2_s);

% Initial state vector [q_m; q_s; qdot_m; qdot_s]
robot_state_initial = [
    q1_m; 
    q2_m; 
    q1_s; 
    q2_s;
    0.0;
    0.0;
    0.0;
    0.0
];

% Create bus objects
elems(1) = Simulink.BusElement; elems(1).Name = 'm1';
elems(2) = Simulink.BusElement; elems(2).Name = 'm2';
elems(3) = Simulink.BusElement; elems(3).Name = 'L1';
elems(4) = Simulink.BusElement; elems(4).Name = 'L2';
elems(5) = Simulink.BusElement; elems(5).Name = 'g';
elems(6) = Simulink.BusElement; elems(6).Name = 'link_width';
elems(7) = Simulink.BusElement; elems(7).Name = 'I1';
elems(8) = Simulink.BusElement; elems(8).Name = 'I2';
for i=1:8, elems(i).DataType = 'double'; end
bus_RobotParams = Simulink.Bus;
bus_RobotParams.Elements = elems;
clear elems;

[dim_theta, ~] = size(tables1.theta);
elems_tbl(1) = Simulink.BusElement; elems_tbl(1).Name = 'center'; elems_tbl(1).Dimensions = 1;
elems_tbl(2) = Simulink.BusElement; elems_tbl(2).Name = 'theta'; elems_tbl(2).Dimensions = [dim_theta, 1];
elems_tbl(3) = Simulink.BusElement; elems_tbl(3).Name = 'r_d'; elems_tbl(3).Dimensions = [dim_theta, 1];
elems_tbl(4) = Simulink.BusElement; elems_tbl(4).Name = 'beta'; elems_tbl(4).Dimensions = [dim_theta, 1];
for i=1:4, elems_tbl(i).DataType = 'double'; end
bus_Table = Simulink.Bus;
bus_Table.Elements = elems_tbl;
clear elems_tbl dim_theta;

% Helper Functions
function [pos, vel, acc] = calc_limit_cycle(axis_name, w, t, center, rad)
    if axis_name == "x"
        pos = center(1) + rad * sin(w * t);
        vel = rad * w * cos(w * t);
        acc = -rad * (w ^ 2) * sin(w * t);
    elseif axis_name == "y"
        pos = center(2) + rad * cos(w * t);
        vel = -rad * w * sin(w * t);
        acc = -rad * (w ^ 2) * cos(w * t);
    end
end

function [table, rho] = prepare_joint_tables(p, p_dot, p_ddot, center_val)
    p = p(:); p_dot = p_dot(:); p_ddot = p_ddot(:);
    
    p_centered = p - center_val;                      
    theta_unwrapped = unwrap(atan2(p_dot, p_centered));
    r_d = sqrt(p_centered.^2 + p_dot.^2);
    
    [sorted_theta, sort_idx] = sort(theta_unwrapped);
    
    table = struct();
    table.center = center_val;
    table.theta = sorted_theta; 
    table.r_d = r_d(sort_idx);
    table.beta = p_ddot(sort_idx);
    
    rho = linspace(theta_unwrapped(1), theta_unwrapped(end), length(p));
end

function [q1, q2] = Inv_kinematics(x_global, y_global, base, L1, L2, sigma)
    x = x_global - base(1);
    y = y_global - base(2);
    
    r_sq = x^2 + y^2;
    c2 = (r_sq - L1^2 - L2^2) / (2*L1*L2);
    
    if abs(c2) > 1
        error('Target point is out of workspace!');
    end
    
    s2 = sigma * sqrt(1 - c2^2);
    q2 = atan2(s2, c2);
    
    k1 = L1 + L2 * c2;
    k2 = L2 * s2;
    q1 = atan2(y, x) - atan2(k2, k1);
end