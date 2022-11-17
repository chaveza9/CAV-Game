clc
clear
%close all
addpath('G:\My Drive\Mixed Traffic\casadi-windows-matlabR2016a-v3.5.5')



%% Define Initial States
% % Vehicle U initial States;
% v_U_0 = 17;
% x_U_0 = 248.333;
% % Vehicle C States;
% v_C_0 = 23.91;
% x_C_0 = 178.448;
% % Vehicle 1 States;
% v_1_0 = 28.62;
% x_1_0 = 194.882;
% % Vehicle 2 States;
% v_2_0 = 28.62;
% x_2_0 = 172.699;

% % Vehicle U initial States;
% v_U_0 = 17;
% x_U_0 = 248.333;
% % Vehicle C States;
% v_C_0 = 23.91;
% x_C_0 = 178.448;
% % Vehicle 1 States;
% v_1_0 = 25.62;
% x_1_0 = 194.882;
% % Vehicle 2 States;
% v_2_0 = 28.62;
% x_2_0 = 162.699;
% Vehicle U initial States;
v_U_0 = 17;
x_U_0 = 888.333;
% Vehicle C States;
v_C_0 = 24;
x_C_0 = 17;
% Vehicle 1 States;
v_1_0 = 18;
x_1_0 = 18;
% Vehicle 2 States;
v_2_0 = 25;
x_2_0 = 0;
N = 300;
% 

% vehicle C terminal states
[tf,x_C_f,v_C_f,xC,vC,uC] = solve_cavC_ocp(x_C_0,v_C_0,x_U_0,v_U_0,N); % optimal cavC traj
%[x_1_f,v_1_f,x1,v1,u1] = solve_cav1_ocp(x_C_f,v_C_f,tf,x_1_0,v_1_0,N); % optimal cav1 traj
% [x_2_f,v_2_f,x2,v2,u2] = solve_cav2_ocp(x_C_f,tf,x_2_0,v_2_0,N); % optimal cav2 traj

import casadi.*

%% Define Phyisical Constraints
u_max = 3.3;    % Vehicle i max acceleration [m/s^2]
u_min = -7;   % Vehicle i min acceleration [m/s^2]
v_min = 15;   % Vehicle i min velocity [m/s]
v_max = 31;   % Vehicle i max velocity [m/s]
%% Tunning Variables
% Longitudinal Maneuver
phi = 0.6; % [seconds] Intervehicle Reaction Time
delta = 1.5; % [meters] intervehicle distance
% alpha = 0.2; % Optimization Weights
% N = 100; % number of control intervals
% v_des = 30; % free flow desired speed;
N = 100;
M = 100;
v_des = 30;
eps = 0.0001;



alpha_energy = 0.2;
alpha_hdv = 0.2;
alpha_speed = 1-alpha_hdv-alpha_energy;
%beta_time = alpha_time;
beta_speed = alpha_speed;
beta_energy = (alpha_energy)/max([u_max, u_min].^2);
beta_hdv = alpha_hdv;
gamma_energy = (alpha_energy)/max([u_max, u_min].^2);
%gamma_energy = 0;
gamma_speed = 1-alpha_energy-alpha_hdv;
gamma_safety = alpha_hdv;
x1_hist = [];
x2_hist = [];
xC_hist = [];
t = 0:tf/N:tf;
x1 = x_1_0 + v_1_0*t;
xC = x_C_0 + v_C_0*t;

%%  Numerical Solution
%opti = casadi.Opti(); % Optimization problem

for i=1:M
    [x_2_f,v_2_f,x2,v2,u2] = solve_hdv_ibrproblem (x1,tf,x_2_0,v_2_0,N,xC);
    x2_hist = cat(2,x2_hist,x2');
    %[x_1_f,v_1_f,x1,v1,u1,x_C_f,v_C_f,xC,vC,uC] = solve_cav1C_ibrproblem (x_2_f,v_2_f,N,x_1_0,v_1_0,x_C_0,v_C_0);
    %%  Numerical Solution
    opti = casadi.Opti(); % Optimization problem

    % ---- decision variables ---------
    % X = opti.variable(2,N+1); % state trajectory
    % pos   = X(1,:);
    % speed = X(2,:);
    % U = opti.variable(1,N);   % control trajectory (throttle)

    % ---- decision variables ---------
    U1 = opti.variable(1,N); % state trajectory
    X1 = opti.variable(2,N+1); % state trajectory
    pos1   = X1(1,:);
    speed1 = X1(2,:);
    U_C = opti.variable(1,N);   % control trajectory (throttle)
    X_C = opti.variable(2,N+1); % state trajectory
    posC = X_C(1,:);
    speedC = X_C(2,:);
    % Define objective funciton integrator
    % Integrate using RK4
    dt = tf/N;
    % Define dynamic constraints
    c = @(u, x) u^2; % Cost Acceleration
    f = @(x,u) [x(2);u]; % dx/dt = f(x,u)
    cost_u1 = 0;
    cost_uC = 0;
    cost_v1 = beta_speed*(speed1(end)-v_des)^2;
    cost_vC = beta_speed*(speedC(end)-v_des)^2;
    cost_safety = exp(x2(end)+phi*v2(end)-posC(end));
   % cost_safety = 1000*(x2(end)+phi*v2(end)-posC(end));

    for k=1:N % loop over control intervals
        % Forward integration
        x1_next = runge_kutta4(f, X1(:,k), U1(:,k), dt);
        xC_next = runge_kutta4(f, X_C(:,k), U_C(:,k), dt);
        cost_u1 = cost_u1 + 0.5*beta_energy*runge_kutta4(c, U1(:,k), X1(:,k), dt);
        cost_uC = cost_uC + 0.5*beta_energy*runge_kutta4(c, U_C(:,k), X_C(:,k), dt);
        %cost_v = cost_v + gamma_speed*runge_kutta4(l,(speed1(:,k)-v_des)^2,X(:,k),dt);
        % Impose multi-shoot constraint
        opti.subject_to(X1(:,k+1)==x1_next); % close the gaps
        opti.subject_to(X_C(:,k+1)==xC_next); % close the gaps
        % safety constraint
        %opti.subject_to(x1(k)-pos(k)>=speed(k)*phi+delta);
    end
    cost = cost_u1 + cost_uC + cost_v1 + cost_vC + cost_safety;
    % ----- Objective function ---------
    opti.minimize(cost);
    % --------- Define path constraints ---------
    opti.subject_to(v_min<=speed1<=v_max);     %#ok<CHAIN> % track speed limit
    opti.subject_to(u_min<=U1<=u_max);     %#ok<CHAIN> % control is limited
    opti.subject_to(v_min<=speedC<=v_max);     %#ok<CHAIN> % track speed limit
    opti.subject_to(u_min<=U_C<=u_max);     %#ok<CHAIN> % control is limited
    % --------- Define Boundary conditions ---------
    opti.subject_to(pos1(1)==x_1_0);   % start at position 0 ...
    opti.subject_to(speed1(1)==v_1_0); % initial speed
    opti.subject_to(posC(1)==x_C_0);   % start at position 0 ...
    opti.subject_to(speedC(1)==v_C_0); % initial speed
    opti.subject_to(pos1(end) - posC(end) >= phi*speedC(end) + delta);
   % opti.subject_to(posC(end) - x2(end) >= phi*v2(end) + delta);
    % Warm Start solver
    % opti.set_initial(speed, 25);
    opti.set_initial(U1, 3);

    %% Solver Parameter Options
    % opti.solver('ipopt',struct('print_time',obj.Verbose,'ipopt',...
    %     struct('max_iter',10000,'acceptable_tol',1e-8,'print_level',0,...
    %     'acceptable_obj_change_tol',1e-6))); % set numerical backend
    opti.solver('ipopt',struct('print_time',1,'ipopt',...
        struct('max_iter',10000,'acceptable_tol',1e-8,'print_level',3,...
        'acceptable_obj_change_tol',1e-6))); % set numerical backend
    % opti.solver('bonmin',struct('print_time',1,'bonmin',...
    %     struct('max_iter',100000))); % set numerical backend
    sol = opti.solve();   % actual solve
    x_1_f = sol.value(pos1(end));
    v_1_f = sol.value(speed1(end));
    x1 = sol.value(pos1);
    v1 = sol.value(speed1);
    u1 = sol.value(U1);
    x_C_f = sol.value(posC(end));
    v_C_f = sol.value(speedC(end));
    xC = sol.value(posC);
    vC = sol.value(speedC);
    uC = sol.value(U_C);
    x1_hist = cat(2,x1_hist,x1');
    xC_hist = cat(2,xC_hist,xC');
    %%check if the traj is converged
    % check vehicle2
    if i>=2
    err2 = norm(x2_hist(:,i)-x2_hist(:,i-1))
    %check vehicle 1 and C
    err1 = norm(x1_hist(:,i)-x1_hist(:,i-1))
    errC = norm(xC_hist(:,i)-xC_hist(:,i-1))
    if err2 <= 0.01 && err1 <= 0.01 && errC <= 0.01
        disp('converge')
        break
    else
        disp('unconverge')
    end
    end

end




Time = tf;
figure
subplot(3,1,1)
time = linspace(0,Time,N+1);
plot(linspace(0,Time,N+1),sol.value(x1))
hold on
plot(linspace(0,Time,N+1),sol.value(x2))
plot(linspace(0,Time,N+1),sol.value(x2)+phi*sol.value(v2)+delta)
hold on 
% plot(linspace(0,Time,N+1),sol.value(x1))
% hold on
% plot(linspace(0,Time,N+1),sol.value(x2))
% hold on
plot(linspace(0,Time,N+1),sol.value(xC))
hold on

grid minor
ylabel("Position x(t) [m]")
legend('x_1','x_2','safe','x_C')

subplot(3,1,2)
plot(linspace(0,Time,N+1),sol.value(v1))
hold on
plot(linspace(0,Time,N+1),sol.value(v2))
hold on 
plot(linspace(0,Time,N+1),sol.value(vC))
hold on 
% plot(linspace(0,Time,N+1),sol.value(v2))
% hold on 
% plot(linspace(0,Time,N+1),sol.value(speedC))
% hold on 
grid minor
ylabel("Speed v(t) [m/s]")
legend('v_1','v_2','v_C')
subplot(3,1,3)
plot(linspace(0,Time,N),sol.value(u1))
hold on
plot(linspace(0,Time,N),sol.value(u2))
hold on 
plot(linspace(0,Time,N),sol.value(uC))
hold on 
% plot(linspace(0,Time,N),sol.value(u2))
% hold on 
% plot(linspace(0,Time,N),sol.value(U_C))
% hold on 
grid minor
legend('u_1','u_2','u_C')
ylabel("Accel u(t) [m/s^2]")
xlabel("Time [sec]")


function x_next = runge_kutta4(f, x, u, dt)
% Runge-Kutta 4 integration
k1 = f(x,         u);
k2 = f(x+dt/2*k1, u);
k3 = f(x+dt/2*k2, u);
k4 = f(x+dt*k3,   u);
x_next = x + dt/6*(k1+2*k2+2*k3+k4);
end

