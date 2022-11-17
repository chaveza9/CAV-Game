function [x_2_f,v_2_f,x2,v2,u2] = solve_hdv_ibrproblem (x1,tf,x_2_0,v_2_0,N,xC)
 %% Define Phyisical Constraints
u_max = 3.3;    % Vehicle i max acceleration [m/s^2]
u_min = -7;   % Vehicle i min acceleration [m/s^2]
v_min = 15;   % Vehicle i min velocity [m/s]
v_max = 31;   % Vehicle i max velocity [m/s]
%% Tunning Variables
phi = 0.6; % [seconds] Intervehicle Reaction Time
delta = 1.5; % [meters] intervehicle distance
v_des = 30;
import casadi.*
% weights
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
gamma_safety = 1;

%%  Numerical Solution
opti = casadi.Opti(); % Optimization problem

% ---- decision variables ---------
X = opti.variable(2,N+1); % state trajectory
pos   = X(1,:);
speed = X(2,:);
U = opti.variable(1,N);   % control trajectory (throttle)
%t1 = opti.variable();

% Define dynamic constraints
c = @(u, x) u^2; % Cost Acceleration
f = @(x,u) [x(2);u]; % dx/dt = f(x,u)
l = @(u, x) u;
% Define objective funciton integrator
% Integrate using RK4
dt = tf/N;
cost_u = 0;
cost_v = 0;
cost_safety = 0;

for k=1:N % loop over control intervals
    % Forward integration
    x_next = runge_kutta4(f, X(:,k), U(:,k), dt);
    cost_u = cost_u + 0.5*gamma_energy*runge_kutta4(c, U(:,k), X(:,k), dt);
    cost_v = cost_v + gamma_speed*runge_kutta4(l,(speed(:,k)-v_des)^2,X(:,k),dt);
    bin = if_else(pos(k)<=xC(k),1, 0);
    cost_safety = bin*(cost_safety + gamma_safety*runge_kutta4(l,pos(:,k)+speed(:,k)*phi+delta-xC(:,k),X(:,k),dt));
    
    % Impose multi-shoot constraint
    opti.subject_to(X(:,k+1)==x_next); % close the gaps
    % safety constraint
    opti.subject_to(x1(k)-pos(k)>=speed(k)*phi+delta);
end
cost = cost_u + cost_v + cost_safety;
% ----- Objective function ---------
opti.minimize(cost);
% --------- Define path constraints ---------
opti.subject_to(v_min<=speed<=v_max);     %#ok<CHAIN> % track speed limit
opti.subject_to(u_min<=U<=u_max);     %#ok<CHAIN> % control is limited
% --------- Define Boundary conditions ---------
opti.subject_to(pos(1)==x_2_0);   % start at position 0 ...
opti.subject_to(speed(1)==v_2_0); % initial speed
%opti.subject_to(x_C_f - pos(end) >= phi*speed(end) + delta)
% Warm Start solver
% opti.set_initial(speed, 25);
opti.set_initial(U, -2);

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
x_2_f = sol.value(pos(end));
v_2_f = sol.value(speed(end));
x2 = sol.value(pos);
v2 = sol.value(speed);
u2 = sol.value(U);

function x_next = runge_kutta4(f, x, u, dt)
% Runge-Kutta 4 integration
k1 = f(x,         u);
k2 = f(x+dt/2*k1, u);
k3 = f(x+dt/2*k2, u);
k4 = f(x+dt*k3,   u);
x_next = x + dt/6*(k1+2*k2+2*k3+k4);
end
end