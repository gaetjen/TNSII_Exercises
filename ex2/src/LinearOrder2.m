function LinearOrder2

% A is the interaction matrix size(2,2)

% B is the constant input vector size(2,1)

% X0 is the initial conditions of trajectory

% statesize is the size of the illustrated part of statespace

% tend is final time of trajectory evaluation (in multiples of tau)

%% eexample input

statesize = 10;

X0 = [6 6]';

tend = 2.5;

A = [-2 3; -3 2 ];

B = [-10 10]';

%%

close all;
fs = 14;    % fontsize for legends

%% solve for steady-state

% steady-state condition is 0 = dv/dt = A dot Xss + B or, equivalently, Xss = A^(-1) dot (-B)


Xss = A \ (-B);

%% obtain eigenvectors and eigenvalues

[V, D] = eig( A );

%% alternative variable names
a11=A(1,1);
a12=A(1,2);
a21=A(2,1);
a22=A(2,2);

b1 = B(1);
b2 = B(2);

xss = Xss(1);
yss = Xss(2);

lambda1  = D(1,1)  % eigenvalues
lambda2  = D(2,2)

E1  = V(:,1)       % eigenvectors
E2  = V(:,2)


%% x-range over which nullclines are evaluated (choose symmetrically around initial condition)

xmin = floor(xss - 0.5*statesize);
xmax = ceil(xss + 0.5*statesize);

%% nullclines dot x = 0 and dot y = 0

xi = linspace(xmin, xmax, 100);
ydx0i = -(a11*xi +b1)/a12;
ydy0i = -(a21*xi +b2)/a22;

%% resulting y-range (choose samre range as for x, to ensure uniform plotting)
ymin = floor(min([ydx0i ydy0i]));
ymax = ceil(max([ydx0i ydy0i]));

%% gradient vectors [dot x, dot y]'

nv = 10;

xv = linspace(xmin, xmax, nv );
yv = linspace(ymin, ymax, nv );

[XV, YV] = meshgrid( xv, yv );

DXV = ( a11*XV + a12*YV + b1);
DYV = ( a21*XV + a22*YV + b2);


%% plot state space
figure;

hold on;

plot( xi, ydx0i, 'r', 'LineWidth', 2);   % nullclines
plot( xi, ydy0i, 'b', 'LineWidth', 2);

h=legend( 'dx/dt=0', 'dy/dt=0', 'X(t)');
set(h,'FontSize', fs );
 
quiver( XV, YV, DXV, DYV, 'k', 'LineWidth', 1 );   % gradients

plot( xss, yss, 'k+', 'MarkerSize', 10, 'LineWidth', 2 );   % steady-state
plot( 0, 0, 'ko', 'MarkerSize', 5, 'LineWidth', 1 );   % coordinate origin

plot( [xmin xmax], [0 0], 'k:', 'LineWidth', 2);    % coordinate axes
plot( [0 0], [ymin ymax], 'k:', 'LineWidth', 2);

hold off;
axis 'square';
axis([xmin xmax ymin ymax]);
xlabel( 'x', 'FontSize', fs );
ylabel( 'y', 'FontSize', fs );

print 'Fig1_state_space' -depsc2;

%% initial conditions

X0 

Xss

C  = V \ (X0 - Xss);  % obtain coefficients from X0 = E1*c1 + E2*c2 + Xxx

c1 = C(1)
c2 = C(2)

%% trajectory computation
tau = 1;
tmin = 0;
tmax = tend;
nt   =100;
ti = linspace(tmin,tmax,nt);

Xi = nan(2,nt);

Xi(1,:) = real( c1*E1(1)*exp(lambda1*ti) + c2*E2(1)*exp(lambda2*ti) + Xss(1) );  
Xi(2,:) = real( c1*E1(2)*exp(lambda1*ti) + c2*E2(2)*exp(lambda2*ti) + Xss(2) );




%% plot time-evolution

figure;

hold on;

plot( ti, Xi(1,:), 'r', 'LineWidth', 2);   % x coordinate
plot( ti, Xi(2,:), 'b', 'LineWidth', 2);   % y coordinate
plot( [tmin tmax], [xss xss], 'r:', 'LineWidth', 1 );   % x steady-state
plot( [tmin tmax], [yss yss], 'b--', 'LineWidth', 1 );   % y steady-state

h=legend( 'x(1)', 'y(1)', 'x_{ss}', 'y_{ss}');
set(h,'FontSize', fs );
 

plot( [tmin tmax], [0 0], 'k:', 'LineWidth', 2);   % coordinate origin

hold off;
axis 'square';
axis([tmin tmax min([xmin ymin]) max([xmax ymax]) ]);
xlabel( 't/\tau', 'FontSize', fs );
ylabel( 'x, y', 'FontSize', fs );

print 'Fig2_time_evolution' -depsc2;



%% plot eigenvectors and eigenvalues

figure;

if ~isreal(E1(1)) 
    E1 = j * E1;
end

if ~isreal(E2(1)) 
    E2 = j * E2;
end


cmin = min( [real([E1' E2' lambda1 lambda2])  imag([E1' E2' lambda1 lambda2]) ] );
cmax = max( [real([E1' E2' lambda1 lambda2])  imag([E1' E2' lambda1 lambda2]) ] );

hold on;

quiver3( 0, 0, 0, E1(1), real(E1(2)), imag(E1(2)), 0, 'r', 'LineWidth', 2);  % E1
quiver3( 0, 0, 0, E2(1), real(E2(2)), imag(E2(2)), 0, 'b', 'LineWidth', 2);  % E2
plot3( real(lambda1), 0, imag(lambda1), 'ro', 'MarkerSize', 10 );   % lambda 1
plot3( 0, real(lambda2), imag(lambda2), 'bo', 'MarkerSize', 10 );   % lambda 2
h=legend( 'e_1', 'e_2', '\lambda_1', '\lambda_2');
set(h,'FontSize', fs );

plot3( [0 real(lambda1)], [0 0], [0 0], 'r-', 'LineWidth', 1 );   % lambda 1
plot3( [real(lambda1) real(lambda1)], [0 0], [0 imag(lambda1)], 'r-', 'LineWidth', 1 );   % lambda 1
plot3( [0 0], [0 real(lambda1)], [0 0], 'b-', 'LineWidth', 1 );   % lambda 2
plot3( [0 0], [real(lambda2) real(lambda2)], [0 imag(lambda2)], 'b-', 'LineWidth', 1 );   % lambda 2
hold off;
axis 'square';
axis([cmin cmax cmin cmax cmin cmax]);
xlabel( 'x, \lambda_1', 'FontSize', fs );
ylabel( 'y, \lambda_2', 'FontSize', fs );
zlabel( 'imag', 'FontSize', fs );

print 'Fig3_eigenvectors' -depsc2;



%% plot trajectories in state-space

figure;

subplot(1,2,1);

hold on;

plot( xi, ydx0i, 'r', 'LineWidth', 2);   % nullclines
plot( xi, ydy0i, 'b', 'LineWidth', 2);
plot( Xi(1,:), Xi(2,:), 'g', 'LineWidth', 4 );      % trajectory
h=legend( 'dx/dt=0', 'dy/dt=0', 'X(t)');
set(h,'FontSize', fs );
 
quiver( XV, YV, DXV, DYV, 'k', 'LineWidth', 1 );   % gradients

plot( xss, yss, 'k+', 'MarkerSize', 10, 'LineWidth', 2 );   % steady-state
plot( 0, 0, 'ko', 'MarkerSize', 5, 'LineWidth', 1 );   % coordinate origin

plot( [xmin xmax], [0 0], 'k:', 'LineWidth', 2);    % coordinate axes
plot( [0 0], [ymin ymax], 'k:', 'LineWidth', 2);


plot( X0(1), X0(2), 'g+', 'MarkerSize', 10, 'LineWidth', 2 ); % initial condition


hold off;
axis 'square';
axis([xmin xmax ymin ymax]);
xlabel( 'x', 'FontSize', fs );
ylabel( 'y', 'FontSize', fs );

subplot(1,2,2);

hold on;

plot( ti, Xi(1,:), 'r', 'LineWidth', 2);   % x coordinate
plot( ti, Xi(2,:), 'b', 'LineWidth', 2);   % y coordinate
plot( [tmin tmax], [xss xss], 'r--', 'LineWidth', 1 );   % x steady-state
plot( [tmin tmax], [yss yss], 'b--', 'LineWidth', 1 );   % y steady-state

h=legend( 'x(1)', 'y(1)', 'x_{ss}', 'y_{ss}');
set(h,'FontSize', fs );
 

plot( [tmin tmax], [0 0], 'k:', 'LineWidth', 2);   % coordinate origin

hold off;
axis 'square';
axis([tmin tmax min([xmin ymin]) max([xmax ymax]) ]);
xlabel( 't/\tau', 'FontSize', fs );
ylabel( 'x, y', 'FontSize', fs );

print 'Fig4_together' -depsc2;


return;

