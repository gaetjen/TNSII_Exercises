function [M, E1, E2, lambda1, lambda2] = ConnectionMatrix( N )

%% Input args

% N size of matrix

% lambda degenerate eignevalues

%% Output args

% M connection matrix

% E1, lambda1 first principal eigenvector and eigenvalue

% E2, lambda2 second principal eignvector and eigenvalue

lambda = 0.8;

theta = linspace(-pi,pi,N+1);

[X,Y] =meshgrid(theta(1:end-1));

M = lambda * cos(X-Y) / (pi*pi);

[V,D] = eig(M);

E1 = V(:,end);
E2 = V(:,end-1);

lambda1 = D(end,end);
lambda2 = D(end-1,end-1);


return;