function X = Factivation( X, Fmax, kappa )


kk = find( X < 0 );
X = Fmax * X.^2 ./ ( kappa*kappa + X.^2 );
X(kk) = zeros(size(kk));

return;