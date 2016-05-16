close all;
occurences = [1 1; 1 0; 0 1; 0 0];
gamma = 0.5;
stats = [gamma/4, 0.5 - gamma / 4, 0.5 - gamma / 4, gamma / 4];
ranges = cumsum(stats);
samples = 10000;
randoms = rand(samples, 1);
idxs = repmat(randoms, 1, 4) < repmat(ranges, samples, 1);
inputs = zeros(samples, 2);
for i = 1:samples
    f = find(idxs(i, :), 1);
    inputs(i, :) = occurences(f, :);
end
psides = sum(inputs) / samples;
cs = psides - psides .^2;
inputProd = inputs(:, 1) .* inputs(:, 2);
cd = (sum(inputProd)/samples) - (psides(1) * psides(2));
C = [cs(1), cd; cd, cs(2)];
ShowEigen(C);

dt = 0.1;
maxt = 100;
time = 0:dt:maxt;
w = repmat(rand(1, 2) * 0.5, length(time), 1);
for i = 1:length(time) - 1
    w(i + 1, :) = max(w(i, :) + (C * w(i, :)')' * dt, 0);
    w(i + 1, :) = min(w(i + 1, :), 1);
end
figure;
plot(w(:, 1), w(:, 2));
xlim([0 1]);
ylim([0 1]);


for i = 1:length(time) - 1
    dw = (C * w(i, :)') - 0.5 * (w(i, :) * C * w(i, :)') * w(i, :)';
    w(i + 1, :) = max(w(i, :) +  dw' * dt, 0);
    w(i + 1, :) = min(w(i + 1, :), 1);
end
figure;
plot(w(:, 1), w(:, 2));
xlim([0 1]);
ylim([0 1]);
 