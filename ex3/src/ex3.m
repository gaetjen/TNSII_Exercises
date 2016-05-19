%authors: Johannes Gätjen, Lorena Morton
close all;
occurences = [1 1; 1 0; 0 1; 0 0];
gamma =1.5;
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
C = [cs(1), cd; cd, cs(2)]
ShowEigen(C);

dt = 0.1;
maxt = 100;
time = 0:dt:maxt;
w = repmat(rand(1, 2, 30) * 0.5, length(time), 1);
for i = 1:length(time) - 1
    w(i + 1, :, :) = max(squeeze(w(i, :, :)) + C * squeeze(w(i, :, :)) * dt, 0);
    w(i + 1, :, :) = min(w(i + 1, :, :), 1);
end
figure;
plot(squeeze(w(:,1,:)), squeeze(w(:, 2, :)), 'b', 'LineWidth', 1);
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 2);
axis equal;
xlim([0 1]);
ylim([0 1]);
xlabel('w_L');
ylabel('w_R');
legend('development of weights');

for i = 1:length(time) - 1
    sw = squeeze(w(i, :, :));
    normFactor = diag((sw' * C * sw));    
    dw = (C * sw) - 0.5 * [normFactor, normFactor]' .* sw;
    w(i + 1, :, :) = max(sw +  dw * dt, 0);
    w(i + 1, :, :) = min(w(i + 1, :, :), 1);
end
figure;
plot(squeeze(w(:,1,:)), squeeze(w(:, 2, :)), 'b', 'LineWidth', 1); hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 2);
axis equal;
xlim([0 1]);
ylim([0 1]);
xlabel('w_L');
ylabel('w_R');
legend('development of weights');
 