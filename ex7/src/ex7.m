% authors: Johannes Gätjen, Lorena Morton
load observations.mat
close all;
x = linspace(-400, 400, 100);
dbin = x(2) - x(1);
hist(sample, x); hold on;
h = get(gca, 'children');

mus = [-55, -100, 100];
sigmas = [40, 90, 35];
gammas = [0.55, 0.2, 0.25];

dists = zeros(3, length(x));
% plot hand estimated
for i = 1:3
    dists(i, :) = gammas(i) * length(sample) * normpdf(x, mus(i), sigmas(i)) * dbin;
end
s = plot(x, dists, 'r', 'LineWidth', 1.5);
t = plot(x, sum(dists), 'g', 'LineWidth', 1.5);
ylabel('sample density (absolute)');
legend([h, s(1), t], {'samples histogram', 'individual distributions', 'sum of distributions'})
%do EM
for i = 1:200
    ex = Expect(sample, gammas, mus, sigmas);
    [gammas, mus, sigmas] = Maximize( sample, ex );
end

% plot em results
figure();
hist(sample, x); hold on;
h = get(gca, 'children');
for i = 1:3
    dists(i, :) = gammas(i) * length(sample) * normpdf(x, mus(i), sigmas(i)) * dbin;
end
s = plot(x, dists, 'r', 'LineWidth', 1.5);
t = plot(x, sum(dists), 'g', 'LineWidth', 1.5);
ylabel('sample density (absolute)');
legend([h, s(1), t], {'samples histogram', 'individual distributions', 'sum of distributions'})