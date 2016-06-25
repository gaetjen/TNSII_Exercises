% authors: Johannes Gätjen, Lorena Morton
close all;
load 'MUA_b_t_g.mat'
avActivity = squeeze(mean(MUA_b_t_g, 1))';

plot(ti, avActivity);
xlabel('time');
ylabel('population activity');

rowMeans = mean(avActivity, 2);
activityZeroMean = avActivity - repmat(rowMeans, 1, Nt);
activityCov = (1 / (Nt - 1)) * (activityZeroMean * activityZeroMean');

figure();
pcolor(activityCov);
axis equal;
axis off;

[U, S, V] = svd(activityZeroMean');

figure();
bar(diag(S));   % 3 large principal components
xlabel('principal components');
ylabel('PC standard deviation');

% end part 3
% projection
transformed = V' * activityZeroMean;

figure();
plot(ti, transformed);
xlabel('time');
ylabel('PC magnitude');

transCov = transformed * transformed' / (Nt - 1);

figure();
pcolor(transCov);
axis equal;
axis off;

figure();
plot3(transformed(1, :), transformed(2, :), transformed(3, :), '.');hold on;
plot3(transformed(1, 1), transformed(2, 2), transformed(3, 3), '*r', 'Markersize', 15);
xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');

transformed(4:end, :) = 0;
backProjection = V' \ transformed;  % B \ A == inv(B) * A
backProjection = backProjection + repmat(rowMeans, 1, Nt);

figure();
plot(ti, backProjection);
xlabel('time');
ylabel('population activity');

% end part 4

fullTransformed = zeros(Nb, Nt, Ng);
figure();
for i = 1:Nb
    fullTransformed(i, :, :) = (V' * squeeze(MUA_b_t_g(i, :, :))')';
    plot3(fullTransformed(i, :, 1), fullTransformed(i, :, 2), fullTransformed(i, :, 3), '.--');
    plot3(fullTransformed(i, 1, 1), fullTransformed(i, 1, 2), fullTransformed(i, 1, 3), '*r', 'Markersize', 15);
    hold on;
end
xlabel('PC 1');
ylabel('PC 2');
zlabel('PC 3');

