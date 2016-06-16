% authors: Johannes Gätjen, Lorena Morton

close all;

rate = 0.15;
beta = 5;
nTrials = 100;

instrewards = zeros(2, nTrials * 2);
action_values = zeros(2, nTrials * 2);
visit_prob = zeros(1, nTrials * 2) + 0.5;
rewards = normrnd(1, 1, [2, nTrials*2]);
rewards(2, 1:nTrials) = normrnd(3, 1, [1, nTrials]);
rewards(1, nTrials+1:end) = normrnd(3, 1, [1, nTrials]);
for i = 2:nTrials*2
    if rand < visit_prob(i-1)
        instrewards(1, i-1) = rewards(1, i);
        avreward = mean(sum(instrewards(:, max(1, i - 11):i-1)));
        error = instrewards(1, i-1) - avreward;
        action_values(1, i) = action_values(1, i-1) + rate * error * (1-visit_prob(i-1));
        action_values(2, i) = action_values(2, i-1) - rate * error * (1-visit_prob(i-1));
    else
        instrewards(2, i-1) = rewards(2, i);
        avreward = mean(sum(instrewards(:, max(1, i - 11):i-1)));
        error = instrewards(2, i - 1) - avreward;
        action_values(2, i) = action_values(2, i-1) + rate * error * visit_prob(i-1);
        action_values(1, i) = action_values(1, i-1) - rate * error * visit_prob(i-1);
    end
    visit_prob(i) = logsig(beta* 0.5 *(action_values(1, i) - action_values(2, i)));
end

totalrewards = cumsum(instrewards, 2);
totalrewards = [totalrewards; sum(totalrewards)];
fh = figure();
set(fh, 'Position', [100, 100, 800, 600])
ca = get(gca, 'ColorOrder');
ca(2, :) = [0.9, 0.7, 0];

subplot(2, 2, 1);
set(gca, 'ColorOrder', ca, 'NextPlot', 'replacechildren');
plot(action_values', 'LineWidth', 1.5);
ylabel('action values')
xlabel('trial');
legend('m_b', 'm_y');

subplot(2, 2, 2);
set(gca, 'ColorOrder', ca, 'NextPlot', 'replacechildren');
plot(totalrewards', 'LineWidth', 1.5);
ylabel('cumulative rewards')
xlabel('trial');
legend({'blue rewards', 'yellow rewards', 'total reward'}, 'Location', 'best');

subplot(2, 2, 4);
set(gca, 'ColorOrder', ca, 'NextPlot', 'replacechildren');
plot(instrewards', '.', 'LineWidth', 1.5);
ylabel('instantaneous rewards')
xlabel('trial');
legend('blue rewards', 'yellow rewards');

subplot(2, 2, 3);
set(gca, 'ColorOrder', ca, 'NextPlot', 'replacechildren');
plot([visit_prob',1-visit_prob'], 'LineWidth', 1.5);
ylabel('visit probability')
xlabel('trial');
legend('blue prob', 'yellow prob');