% authors: Johannes Gätjen, Lorena Morton

close all;
% betas = 0.3:1:25;
% rates = logspace(-2, 0, 20);
% rates = linspace(0.01, 1, 20);
betas = [1];
rates = [0.1];
[bs, rs] = meshgrid(betas, rates);
avrewards = zeros(size(bs));
numrep = 1;
for rep = 1:numrep
    display('nxt iter')
    rewardgrid = zeros(size(bs));

    for r = 1:length(rates)
        rate = rates(r);
        for b = 1:length(betas)
            beta = betas(b);
            nTrials = 150;
            instrewards = zeros(2, nTrials * 2);
            action_values = zeros(2, nTrials * 2);
            visit_prob = zeros(1, nTrials * 2) + 0.5;
            rewards = normrnd(1, 1, [2, nTrials*2]);
            rewards(2, 1:nTrials) = normrnd(3, 1, [1, nTrials]);
            rewards(1, nTrials+1:end) = normrnd(3, 1, [1, nTrials]);
            for i = 2:nTrials*2
                if rand < visit_prob(i-1)
                    instrewards(1, i-1) = rewards(1, i);
                    error = instrewards(1, i-1) - action_values(1, i-1);
                    action_values(1, i) = action_values(1, i-1) + rate * error;
                    action_values(2, i) = action_values(2, i-1);
                else
                    instrewards(2, i-1) = rewards(2, i);
                    error = instrewards(2, i - 1) - action_values(2, i-1);
                    action_values(2, i) = action_values(2, i-1) + rate * error;
                    action_values(1, i) = action_values(1, i-1);
                end
                visit_prob(i) = logsig(beta* 0.5 *(action_values(1, i) - action_values(2, i)));
            end
            rewardgrid(r, b) = sum(sum(instrewards));
        end
    end
    avrewards = avrewards + rewardgrid / numrep;
end
% figure();
% pcolor(rs, bs, avrewards);
% xlabel('learning rate');
% ylabel('beta')
%zlabel('total reward');

[m, i] = max(avrewards(:));
avrewards(i)
[r, b] = ind2sub(size(avrewards), i);
display('rate');
rates(r)
display('beta');
betas(b)


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