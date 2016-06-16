% authors: Johannes Gätjen, Lorena Morton

close all;
betas = 0.3:1:25;
% rates = logspace(-2, 0, 20);
rates = linspace(0.01, 1, 20);
[bs, rs] = meshgrid(betas, rates);
avrewards = zeros(size(bs));
numrep = 50;
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
            rewardgrid(r, b) = sum(sum(instrewards));
        end
    end
    avrewards = avrewards + rewardgrid / numrep;
end
figure();
%pcolor(rs, bs, avrewards);
surf(rs, bs, avrewards);
xlabel('learning rate');
ylabel('beta')
%zlabel('total reward');

[m, i] = max(avrewards(:));
avrewards(i)
[r, b] = ind2sub(size(avrewards), i);
display('rate');
rates(r)
display('beta');
betas(b)