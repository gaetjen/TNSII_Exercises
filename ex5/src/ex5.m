% authors: Johannes Gätjen, Lorena Morton
close all;
numTrials = 100;
stimA = rand(1, numTrials) < 1/3;
stimB = rand(1, numTrials) < 1/5;
reward = stimA;
stim = [stimA;stimB];
weights = zeros(size(stim));
errors = zeros(1, length(weights));
rate = 0.05;
for i = 1:length(weights) - 1
    errors(i) = reward(i) - weights(:, i)' * stim(:, i);
    weights(:, i+1) = weights(:, i) + stim(:, i) .* errors(i) * rate;
end
figure()
ShowSequence( stim, reward, errors, weights )

stimA = rand(1, numTrials) < 1/3;
stimB = rand(1, numTrials) < 1/5;
reward = rand(1, numTrials) < stimA/2;
stim = [stimA;stimB];
weights = zeros(size(stim));
errors = zeros(1, length(weights));
rate = 0.05;
for i = 1:length(weights) - 1
    errors(i) = reward(i) - weights(:, i)' * stim(:, i);
    weights(:, i+1) = weights(:, i) + stim(:, i) .* errors(i) * rate;
end
figure()
ShowSequence( stim, reward, errors, weights )

stimA = rand(1, numTrials) < 1/3;
stimB = rand(1, numTrials) < 1/5;
reward = stimA & ~stimB;
stim = [stimA;stimB];
weights = zeros(size(stim));
errors = zeros(1, length(weights));
rate = 0.05;
for i = 1:length(weights) - 1
    errors(i) = reward(i) - weights(:, i)' * stim(:, i);
    weights(:, i+1) = weights(:, i) + stim(:, i) .* errors(i) * rate;
end
figure()
ShowSequence( stim, reward, errors, weights )