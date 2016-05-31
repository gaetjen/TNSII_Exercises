% authors: Johannes Gätjen, Lorena Morton
close all;
numTrials = 1000;
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
ShowSequence( stim(:, 1:100), reward(:, 1:100), errors(:, 1:100), weights(:, 1:100))
steadyState = mean(weights(:, end-25:end), 2)

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
ShowSequence( stim(:, 1:100), reward(:, 1:100), errors(:, 1:100), weights(:, 1:100))
steadyState = mean(weights(:, end-25:end), 2)

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
ShowSequence( stim(:, 1:100), reward(:, 1:100), errors(:, 1:100), weights(:, 1:100))
steadyState = mean(weights(:, end-25:end), 2)

% steadyState =
% 
%     1.0000
%     0.0001
% 
% 
% steadyState =
% 
%     0.5349
%     0.0696
% 
% 
% steadyState =
% 
%     0.8550
%    -0.2418
