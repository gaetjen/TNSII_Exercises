% authors: Johannes Gätjen, Lorena Morton

stimA = rand(1, 100) > 1/3;
stimB = rand(1, 100) > 1/5;
reward = stimA;
stim = [stimA;stimB];
weights = zeros(size(stim));
errors = zeros(1, length(weights));
rate = 0.05;
for i = 1:length(weights) - 1
    errors(i) = reward(i) - weights(:, i)' * stim(:, i);
    weights(:, i+1) = weights(:, i) + stim(:, i) .* errors(i) * rate;
end
ShowSequence( stim, reward, errors, weights )