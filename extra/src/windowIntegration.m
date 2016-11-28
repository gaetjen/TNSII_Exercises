function [ integrated ] = windowIntegration( cond, act, respHist, windowSize )
%WINDOWINTEGRATION calculate log-posterior for sequence of responses, given
% an integration window size and conditional probabilities
integrated = zeros(size(cond, 1), size(cond, 2), size(act, 2));
for i = 1:size(act, 2);
    lower = max(1, i-windowSize);
    upper = i;
    acts = act(:, lower:upper);
    integrated(:, :, i) = posterior(cond, acts, respHist);
end

end

