% authors: Johannes Gätjen, Lorena Morton
function [gammas, mus, sigmas] = Maximize( samples, probs )
gammas = mean(probs, 2);
mus = mean(probs .* repmat(samples, length(gammas), 1), 2) ./ gammas;
sigmas = gammas * 0;
for i = 1:length(sigmas)
    sigmas(i) = sqrt(mean(probs(i, :) .* ((samples - mus(i)) .^ 2)) / gammas(i));
end
end

