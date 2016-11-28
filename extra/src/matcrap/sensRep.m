function [ means ] = sensRep( preferred, stimuli, tuningWidth, sensitivity )
%SENSREP calculate sensory representation of stimuli
%   preferred: preferred stimulus orientation of different channels
%   stimuli: 2×t array of stimuli, orientation (top) and contrast (bottom)
contrast = repmat(stimuli(2, :), length(preferred), 1);
orientation = repmat(stimuli(1, :), length(preferred), 1);
preferred = repmat(preferred, size(stimuli, 2), 1)';
means = sensitivity * contrast .* exp(2 * cos(tuningWidth*(preferred - orientation))); % circular gaussian tuning

end

