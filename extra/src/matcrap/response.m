function [ responses ] = response( numChannels, stimuli, tuningWidth, sensitivity )
%RESPONSE responses of channels to stimuli
%   numChannels: number of channels
%   stimuli:     stimuli over time, comprising orientation (top) and
%   contrast (bottom)
%   tuningWidth: tuning width for orientation
%   sensitivity: channels' sensitivity
preferred = linspace(0, 2*pi, numChannels + 1);
preferred = preferred(1:end-1);
means = sensRep(preferred, stimuli, tuningWidth, sensitivity);
responses = raylrnd(means / sqrt(pi/2));
end

