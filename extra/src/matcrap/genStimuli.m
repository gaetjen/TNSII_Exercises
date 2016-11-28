function [ stimuli ] = genStimuli( contrast, duration, time, deltaT )
%GENSTIMULI generate sequence of stimuli
%   contrast: mean contrast
%   duration: mean duration
%   time:     total time of stimuli to generate
%   deltaT:   time resolution (step size)
%   stimuli:  orientation (top row) and contrast (bottom row) over time
stimuli = zeros(2, time/deltaT);
t = 1;
while t <= time
    d = exprnd(duration);
    c = exprnd(contrast);
    o = rand * 2 * pi;
    tidx = round(t);
    stimuli(1, tidx:end) = o;
    stimuli(2, tidx:end) = c;
    t = t + (d/deltaT);
end
end

