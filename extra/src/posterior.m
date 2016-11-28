function [ postLog ] = posterior( logCond, activities, respHist )
%POSTERIOR calculate posterior probability of orientation and contrast,
%given channel activities, accumulated over time
%
%   activities: m × n matrix, m channels, n stimuli, assumes evenly spaced
%   preferred channel orientation
%   logCond:    log likelihood of orientation and contrast, given activity
%   respHist:   the borders for the activities

postLog = logCond(:, :, 1) * 0;
[numChan, numStim] = size(activities);
chanActSpace = linspace(0, 2*pi, numChan+1);
chanActSpace = chanActSpace(1:end-1);
chanCondSpace = linspace(0, 2*pi, size(logCond, 2)+1);
chanCondSpace = chanCondSpace(1:end-1);
for c = 1:numChan
    idxC = find(chanActSpace(c) <= chanCondSpace, 1);
    for s = 1:numStim
        act = activities(c, s);
        idxA = find(respHist>=act, 1);
        postLog = postLog + circshift(logCond(:, :, idxA-1), [0, idxC-1]);
    end
end

end

