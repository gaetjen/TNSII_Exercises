% stim = genStimuli(1, 10, 100000, 1);
% resp = response(1, stim, 1, 1);
close all;

conSamples = sort(exprnd(1, 1, 500));
oriSamples = sort(rand(1, 500)*2*pi);
stim = zeros(2, 500);
resp = zeros(500, 500, 500);
densities = exppdf(conSamples, 1)';
densities = repmat(densities/(2*pi), 1, 500);
for i = 1:500
    stim(2, :) = conSamples(i);
    for j = 1:500
        stim(1, :) = oriSamples(j);
        resp(i, j, :) = response(1, stim, 1, 1);
    end
    if mod(i, 10) == 0
        i
    end
end
%%%%%%%%%%%%%%%
conspace = logspace(log10(min(conSamples)), log10(max(conSamples)), 36);
orispace = linspace(0, 2*pi, 36);
respspace= logspace(log10(min(resp(:))), log10(max(resp(:))), 24);

joint = zeros(500, 500, length(respspace));
for i = 1:500
    for j = 1:500
        respDensity = histc(resp(i, j, :), respspace);
        respDensity = respDensity / trapz(respspace, respDensity); % make histogram into density function with area 1
        joint(i, j, :) = respDensity * densities(i, j);
    end
    if mod(i, 40) == 0
        i
    end
end

discreteProb = zeros(length(conspace)-1, length(orispace)-1, length(respspace));
for i = 2:length(conspace)
    for j = 2:length(orispace)
        subspace = joint(conSamples < conspace(i) & conSamples >= conspace(i-1), oriSamples < orispace(j) & oriSamples >= orispace(j-1), :);
        sum1 = sum(subspace, 1);
        discreteProb(i, j, :) = sum(sum1, 2);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i = 1:15 
%     figure();
%     h = surf(conspace, orispace, discreteProb(:, :, i)');
%     set(h, 'edgecolor', 'none');
%     xlabel('contrast');
%     set(gca, 'XScale', 'log')
% end

[v, i] = max(joint(:));
[maxConI, maxOriI, maxXI] = ind2sub(size(joint), i)
maxes = [conSamples(maxConI), oriSamples(maxOriI), respspace(maxXI)]

[v, i] = max(discreteProb(:));
[maxConI, maxOriI, maxXI] = ind2sub(size(discreteProb), i)
maxes = [conspace(maxConI), orispace(maxOriI), respspace(maxXI)]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% is this OK? do I have to normalize something first?
marginalRespProb = respspace * 0;

for i = 1:length(respspace)
    marginalRespProb(i) = sum(sum(joint(:, :, i)));
end

marginalRespProb = marginalRespProb / trapz(respspace, marginalRespProb);
semilogx(respspace, marginalRespProb);
title('marginal response probability');


conditionals = discreteProb * 0;
for i = 1:length(conspace)
    for j = 1:length(orispace)
        for k = 1:length(respspace)-1
            conditionals(i, j, k) = discreteProb(i, j, k) / marginalRespProb(k);
        end
    end
end


post = conditionals(:, :, 1) * 0;
for c = 1:36
    act = r(c, 3);
    idx = find(respspace>act, 1);
    post = post + circshift(conditionals(:, :, idx), [0, c-1]);
end


