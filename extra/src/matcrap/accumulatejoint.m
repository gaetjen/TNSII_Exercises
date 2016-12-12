% stim = genStimuli(1, 10, 100000, 1);
% resp = response(1, stim, 1, 1);
close all;

numIter = 100;
conSamplesAll = sort(exprnd(1, 1, 500 * numIter));
conspace = logspace(log10(min(conSamplesAll)), log10(max(conSamplesAll)+0.1), 8);
conSelection = 1:numIter:(numIter*500);
orispace = linspace(0, 2*pi, 36);

for iter = 1:numIter
    iter
    conSamples = conSamplesAll(conSelection);
    conSelection = conSelection + 1;

    oriSamples = sort(rand(1, 500)*2*pi);

    stim = zeros(2, 500);
    resp = zeros(500, 500, 500);
    %densities = exppdf(conSamples, 1)';
    %densities = log(repmat(densities/(2*pi), 1, 500));

    for i = 1:500
        stim(2, :) = conSamples(i);
        for j = 1:500
            stim(1, :) = oriSamples(j);
            resp(i, j, :) = response(1, stim, 1, 1);
        end
    end
    if iter == 1
        respspace= logspace(log10(min(resp(:))), log10(max(resp(:))), 12);
        respHist = [-inf, respspace(2:end-1), inf];
        joint = zeros(length(conspace) - 1, length(orispace) - 1, length(respspace)-1);
    end

    
    for i = 2:length(conspace)
        conSelect = conSamples >= conHist(i - 1) & conSamples < conHist(i);
        for j = 2:length(orispace)
            oriSelect = oriSamples >= orispace(j - 1) & oriSamples < orispace(j); 
            selected = resp(conSelect, oriSelect, :);
            counts = histc(selected(:), respHist);
            joint(i-1, j-1, :) = counts(1:end-1) + squeeze(joint(i-1, j-1, :));
        end
    end
end

marginalCount = respspace(1:end-1) * 0;

for i = 1:length(respspace)-1
    marginalCount(i) = sum(sum(joint(:, :, i)));
end

[v, i] = max(joint(:));
[maxConI, maxOriI, maxXI] = ind2sub(size(joint), i)
maxes = [conspace(maxConI+1), orispace(maxOriI+1), respspace(maxXI+1)]

for i = 1:length(respspace)-1
    figure();
    h = surf(conspace(2:end), orispace(2:end), joint(:, :, i)');
    set(h, 'edgecolor', 'none');
    xlabel('contrast');
    set(gca, 'XScale', 'log')
end


% maxes =
% 
%     0.1231    0.3590   14.9255

% maxes =
% 
%     2.1565    3.0518    0.4908

% 100 iterations
% maxes =
% 
%     1.5765    0.7181   19.7892