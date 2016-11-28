% stim = genStimuli(1, 10, 100000, 1);
% resp = response(1, stim, 1, 1);
close all;

conSamples = sort(exprnd(1, 1, 500));
conspace = logspace(log10(min(conSamples)), log10(max(conSamples)+0.1), 8);
conHist = [-inf, conspace(2:end-1), inf];
conProb = histc(conSamples, conHist);
conProb = conProb / 500;

oriSamples = sort(rand(1, 500)*2*pi);
orispace = linspace(0, 2*pi, 36);
oriProb = histc(oriSamples, orispace);
oriProb = oriProb / 500;

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
    if mod(i, 10) == 0
        i
    end
end

respspace= logspace(log10(min(resp(:))), log10(max(resp(:))), 12);
respHist = [-inf, respspace(2:end-1), inf];


joint = zeros(length(conspace) - 1, length(orispace) - 1, length(respspace)-1);

for i = 2:length(conspace)
    conSelect = conSamples >= conHist(i - 1) & conSamples < conHist(i);
    for j = 2:length(orispace)
        oriSelect = oriSamples >= orispace(j - 1) & oriSamples < orispace(j); 
        selected = resp(conSelect, oriSelect, :);
        counts = histc(selected(:), respHist);
        joint(i-1, j-1, :) = counts(1:end-1);
    end
end
joint = joint + 1e-6;

% for i = 1:length(respspace)-1
%     figure();
%     h = surf(conspace(2:end), orispace(2:end), joint(:, :, i)');
%     set(h, 'edgecolor', 'none');
%     xlabel('contrast');
%     set(gca, 'XScale', 'log')
% end

marginalCount = respspace(1:end-1) * 0;

for i = 1:length(respspace)-1
    marginalCount(i) = sum(sum(joint(:, :, i)));
end

figure();
semilogx(respspace(2:end), marginalCount);

[v, i] = max(joint(:));
[maxConI, maxOriI, maxXI] = ind2sub(size(joint), i)
maxes = [conspace(maxConI+1), orispace(maxOriI+1), respspace(maxXI+1)]


conditionals = joint * 0;
for i = 1:length(conspace)-1
    for j = 1:length(orispace)-1
        for k = 1:length(respspace)-1
            conditionals(i, j, k) = joint(i, j, k) / marginalCount(k);
        end
    end
end

conditionals = log(conditionals);

stim = genStimuli(1, 2, 1000, 1);
res = response(length(orispace) - 1, stim, 1, 1);
error = zeros(2, 40);
for size = 1:40
    size
    integr = windowIntegration(conditionals, res, respHist, size);
    detected = infer(integr, conspace, orispace);
    d = abs(stim - detected(2:3, :));
    error(:, size) = sum(d, 2);
end
figure();
plot(error')



% plot(abs(d'));
% figure();
% stem(diff(stim(1, :)));