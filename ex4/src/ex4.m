%authors: Johannes Gätjen, Lorena Morton
% comment and uncomment according to plotting wishes
% kappas = 0.01:0.01:2;
% rmse = zeros(size(kappas));
% sigmas = 0.01:0.5:250;
% rmse = zeros(size(sigmas));
numN = 35:200;
rmse = zeros(size(numN));
for i = 1:length(rmse)
    close all;
    rMax = 1;
%     kappa = kappas(i);
%     sigma = 30;
%     upstream = 40;
%     kappa = 0.22;
%     sigma = sigmas(i);
%     upstream = 40;
    kappa = 0.22;
    sigma = 30;
    upstream = numN(i);
    nuPref = linspace(0, 1000, upstream);
    inputs = rand(1, 10000) * 1000; % average density of 10 per Hz
    [linMean, linNoisy] = GaussResp_LinearSTD(inputs, nuPref, rMax, kappa);
    [consMean, consNoisy] = GaussResp_ConstantSTD(inputs, nuPref, rMax, sigma);

%     plot(inputs, linMean(:, [3, 10, 20, 35]), '.');
%     xlabel('input frequency [Hz]');
%     ylabel('neuron response');
%     figure();
%     plot(inputs, linNoisy(:, [3, 10, 20, 35]), '.');
%     xlabel('input frequency [Hz]');
%     ylabel('neuron response');
%     figure();
%     plot(inputs, consMean(:, [3, 10, 20, 35]), '.');
%     xlabel('input frequency [Hz]');
%     ylabel('neuron response');
%     figure();
%     plot(inputs, consNoisy(:, [3, 10, 20, 35]), '.');
%     xlabel('input frequency [Hz]');
%     ylabel('neuron response');

    cov = covariance(linMean, consMean);
    
%     figure('Position', [200, 200, 900, 450]);
%     subplot(1, 2, 1);
%     h = pcolor(cov);
%     set(h, 'Edgecolor', 'none');
%     colorbar('northoutside');
%     xlabel('linear population');
%     ylabel('constant population');
    
    cov = covariance(linNoisy, consNoisy);
    
%     subplot(1, 2, 2);
%     h = pcolor(cov);
%     set(h, 'Edgecolor', 'none');
%     colorbar('northoutside');
%     xlabel('linear population');
%     ylabel('constant population');

    inputs = rand(1, 10000) * 1000;
    [linMean, linNoisy] = GaussResp_LinearSTD(inputs, nuPref, rMax, kappa);
    consFfwd = max(cov * linNoisy', 0);
    nuInf = sum(consFfwd' .* repmat(nuPref, size(consFfwd, 2), 1), 2) ./ sum(consFfwd', 2);
    rmse(i) = sqrt(mean((inputs' - nuInf) .^2)) ;
    
%     figure();
%     plot(inputs, consFfwd([3, 10, 20, 35], :), '.');
%     figure();
%     plot(inputs, nuInf, '.');
%     xlabel('input frequencies');
%     ylabel('output frequencies');
end
figure();
plot(numN, rmse, '.');
ylabel('RMSE');
xlabel('population size');