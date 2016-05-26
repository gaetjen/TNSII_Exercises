function [rj_mean, rj_noisy] = GaussResp_ConstantSTD(nu,nu_pref,RMax,sigma)

% nu = stimulus frequency vector (of type [1,N])
% kappa = parameter which determines the standard deviation (a number)
% nu_pref = [1,N] vector of preferred frequencies
% RMax = R_{max} (a  single number)
% rj_noisy and rj_mean are vectors of type [1,N]

[Nu_pref, Nu] = meshgrid( nu_pref, nu );



rj_mean = RMax*exp(-(1/2).*((Nu-Nu_pref)/(sigma)).^2);
rj_noisy = rj_mean + normrnd( 0, rj_mean);
rj_noisy(rj_noisy<0)=0;

