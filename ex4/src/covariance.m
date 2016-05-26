function Wji = covariance( Ri, Rj )

% Ri responses of Ni neurons i, to K stimuli, size [ K Ni ]
% Rj responses of Nj neurons j, to the same K stimuli, size [K Nj]
% Wji matrix of response covariances for all neuron pairs j, i

[K, Ni] = size( Ri );
[K, Nj] = size( Rj );

% product of responses for all pairs j, i, summed up and devided by K
% size [Nj, Ni]
expRJRI = Rj'*Ri/K;

% compute expected individual response by averaging over stimuli
% size [1 Ni]
% size [1 Nj]
expRi = mean(Ri,1);
expRj = mean(Rj,1);

% compute covariance matrix
Wji = expRJRI - expRj'*expRi;

% display covariance matrix

% figure;
% pcolor( Wji ); 
% axis 'square';
% ylabel('Nj neurons j');
% xlabel('Ni neurons i');
