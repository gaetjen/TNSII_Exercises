function ShowSequence( U, R, D, W )
% Usage:               ShowSequence(U,R,D,W)
%
% U(2,Ntrials)         matrix with zeros and ones, indicating presence or
%                      absence of 2 stimuli
% R(1,Ntrials)         vector with zeros and ones, indicating reward
%                      awareded or withheld
% D(1,Ntrials)         vector of scalar, giving the "delta" of each trial
% W(2,Ntrials)         matrix of scalars, giving the linear weights after
%                      each trial

[N,Ntrials] = size(U);

xi = [-0.25 -0.25 0.25 0.25];
yi = [-0.25 0.25 0.25 -0.25];
ri = [0 1 1 0];

clr = [ 1 0 0; 0 0 1;  0 1 0; 0.8 0.8 0];

clf;
subplot(2,1,1);

hold on;
for c=1:N
    kk = find(U(c,:) == 1);
    for i=1:length(kk)
        fill( kk(i)+xi, c+yi, clr(c,:));
    end
end

kk = find(R(1,:) > 0);
for i=1:length(kk)
    fill( kk(i)+xi, N+1+yi, clr(3,:)*R(kk(i)) );
end
    
hold off;
axis([0 Ntrials+1 0 N+2]);
ylabel('Stimuli, Reward');
xlabel('Trials');

mn = min( min(D), min(min(W)) );
mx = max( max(D), max(max(W)) );

subplot(2,1,2);
hold on;
for c=1:N
    plot(1:Ntrials, W(c,:), ':', 'Color', clr(c,:), 'LineWidth', 2.0);
end   
x = 1:Ntrials;
kk = find(D(:)~=0);
plot(x(kk), D(kk), 'ko', 'LineWidth', 2.0);
hold off;
axis([0 Ntrials+1 mn mx]);
ylabel('Weights, Delta');
xlabel('Trials');
