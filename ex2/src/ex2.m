%authors: Johannes Gätjen, Lorena Morton
%close all;
kappa = 6; %free parameter, change between 2 and 6

deltaT = 0.01;
maxT = 100;
time_vector = 0:deltaT:maxT;

Fmax = 40;

input = [-10; 10];
initial = [7; 11];
activity = repmat(initial, 1, length(time_vector));
connectionM = [-2 3; -3 2 ];
for t = 1:length(time_vector) - 1
    currentSteadyState = Factivation((connectionM * activity(:, t)) + input, Fmax, kappa); % v_i^ss
    activity(:, t+1) = currentSteadyState + (activity(:, t) - currentSteadyState) * exp(-deltaT);
end
figure;
plot(time_vector, activity);hold on;
xlabel('t [aut]');
ylabel('population activity');
legend('v_1', 'v_2');
figure();
plot(activity(1, :), activity(2, :)); hold on
xlabel('v_1');
ylabel('v_2');
plot(initial(1), initial(2), '+');
legend('trajectory');