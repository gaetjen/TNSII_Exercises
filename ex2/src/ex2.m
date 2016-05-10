%authors: Johannes Gätjen, Lorena Morton
%threshold for one or two steady states between 2.38 and 2.381
kappa = 6; %free parameter, change between 2 and 6


deltaT = 0.01;
maxT = 10;
time_vector = 0:deltaT:maxT;

Fmax = 40;

input = [-10; 10];
initial = [5; 15];
activity = repmat(initial, 1, length(time_vector));
connectionM = [-2 3; -3 2 ];
for t = 1:length(time_vector) - 1
    currentSteadyState = Factivation((connectionM * activity(:, t)) + input, Fmax, kappa); % v_i^ss
    activity(:, t+1) = currentSteadyState + (activity(:, t) - currentSteadyState) * exp(-deltaT);
end

plot(time_vector, activity(1, :));hold on;
plot(time_vector, activity(2, :), 'r');
figure();
plot(activity(1, :), activity(2, :));