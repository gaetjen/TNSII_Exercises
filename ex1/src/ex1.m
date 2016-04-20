%authors: Johannes Gätjen, Lorena Morton

numPop = 20; % N
%input vector h
input = randomInput();
[connectionM, e1, e2, lambda1, lambda2] = ConnectionMatrix(numPop);
plot(1:20, e1);
hold on;
plot(1:20, e2, 'r');
plot(1:20, input);

e1Proj = [e1' * e1, e1' * e2];
e2Proj = [e2' * e1, e2' * e2];
inputProj = [input * e1, input * e2];

figure();
plot([0, e1Proj(1)], [0, e1Proj(2)], [0, e2Proj(1)], [0, e2Proj(2)], 'b', 'LineWidth', 2)
hold on;
plot([0, inputProj(1)], [0, inputProj(2)], 'g', 'LineWidth', 2)
xlim([-1.5, 2])
ylim([-1.5, 2])

figure();
plot([0, e1Proj(1)], [0, e1Proj(2)], [0, e2Proj(1)], [0, e2Proj(2)], 'b', 'LineWidth', 2)
hold on;
for i = 1:20
    input = randomInput();
    steadyState = (input * e1) / (1 - lambda1) * e1 + (input * e2) / (1 - lambda2) * e2;
    projection = [steadyState' * e1, steadyState' * e2];
    plot([0, projection(1)], [0, projection(2)], '-r*');
end

dt = 0.05;
time = 0:dt:20;
figure();
plot([0, e1Proj(1)], [0, e1Proj(2)], [0, e2Proj(1)], [0, e2Proj(2)], 'b', 'LineWidth', 2)
hold on;
for i = 1:10
    input = randomInput();
    activity = repmat(input, length(time), 1); % vector v in assignment, over time
    for t = 1:length(time) - 1
        currentSteadyState = (connectionM * activity(t, :)')' + input; % v_i^ss
        activity(t+1, :) = currentSteadyState + (activity(t, :) - currentSteadyState) * exp(-dt);
    end
    inputProj = [input * e1, input * e2];
    plot(inputProj(1), inputProj(2), 'r*');
    steadyState = ((input * e1) / (1 - lambda1)) * e1 + ((input * e2) / (1 - lambda2)) * e2;
    steadyStateProj = [steadyState' * e1, steadyState' * e2];
    plot(steadyStateProj(1), steadyStateProj(2), 'g*');
    activityProj = [activity * e1, activity * e2];
    plot(activityProj(:, 1), activityProj(:, 2));
    axis equal;
end



