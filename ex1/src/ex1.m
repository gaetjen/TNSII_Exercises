%authors: Johannes Gätjen, Lorena Morton

numPop = 20; % N
%input vector h
input = randomInput();
[connectionM, e1, e2, lambda1, lambda2] = ConnectionMatrix(numPop);
plot(1:20, e1, 'LineWidth', 2);
hold on;
plot(1:20, e2, 'r', 'LineWidth', 2);
plot(1:20, input, 'g', 'LineWidth', 2);
legend('eigenvector E1', 'eigenvector E2', 'input activity');
xlabel('population index');
ylabel('activity');
xlim([1, 20]);

e1Proj = [e1' * e1, e1' * e2];
e2Proj = [e2' * e1, e2' * e2];
inputProj = [input * e1, input * e2];

figure();
eigenh = plot([0, e1Proj(1)], [0, e1Proj(2)], [0, e2Proj(1)], [0, e2Proj(2)], 'b', 'LineWidth', 2);
hold on;
inputh = plot([0, inputProj(1)], [0, inputProj(2)], 'g', 'LineWidth', 2);
xlabel('E1');
ylabel('E2');
xlim([-1.5, 2])
ylim([-1.5, 2])
legend([eigenh(1), inputh], {'eigenvectors', 'input projection'});
axis equal;

figure();
plot([0, e1Proj(1)], [0, e1Proj(2)], [0, e2Proj(1)], [0, e2Proj(2)], 'b', 'LineWidth', 2)
hold on;
for i = 1:20
    input = randomInput();
    steadyState = (input * e1) / (1 - lambda1) * e1 + (input * e2) / (1 - lambda2) * e2;
    projection = [steadyState' * e1, steadyState' * e2];
    ssh = plot([0, projection(1)], [0, projection(2)], '-r*');
end
xlabel('E1');
ylabel('E2');
legend(ssh, 'steady states');

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
    inputh = plot(inputProj(1), inputProj(2), 'r*');
    steadyState = ((input * e1) / (1 - lambda1)) * e1 + ((input * e2) / (1 - lambda2)) * e2;
    steadyStateProj = [steadyState' * e1, steadyState' * e2];
    ssh = plot(steadyStateProj(1), steadyStateProj(2), 'g*');
    activityProj = [activity * e1, activity * e2];
    activityh = plot(activityProj(:, 1), activityProj(:, 2));
    axis equal;
end
xlabel('E1');
ylabel('E2');
legend([inputh; ssh; activityh], {'initial activity'; 'steady states'; 'activity time-evolution'}, 'Location', 'best')


figure();
surfl(1:20, time,activity);
shading interp;
colormap(gray);
xlabel('population index');
ylabel('time [aut]');
zlabel('activity');


