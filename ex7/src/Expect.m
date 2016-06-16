% authors: Johannes Gätjen, Lorena Morton
function probs = Expect( samples, gammas, mus, sigmas )
probs = zeros(length(gammas), length(samples));

for i = 1:length(gammas)
    probs(i, :) = gammas(i) * normpdf(samples, mus(i), sigmas(i));
end
total = sum(probs);
for i = 1:length(gammas)
    probs(i, :) = probs(i, :) ./ total;
end

end

