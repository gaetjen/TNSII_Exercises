%authors: Johannes Gätjen, Lorena Morton
function inputVector = randomInput()
    inputVector = zeros(1,20);
    pos = randi(19);
    inputVector([pos, pos + 1]) = randi(3);
end