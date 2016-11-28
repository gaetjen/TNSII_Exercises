function [ maxv ] = infer( integrated, conspace, orispace )
%MAXES inferred orientation and contrast, given a series of 
%   Detailed explanation goes here
maxv = zeros(3, size(integrated, 3));
for i = 1:size(integrated, 3)
    intg = integrated(:, :, i);
    [maxv(1, i), idx] = max(intg(:));
    [conI, oriI] = ind2sub(size(intg), idx);
    maxv(2, i) = (conspace(conI+1)+conspace(conI))/2;
    maxv(3, i) = (orispace(oriI+1)+orispace(oriI))/2;
end

end
