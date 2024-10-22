function [ unmappedInputs ] = unmapInputs(Input)

[sortedVal, sortedInd] = sort(Input, 'ascend','ComparisonMethod','real');

unmappedInputs = zeros(size(Input));
unmappedInputs(sortedInd) = 0:length(Input)-1;


