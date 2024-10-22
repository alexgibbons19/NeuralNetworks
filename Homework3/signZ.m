function [sgnZ] = signZ(Z)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    sgnZ = zeros(size(Z));
    for i = 1:length(Z) 
        for j = 1:height(Z)
            if(Z(j,i) >= 0)
                sgnZ(j,i) = 1;
            else
                sgnZ(j,i) = -1;
            end
        end
    end     
end

