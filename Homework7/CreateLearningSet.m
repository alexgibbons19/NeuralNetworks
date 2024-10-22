function [TS_Learning_Set] = CreateLearningSet( Time_Series,n,k )
% TimeSeries - an array containing a time series
% n – the number of members in the series
%   x(n+1) = f(x1,...,xn)
% k – the number of samples to be included in the learning set

if length(Time_Series) < n || length(Time_Series) < (n + k)
    err('n cannot be larger than the length of the length of the Time Series');
end

TS_Learning_Set = zeros(k,n);

for i=1:k
    TS_Learning_Set(i,:) = Time_Series(i:n-1+i);
end