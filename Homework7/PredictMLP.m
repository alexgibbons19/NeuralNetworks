function [ActualOutput,PredictedSet] = PredictMLP(testing,Network,predictions)

[m,n] = size(testing);
PredictedSet = zeros(predictions,n);
ActualOutput = zeros(1,predictions);

for i=1:predictions
    PredictedSet(i,:) = testing;
    [RMSE,ActualOutput(i)] = TestingMLP(PredictedSet(i,:), Network);
    testing(1,1:n-1) = testing(1,2:n);
    testing(1,n) = ActualOutput(i);
end
