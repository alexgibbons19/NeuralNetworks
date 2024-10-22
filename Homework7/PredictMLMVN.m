function [ActualOutput,PredictedSet] = PredictMLMVN(testing,w1,w2,wout,predictions)

[m,n] = size(testing);
PredictedSet = zeros(predictions,n);
ActualOutput = zeros(1,predictions);

for i=1:predictions
    PredictedSet(i,:) = testing;
    [RMSE,ActualOutput(i),y_d] = Net_test(PredictedSet(i,:), w1,w2,wout);
    testing(1,1:n-1) = testing(1,2:n);
    testing(1,n) = ActualOutput(i);
end