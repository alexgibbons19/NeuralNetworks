clear;clc;

load('Poland_Power_Consumption_Load_1990s.mat');

n=183;
k=365;
globalThresh = 0.005;
LearningSetMLP = CreateLearningSet(Data_MLP,n,k);
TestingDataMLP = CreateLearningSet(Data_MLP,n,90);
learnRMSEMLP = zeros(1,4);
%% MLP Learning
[Network2NMLP, learnRMSEMLP(1)] = LerningMLP(LearningSetMLP, [2 64], globalThresh, 10000 );
[Network4NMLP, learnRMSEMLP(2)] = LerningMLP(LearningSetMLP, [4 64], globalThresh, 10000 );
[Network6NMLP, learnRMSEMLP(3)] = LerningMLP(LearningSetMLP, [6 64], globalThresh, 10000 );
[Network8NMLP, learnRMSEMLP(4)] = LerningMLP(LearningSetMLP, [8 64], globalThresh, 10000 );

pause(3)

%% MLP Predictions
numPredictions = 90;
DesiredOutputMLP = TestingDataMLP(:,n);
testingMLP = TestingDataMLP(1,1:n);

[ActualOutput2NMLP,PredictedSet2NMLP] = PredictMLP(testingMLP,Network2NMLP,numPredictions);
[ActualOutput4NMLP,PredictedSet4NMLP] = PredictMLP(testingMLP,Network4NMLP,numPredictions);
[ActualOutput6NMLP,PredictedSet6NMLP] = PredictMLP(testingMLP,Network6NMLP,numPredictions);
[ActualOutput8NMLP,PredictedSet8NMLP] = PredictMLP(testingMLP,Network8NMLP,numPredictions);

%% Get RMSE for each neural layout
testRMSEMLP = zeros(1,4);
testRMSEMLP(1) = sum((ActualOutput2NMLP - DesiredOutputMLP').^2)/numPredictions;
testRMSEMLP(2) = sum((ActualOutput4NMLP - DesiredOutputMLP').^2)/numPredictions;
testRMSEMLP(3) = sum((ActualOutput6NMLP - DesiredOutputMLP').^2)/numPredictions;
testRMSEMLP(4) = sum((ActualOutput8NMLP - DesiredOutputMLP').^2)/numPredictions;

%% MLMVN Learning
globalThresh = 0.05;
LearningSetMLMVN = CreateLearningSet(Data_MVN,n,k);
TestingDataMLMVN = CreateLearningSet(Data_MVN,n,90);
learnRMSEMLMVN = zeros(1,4);

[w1N2,w2N2,woutN2,iter2N,learnRMSEMLMVN(1)] = Net_learn(LearningSetMLMVN,[2 64],globalThresh);
[w1N4,w2N4,woutN4,iter4N,learnRMSEMLMVN(2)] = Net_learn(LearningSetMLMVN,[4 64],globalThresh);
[w1N6,w2N6,woutN6,iter6N,learnRMSEMLMVN(3)] = Net_learn(LearningSetMLMVN,[6 64],globalThresh);
[w1N8,w2N8,woutN8,iter8N,learnRMSEMLMVN(4)] = Net_learn(LearningSetMLMVN,[8 64],globalThresh);

%% MLMVN Testing
numPredictions = 90;
DesiredOutputMLMVN = TestingDataMLMVN(:,n);

testingMLMVN = TestingDataMLMVN(1,1:n);
testRMSEMLMVN = zeros(1,4);
[ActualOutput2NMLMVN,PredictedSet2NMLMVN] = PredictMLMVN(testingMLMVN,w1N2,w2N2,woutN2,numPredictions);
[ActualOutput4NMLMVN,PredictedSet4NMLMVN] = PredictMLMVN(testingMLMVN,w1N4,w2N4,woutN4,numPredictions);
[ActualOutput6NMLMVN,PredictedSet6NMLMVN] = PredictMLMVN(testingMLMVN,w1N6,w2N6,woutN6,numPredictions);
[ActualOutput8NMLMVN,PredictedSet8NMLMVN] = PredictMLMVN(testingMLMVN,w1N8,w2N8,woutN8,numPredictions);



testRMSEMLMVN(1) = sum((ActualOutput2NMLMVN - DesiredOutputMLMVN').^2)/numPredictions;
testRMSEMLMVN(2) = sum((ActualOutput4NMLMVN - DesiredOutputMLMVN').^2)/numPredictions;
testRMSEMLMVN(3) = sum((ActualOutput6NMLMVN - DesiredOutputMLMVN').^2)/numPredictions;
testRMSEMLMVN(4) = sum((ActualOutput8NMLMVN - DesiredOutputMLMVN').^2)/numPredictions;

%% Display colected data
neuronLabels = categorical(["2 Neurons","4 Nuerons","6 Neurons","8 Neurons"]);
figure ();
bar(neuronLabels,learnRMSEMLP)
title('RMSEs for each hidden layer type after MLP learning');
figure ();
bar(neuronLabels,testRMSEMLP)
title('RMSEs for each hidden layer type after MLP testing/prediction');

figure ();
hold off
plot(DesiredOutputMLP,'Or'); 
hold on
plot(ActualOutput2NMLP, '*b');
title('Actual vs Desired 2N MLP')

figure ();
hold off
plot(DesiredOutputMLP,'Or'); 
hold on
plot(ActualOutput4NMLP, '*b');
title('Actual vs Desired 4N MLP')

figure ();
hold off
plot(DesiredOutputMLP,'Or'); 
hold on
plot(ActualOutput6NMLP, '*b');
title('Actual vs Desired 6N MLP')

figure ();
hold off
plot(DesiredOutputMLP,'Or'); 
hold on
plot(ActualOutput8NMLP, '*b');
title('Actual vs Desired 8N MLP')

figure()
bar(neuronLabels,learnRMSEMLMVN)
title('RMSEs for each hidden layer type after MLMVN learning');

figure()
bar(neuronLabels,testRMSEMLMVN)
title('RMSEs for each hidden layer type after MLMVN testing/prediction');

figure()
bar(neuronLabels,[iter2N,iter4N,iter6N,iter8N])
title('Iterations for each hidden layer type during learning');

figure ();
hold off
plot(DesiredOutputMLMVN,'Or'); 
hold on
plot(ActualOutput2NMLMVN, '*b');
title('Actual vs Desired 2N MLMVN')

figure ();
hold off
plot(DesiredOutputMLMVN,'Or'); 
hold on
plot(ActualOutput4NMLMVN, '*b');
title('Actual vs Desired 4N MLMVN')

figure ();
hold off
plot(DesiredOutputMLMVN,'Or'); 
hold on
plot(ActualOutput6NMLMVN, '*b');
title('Actual vs Desired 6N MLMVN')

figure ();
hold off
plot(DesiredOutputMLMVN,'Or'); 
hold on
plot(ActualOutput8NMLMVN, '*b');
title('Actual vs Desired 8N MLMVN')
