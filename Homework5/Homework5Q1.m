clc;
clear;

% Load in data set
load('wdbc_MLP_Learning_Testing_Validation_Data.mat');

% create empty arrays to be inserted into table
numNeuronsArr = zeros(5,1);
allIter = zeros(5,1);
allLearnRMSE = zeros(5,1);
allTestRMSE = zeros(5,1);
allCR = zeros(5,1);

% set values for arguments
thresh = 0.1;
maxIter = 5000;
numOutputNeurons = 1;
printRate = 5000;

% loop through to get data using input neurons 2-10
for i = 1:5
	numNeuronsArr(i) = i*2;
	[tempNetW, allIter(i), allLearnRMSE(i)] = LearningMLP(Learning_wdbc_MLP,i*2,numOutputNeurons,thresh,maxIter,printRate);
	[allTestRMSE(i),allCR(i)] = TestingMLP2(Testing_wdbc_MLP,i*2,tempNetW);
end

% enter data into table
MLPResults = table;
MLPResults.numNeurons = numNeuronsArr;
MLPResults.LearningRMSE = allLearnRMSE;
MLPResults.iterations = allIter;
MLPResults.TestingRMSE = allTestRMSE;
MLPResults.ClassificationRate = allCR;

