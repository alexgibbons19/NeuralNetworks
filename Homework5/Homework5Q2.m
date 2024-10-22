% After testing multiple times and comparing the results between
% using anywhere from 2-10 hidden neurons, I was unable to accurately
% choose what number of hidden neurons would give the best results. I
% believe this is due to the rng not being truly random. DUe to this, I
% decided to perfomr question 2 on all 5 different number of hidden neurons
% and comparing the mean results of them to determine what number of hidden
% neurons has the highest Classification Rate.

clc; clear;
load('wdbc_MLP_Learning_Testing_Validation_Data.mat');

maxIter = 5000;
thresh = 0.1;
numOutputNeurons = 1;
printRate = 5000;

experiment = [1:5]';

allCR = zeros(5);
allIter = zeros(5);
allLRMSE = zeros(5);
allTRMSE = zeros(5);

for j=1:5
	for i=1:5
		[tempNetW, allIter(i,j), allLRMSE(i,j)] = LearningMLP(Learning_wdbc_MLP,j*2,numOutputNeurons,thresh,maxIter,printRate);
		[allTRMSE(i,j),allCR(i,j)] = TestingMLP2(Testing_wdbc_MLP,j*2,tempNetW);
	end
end


meanLRMSE = zeros(5,1);
meanIter = zeros(5,1);
meanTRMSE  = zeros(5,1);
meanCR = zeros(5,1);

meanLRMSE = mean(allLRMSE);
meanIter = mean(allIter);
meanTRMSE = mean(allTRMSE);
meanCR = mean(allCR);

neurons = [2:2:10]';

avgResults = table;
avgResults.numNeurons = neurons;
avgResults.meanLRMSE = meanLRMSE';
avgResults.meanIterations = meanIter';
avgResults.meanTRMSE = meanTRMSE';
avgResults.meanClassificationRate = meanCR';




