clc; clear;
load('Iris_Data_Homework6.mat');

% declare constants
err_thresh = 0.01;
max_iter = 5000;
numFolders = 5;
numOutputNeurons = 3;

%% -------- Divide Data into folders -------- %%
% Divide Datasets into 5 folders
MLP_Learning_folders = datasetDivision(IrisMLPLearn3,3,5);
MLP_Testing_folders = datasetDivision(IrisMLP,3,5);
MLMVN_Learning_folders = datasetDivision(IrisMLMVNLearn3,3,5);
MLMVN_Testing_folders = datasetDivision(Iris_Data_MLMVN,3,5);
SVM_folders = datasetDivision(IrisSVM,3,5);

%% -------- MLP -------- %%
%% learning/testing MLP 4 hidden neurons
% allocate data to store results
MLP_4HNeurons_Iters = zeros(5,1);
MLP_4HNeurons_RMSEs = zeros(5,1);
MLP_4HNeurons_ClassIfs = zeros(5,1);
numHiddenNeurons = 4;

for i=1:numFolders
    if(i==1)
        learningSet = cell2mat(MLP_Learning_folders(2:numFolders)');
    elseif(i==numFolders)
        learningSet = cell2mat(MLP_Learning_folders(1:numFolders-1)');
    else
        learningSet = [cell2mat(MLP_Learning_folders(1:i-1)');
            cell2mat(MLP_Learning_folders(i+1:numFolders)')];
    end
    [tempNetW,MLP_4HNeurons_Iters(i),MLP_4HNeurons_RMSEs(i)] = LearningMLPkOut(learningSet,numHiddenNeurons,numOutputNeurons,err_thresh,max_iter,5000);
    [MLP_4HNeurons_ClassIfs(i)] = TestingMLPkOut(cell2mat(MLP_Testing_folders(i)),numHiddenNeurons,numOutputNeurons,tempNetW);
end

%% learning/testing MLP 6 hidden neurons
% allocate data to store results
MLP_8HNeurons_Iters = zeros(5,1);
MLP_8HNeurons_RMSEs = zeros(5,1);
MLP_8HNeurons_ClassIfs = zeros(5,1);
numHiddenNeurons = 8;

for i=1:numFolders
    if(i==1)
        learningSet = cell2mat(MLP_Learning_folders(2:numFolders)');
    elseif(i==numFolders)
        learningSet = cell2mat(MLP_Learning_folders(1:numFolders-1)');
    else
        learningSet = [cell2mat(MLP_Learning_folders(1:i-1)');
            cell2mat(MLP_Learning_folders(i+1:numFolders)')];
    end
    [tempNetW,MLP_8HNeurons_Iters(i),MLP_8HNeurons_RMSEs(i)] = LearningMLPkOut(learningSet,numHiddenNeurons,numOutputNeurons,err_thresh,max_iter,5000);
    [MLP_8HNeurons_ClassIfs(i)] = TestingMLPkOut(cell2mat(MLP_Testing_folders(i)),numHiddenNeurons,numOutputNeurons,tempNetW);
end

%% learning/testing MLP 8 hidden neurons
% allocate data to store results
MLP_12HNeurons_Iters = zeros(5,1);
MLP_12HNeurons_RMSEs = zeros(5,1);
MLP_12HNeurons_Classifs = zeros(5,1);
numHiddenNeurons = 12;

for i=1:numFolders
    if(i==1)
        learningSet = cell2mat(MLP_Learning_folders(2:numFolders)');
    elseif(i==numFolders)
        learningSet = cell2mat(MLP_Learning_folders(1:numFolders-1)');
    else
        learningSet = [cell2mat(MLP_Learning_folders(1:i-1)');
            cell2mat(MLP_Learning_folders(i+1:numFolders)')];
    end
    [tempNetW,MLP_12HNeurons_Iters(i),MLP_12HNeurons_RMSEs(i)] = LearningMLPkOut(learningSet,numHiddenNeurons,numOutputNeurons,err_thresh,max_iter,5000);
    [MLP_12HNeurons_Classifs(i)] = TestingMLPkOut(cell2mat(MLP_Testing_folders(i)),numHiddenNeurons,numOutputNeurons,tempNetW);
end

%% -------- SVM -------- %%
SVM_Gauss_Classifs = zeros(5,1);
SVM_Poly_Classifs = zeros(5,1);
SVM_Lin_Classifs = zeros(5,1);
for i=1:numFolders
    if(i==1)
        LearningSVM = cell2mat(SVM_folders(2:numFolders)');
    elseif(i==numFolders)
        LearningSVM = cell2mat(SVM_folders(1:numFolders-1)');
    else
        LearningSVM = [cell2mat(SVM_folders(1:i-1)');
            cell2mat(SVM_folders(i+1:numFolders)')];
    end
    TestingSVM = cell2mat(SVM_folders(i));
    LabelsLearningSVM = LearningSVM(:,end);
    [n, m] = size(LearningSVM);
    LearningDataSVM = LearningSVM(:,1:m-1);
    [n, m] = size(TestingSVM);
    LabelsTestingSVM = TestingSVM(:,end);
    TestingDataSVM = TestingSVM(:,1:m-1);
    %% Gaussian
    t = templateSVM('KernelFunction','gaussian');
    Mdl = fitcecoc(LearningDataSVM,LabelsLearningSVM, 'Learners', t);
    classificationResult = predict(Mdl,TestingDataSVM);
    SVM_Gauss_Classifs(i) = mean(LabelsTestingSVM==classificationResult)*100;
    
    %% Polynomial
    t = templateSVM('KernelFunction','polynomial');
    Mdl = fitcecoc(LearningDataSVM,LabelsLearningSVM, 'Learners', t);
    classificationResult = predict(Mdl,TestingDataSVM);
    SVM_Poly_Classifs(i) = mean(LabelsTestingSVM==classificationResult)*100;
    
    
    %% Linear
    t = templateSVM('KernelFunction','linear');
    Mdl = fitcecoc(LearningDataSVM,LabelsLearningSVM, 'Learners', t);
    classificationResult = predict(Mdl,TestingDataSVM);
    SVM_Lin_Classifs(i) = mean(LabelsTestingSVM==classificationResult)*100;
    
end
%% -------- MLMVN -------- %%
%% Assign Parameters
outneur_num = 3;  % # of output neurons
sec_nums = [2, 2, 2]; % # of classes (sectors) per each of output neurons
RMSE_thresh = pi/8 ; % pi/8 - threshold for snugular RMSE (soft margins)
local_thresh = 0; % threshols for angular deviation to decide whether to correct the weights
win_ang = 3*pi/2; % a desired output for a winning class
max_iter = 100000;
%% 4 Hidden Neurons
MLMVN_4HNeurons_Iters = zeros(5,1);
MLMVN_4HNeurons_RMSEs = zeros(5,1);
MLMVN_4HNeurons_ClassIfs = zeros(5,1);
hidneur_num = 4;  % # of hidden neurons

for i=1:numFolders
    if(i==1)
        MLMVNlearningSet = cell2mat(MLMVN_Learning_folders(2:numFolders)');
    elseif(i==numFolders)
        MLMVNlearningSet = cell2mat(MLMVN_Learning_folders(1:numFolders-1)');
    else
        MLMVNlearningSet = [cell2mat(MLMVN_Learning_folders(1:i-1)');
            cell2mat(MLMVN_Learning_folders(i+1:numFolders)')];
    end
    [hidneur_weights, outneur_weights, MLMVN_4HNeurons_Iters(i)] = Net_learn(MLMVNlearningSet, hidneur_num, outneur_num, sec_nums, RMSE_thresh, local_thresh);
    pause (5);
    MLMVN_4HNeurons_ClassIfs(i) = Net_test(cell2mat(MLMVN_Testing_folders(i)), hidneur_weights, outneur_weights, win_ang)*100;
    fprintf('\nAccuracy = %f\n',MLMVN_4HNeurons_ClassIfs(i))
end

%% 8 Hidden Neurons
MLMVN_8HNeurons_Iters = zeros(5,1);
MLMVN_8HNeurons_RMSEs = zeros(5,1);
MLMVN_8HNeurons_ClassIfs = zeros(5,1);
hidneur_num = 8;  % # of hidden neurons

for i=1:numFolders
    if(i==1)
        MLMVNlearningSet = cell2mat(MLMVN_Learning_folders(2:numFolders)');
    elseif(i==numFolders)
        MLMVNlearningSet = cell2mat(MLMVN_Learning_folders(1:numFolders-1)');
    else
        MLMVNlearningSet = [cell2mat(MLMVN_Learning_folders(1:i-1)');
            cell2mat(MLMVN_Learning_folders(i+1:numFolders)')];
    end
    [hidneur_weights, outneur_weights, MLMVN_8HNeurons_Iters(i)] = Net_learn(MLMVNlearningSet, hidneur_num, outneur_num, sec_nums, RMSE_thresh, local_thresh);
    MLMVN_8HNeurons_ClassIfs(i) = Net_test(cell2mat(MLMVN_Testing_folders(i)), hidneur_weights, outneur_weights, win_ang)*100;
    fprintf('\nAccuracy = %f\n',MLMVN_8HNeurons_ClassIfs(i))
end

%% 12 Hidden Neurons
MLMVN_12HNeurons_Iters = zeros(5,1);
MLMVN_12HNeurons_RMSEs = zeros(5,1);
MLMVN_12HNeurons_ClassIfs = zeros(5,1);
hidneur_num = 12;  % # of hidden neurons

for i=1:numFolders
    if(i==1)
        MLMVNlearningSet = cell2mat(MLMVN_Learning_folders(2:numFolders)');
    elseif(i==numFolders)
        MLMVNlearningSet = cell2mat(MLMVN_Learning_folders(1:numFolders-1)');
    else
        MLMVNlearningSet = [cell2mat(MLMVN_Learning_folders(1:i-1)');
            cell2mat(MLMVN_Learning_folders(i+1:numFolders)')];
    end
    [hidneur_weights, outneur_weights, MLMVN_12HNeurons_Iters(i)] = Net_learn(MLMVNlearningSet, hidneur_num, outneur_num, sec_nums, RMSE_thresh, local_thresh);
    MLMVN_12HNeurons_ClassIfs(i) = Net_test(cell2mat(MLMVN_Testing_folders(i)), hidneur_weights, outneur_weights, win_ang)*100;
    fprintf('\nAccuracy = %f\n',MLMVN_12HNeurons_ClassIfs(i))
end
%% -------- Compile Data -------- %%
MLPmeans = zeros(5,1);
SVMmeans = zeros(5,1);
MLMVNmeans = zeros(5,1);
ExperimentMeans = zeros(5,1);
for i=1:5
    % a classification rate for each experiment with MLP
    MLPmeans(i) = (MLP_4HNeurons_ClassIfs(i)+MLP_8HNeurons_ClassIfs(i)+MLP_12HNeurons_Classifs(i))/3;
    % a classification rate for the same folder for each experiment with SVM
    SVMmeans(i) = (SVM_Gauss_Classifs(i)+SVM_Poly_Classifs(i)+SVM_Lin_Classifs(i))/3;
    classification rate for each experiment with MLMVN
    MLMVNmeans(i) = (MLMVN_4HNeurons_ClassIfs(i) + MLMVN_8HNeurons_ClassIfs(i) + MLMVN_12HNeurons_ClassIfs(i))/3;
    ExperimentMeans(i) = (MLPmeans(i) + SVMmeans(i) + MLMVNmeans(i))/3;
end
% average classification rates over all experiments 
% with MLP, SVM, and MLMVN, respectively
MLPTotMean = mean(MLPmeans);
SVMTotMean = mean(SVMmeans);
MLMVNTotMean = mean(MLMVNmeans);

titles = ["Type"; "Experiment 1"; "Experiment 2"; "Experiment 3"; "Experiment 4"; "Experiment 5"];
Type = ["MLP"; "SVM"; "MLMVN"; "TotalMean"];
Experiment_1 = [MLPmeans(1);SVMmeans(1);MLMVNmeans(1);ExperimentMeans(1)];
Experiment_2 = [MLPmeans(2);SVMmeans(2);MLMVNmeans(2);ExperimentMeans(2)];
Experiment_3 = [MLPmeans(3);SVMmeans(3);MLMVNmeans(3);ExperimentMeans(3)];
Experiment_4 = [MLPmeans(4);SVMmeans(4);MLMVNmeans(4);ExperimentMeans(4)];
Experiment_5 = [MLPmeans(5);SVMmeans(5);MLMVNmeans(5);ExperimentMeans(5)];

Means = table(Type,Experiment_1,Experiment_2,Experiment_3,Experiment_4,Experiment_5);
TotalMeans = table(MLPTotMean,SVMTotMean,MLMVNTotMean);












