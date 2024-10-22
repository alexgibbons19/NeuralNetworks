function [ClassifRate] = TestingMLPkOut(Input, nhiddenneurons, noutputs, netw)
% This program performs test of a trained MLP-MLF with 1 hidden layer
% and a binary output
% It takes inputs with desired outputs, calculates actual outputs and RMSE
% between actual and desired outputs

% Input contains a testing set (last column is a desired output)
% netw is an array of weights
% nhiddenneurons is the # of neurons in a single hidden layer

% new output neurons is 3
% noutputs = 3; % # of output neurons

[N,ninputs]=size(Input);

% Desired outputs
targets=Input(:,ninputs);

ninputs=ninputs-1;

% inputs contain only inputs
inputs = Input(: , 1:ninputs);

% An array to store actual outputs after learning
ActualOutputs = zeros(1, N);

% N is now the number of learning samples
% ninputs is now the number of inputs

% a for loop over all learning samples
for j=1:N
         % calculation of the actual output of the network for the j-th
         % sample
         output  = EvalNN( inputs(j,:),netw,ninputs,nhiddenneurons,noutputs );
         % disp(output)
         % interpretation of output in the case of more than 1 output
         % neurons (each one of them has binary output)
             
         % output at this point is an array containing noutputs
         % elements
             
         % evaluation of the distance of output from each of output
         % neuorns from 1, which is a target, that is we need to find
         % an output neuron fenerating the most reliable output
         outputDistance = abs(output - 1);
         
         % after the next statement is executed, IndOutputNeuron should
         % contain the index of output neuron generated the output
         % closest to 1
         [minDistance, IndOutputNeuron] = min(outputDistance);
         %disp(minDistance)
         %disp(IndOutputNeuron-1)
         
         % Accumulation of actual outputs
         % actual output for the j-th testing sample is assigned to 
         % IndOutputNeuron-1 this should be interpreted as a label of
         % the class (e.g., if there are 3 output neurons, an actual
         % output will be equal to 0 or 1 or 2
         ActualOutputs(j) = IndOutputNeuron-1;
           

         
         % Accumulation of actual outputs
         % ActualOutputs(j) = output;
        % disp([' Inputs [',num2str(inputs(j,:)), ' ] --> Actual Output:  [',num2str(output),']', '] --> Class: [',num2str(ActualOutputs(j)),'] --> Desired Output:  [',num2str(targets(j)),']'] )
         %disp(ActualOutputs(j));
end

% MSE over all testing samples
error = sum((ActualOutputs - targets').^2)/N;
% RMSE
RMSE = sqrt(error);

% Evaluation of the classification rate
% Results will contain 1 if there is no error and 0 otherwise
Results = (ActualOutputs' == targets);
NumOfCorrectOutputs = sum (Results);
ClassifRate = (NumOfCorrectOutputs / N) * 100;

%fprintf('Prediction/Recognition Error = %f \n', RMSE);
fprintf('Classification Rate = %f \n', ClassifRate);

figure (2);
hold off
plot(targets,'Or'); 
hold on
plot(ActualOutputs, '*b');

end
