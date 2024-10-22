
%This is for the segmentation single output neuron data set.
%Build a new Y array for the training and testing data, with multiple
%outputs per learning sample, corresponding to the multiple output neurons.

function [Y_learn_new, Y_test_new] = MultOutNeurLabels(Y_learn, Y_test)

%number of samples
N_learn = size(Y_learn, 1);
N_test = size(Y_test, 1);

%number of classes
num_classes = 7;

Y_learn_new = ones(N_learn, num_classes);
Y_test_new = ones(N_test, num_classes);

%Learning Y
for ii = 1 : N_learn
    
    class_label = Y_learn(ii);
    
    %The corresponding output neuron in Y_learn_new should have desired
    %output 0 (the rest are 1)
    Y_learn_new(ii, class_label+1) = 0;
end


%Testing Y
for ii = 1 : N_test
    
    class_label = Y_test(ii);
    
    %The corresponding output neuron in Y_test_new should have desired
    %output 0 (the rest are 1)
    Y_test_new(ii, class_label+1) = 0;
end