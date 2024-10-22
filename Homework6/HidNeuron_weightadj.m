
%Weights adjustment for a hidden neuron
function w_adj = HidNeuron_weightadj(X_pinv, w, delta)
% (c) E.Aizenberg & I.Aizenberg 2014-2018

%function w_adj = HidNeuron_weightadj(X_pinv, w, delta, N)


%X = matrix of hidden neuron inputs (N x n), where N=number of learning
%samples, n = number of input variables

%w = (n+1 x 1) vector of weights of the hidden neuron

%delta = (N x 1) vector of errors

%X_pinv = pre-computed pseudo-inverse of the matrix of network inputs



%LLS: apply X_pinv to delta
adj_vec = X_pinv * (1 .* delta); 

%the new weights are given by 
w_adj = w + adj_vec;




