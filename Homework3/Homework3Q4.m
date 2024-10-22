clc; clear;
X = [1,1,1,1,1,1,1,1;
    0.5,0.5,0.4,0.4,-0.3,-0.3,-0.7,-0.7;
    1,1,-0.5,-0.5,0.7,0.7,-1,-1;
    0.5,-0.3,0.4,-0.5,0.5,-0.4,0.3,-0.5];

D = [1,1,1,-1,1,1,-1,-1];

% weights obtained from Homework 2
W_HW2 = [0.2500,0.0238,0.8571,0.1667];

[WstartHW2,Wa,iter_a] = errorCorrection(X,D,30,1,W_HW2);
[WstartHW3,Wb,iter_b] = errorCorrection(X,D,30,0);

disp('Results when using "ideal weights":');
disp('Starting weights:');
disp(WstartHW2);
disp('Weights obtained:')
disp(Wa);
disp('Num iters with "ideal" weights:');
disp(iter_a);
disp('Results when using random weights:');
disp('Starting weights:');
disp(WstartHW3);
disp('Weights obtained:');
disp(Wb);
disp('Num iters with random weights:');
disp(iter_b);