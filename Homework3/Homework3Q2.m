clc; clear;
% input/output mapping from slide 16
X16 = [1,1,1,1,1,1,1,1;1,1,1,1,-1,-1,-1,-1;1,1,-1,-1,1,1,-1,-1;1,-1,1,-1,1,-1,1,-1];
D16 = [1,1,1,-1,1,1,-1,-1];

% input/output mapping from slide 17
X17 = [1,1,1,1,1,1,1,1;1,1,1,1,-1,-1,-1,-1;1,1,-1,-1,1,1,-1,-1;1,-1,1,-1,1,-1,1,-1];
D17 = [1,-1,-1,-1,1,-1,1,-1];

MaxIter = 30;

% learn both i/o mappings with random weights
[W16Start,W16,iter16] = errorCorrection(X16,D16,MaxIter,0);
[W17aStart,W17a,iter17a] = errorCorrection(X17,D17,MaxIter,0);

% use Normalized hebbian weights from Homework 2
W17bStart = [-0.2500,-0.2500,0.2500,0.7500];
[W17bStart,W17b,iter17b] = errorCorrection(X17,D17,MaxIter,1,W17bStart);

% Compare results
Y16 = W16*X16;
Z16 = signZ(Y16);
Y17a = W17a*X17;
Z17a = signZ(Y17a);
Y17b = W17b*X17;
Z17b = signZ(Y17b);

% Display results
% Vector from slide 16
disp("The vector from slide 16 with random starting weights:");
disp("iters: ");
disp(iter16);
disp("Starting weights: ");
disp(W16Start);
disp("Normalized weights recieved: ");
disp(W16);
disp("Real results: ");
disp(Z16);
disp("Desired results: ");
disp(D16);

% Vector from slide 17
disp("The vector from slide 17 with random starting weights:");
disp("iters: ");
disp(iter17a);
disp("Starting weights: ");
disp(W17aStart);
disp("Normalized weights recieved: ");
disp(W17a);
disp("Real results: ");
disp(Z17a);
disp("Desired results: ");
disp(D17);

disp("The vector from slide 17 with starting weights from HW2:");
disp("iters: ");
disp(iter17b);
disp("Starting weights: ");
disp(W17bStart);
disp("Normalized weights recieved: ");
disp(W17b);
disp("Real results: ");
disp(Z17b);
disp("Desired results: ");
disp(D17);