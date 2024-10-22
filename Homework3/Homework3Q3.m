clc; clear;
% input/output mapping from slide 16
X16 = [1,1,1,1,1,1,1,1;1,1,1,1,-1,-1,-1,-1;1,1,-1,-1,1,1,-1,-1;1,-1,1,-1,1,-1,1,-1];
D16 = [1,1,1,-1,1,1,-1,-1];

% input/output mapping from slide 17
X17 = [1,1,1,1,1,1,1,1;1,1,1,1,-1,-1,-1,-1;1,1,-1,-1,1,1,-1,-1;1,-1,1,-1,1,-1,1,-1];
D17 = [1,-1,-1,-1,1,-1,1,-1];

MaxIter = 30;

% learn both i/o mappings with random weights 5 times, store starting
% vectors
W16SR = zeros(5,height(X16));
W16RR = zeros(5,height(X16));
iter16R = zeros(5,1);
W17SR = zeros(5,height(X17));
iter17R = zeros(5,1);

HWW16 = [0.2500,0.2500,0.7500,0.2500];

for i=1:5
  [W16Start,W16,iter16] = errorCorrection(X16,D16,MaxIter,0);
  W16SR(i,:) = W16Start;
  Q = dot(HWW16,W16Start);
  disp('Dot Profuct of Hebb weights and Random start weight');
  disp(Q);
  W16RR(i,:) = W16;
  Q = dot(HWW16,W16);
  disp('Dot Profuct of Hebb weights and resutling weight');
  disp(Q);
  iter16R(i) = iter16;
  [W17Start,W17,iter17] = errorCorrection(X17,D17,MaxIter,0);
  W17SR(i,:) = W17Start;
  iter17R(i) = iter17;
end

AvgIter16R = mean(iter16R);
AvgIter17R = mean(iter17R);

% disp randomly generated weighted vectors with mean iterations
disp("The 5 randomly generated starting weighted vectors for X16:");
disp(W16SR);
disp("Average number of iterations from these 5 random vectors:");
disp(AvgIter16R);

disp("The 5 randomly generated starting weighted vectors for X17:");
disp(W17SR);
disp("Average number of iterations from these 5 random vectors:");
disp(AvgIter17R);




