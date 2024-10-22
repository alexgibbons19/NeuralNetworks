clc; clear;
% declare input vector
x0 = [1,1,1,1];
x1 = [1,0.5,-0.5,-1];
x2 = [1,-0.3,0.7,-1];
% declare desired output vectors
d1 = [1,1,1,1];		% NULL
d2 = [1,1,1,-1];	% AND
d3 = [1,1,-1,1];	% x1 not x2
d4 = [1,1,-1,-1];	% x1
d5 = [1,-1,1,1];	% not x1 x2
d6 = [1,-1,1,-1];	% x2
d7 = [1,-1,-1,1];   % XOR
d8 = [1,-1,-1,-1];  % OR
d9 = [-1,1,1,1];    % NOR
d10 = [-1,1,1,-1];  % NXOR
d11 = [-1,1,-1,1];  % not x2
d12 = [-1,1,-1,-1]; % x1 or not x2
d13 = [-1,-1,1,1];  % not x1
d14 = [-1,-1,1,-1]; % not x1 or x2
d15 = [-1,-1,-1,1]; % NAND
d16 = [-1,-1,-1,-1];% Identity

X = [x0;x1;x2];
D = [d1;d2;d3;d4;d5;d6;d7;d8;d9;d10;d11;d12;d13;d14;d15;d16];
Z = zeros(size(D));
for i = 1:height(D)
  % get weighted vector
  W = Hebb(X,D(i,:));
  for k = 1:length(X)
    Z(i,k) = W*X(:,k);
  end
end
disp("Linear Combinations of weights obtained and inputs: ");
disp(Z');
sgnZ = sign(Z);
disp("Sign of linear combinations obtained: ");
disp(sgnZ');
disp("Desired outputs: ");
disp(D');
% find the vectors that are not the same
HebbianRuleNotImplemented = D ~= sgnZ;
disp("Where the weights obtained do not implement the Hebbian rule: ");
disp(HebbianRuleNotImplemented');


