clc; clear;
x0 = [1,1,1,1,1,1,1,1];
x1 = [0.5,0.5,0.4,0.4,-0.3,-0.3,-0.7,-0.7];
x2 = [1,1,-0.5,-0.5,0.7,0.7,-1,-1];
x3 = [0.5,-0.3,0.4,-0.5,0.5,-0.4,0.3,-0.5];

d1 = [1,1,1,-1,1,1,-1,-1];
d2 = [1,-1,-1,-1,1,-1,1,-1];

X = [x0;x1;x2;x3];
D = [d1;d2];
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
sgnZ = signZ(Z);
disp("Sign of linear combinations obtained: ");
disp(sgnZ');
disp("Desired outputs: ");
disp(D');
% find the vectors that are not the same
HebbianRuleNotImplemented = D ~= sgnZ;
disp("Where the weights obtained do not implement the Hebbian rule: ");
disp(HebbianRuleNotImplemented');