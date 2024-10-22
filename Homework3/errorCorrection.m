function [Wstart,W, iter] = errorCorrection(X, Desired, MaxIter, opt, Wstart)
%  $Author: Alex Gibbons
%input:
%  X = input vector
%  Desired = desired output(s)
%  MaxIter - maximum number of iterations before terminating
%  opt - determines if a starting weighted vector will be used or randomly generated
%      - 1 for passed
%      - 0 for randomly generated
%  Wstart - starting weighted vector to be used
%output:
%  Wstart = starting weighted vector used
%  W = resulting weighted vector
%  iter = number of iterations needed to learn

  if nargin<2,error('Insufficient arguments: please enter Learning Set'),end
  if nargin<3,  opt = 0; end
  if nargin<4 && opt == 1,error('Please enter starting weighted vector');  end
  N = length(X);
  M = height(X);
  LR = 1;
  if opt == 0
    Wstart = zeros(1,M);
    for k = 1:M
      Wstart(k) = -0.5 + rand;
    end
  end  
  
  W = Wstart;

  flag = true;
  iter = 1;
  
  while iter < MaxIter
    j = 1;
    while j <= length(X)
      Z = W * X(:,j);
      y = signZ(Z);
      if y ~= Desired(j)
        flag = false;
        err = Desired(j) - y;
        for i = 1:M
          W(i) = W(i) + (LR*err*((X(i,j)^-1)));
        end
      end
      j = j+1;
      if j > N
        if flag == false
          iter = iter + 1;
          j = 1;
          flag = true;
        else
          return
        end
      end     
    end
  end

