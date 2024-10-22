function [ Weights, iterations ] = errorCorrectionRmse ( Input, eps, Start, W )
% errorCorrection(Input) returns a weighting vector for the input/output
% mapping presented by N x (n+1) matrix Input containing N rows of learning
% samples, n inputs each foollowed by a desired output, found using the
% error-correction learning rule
% If Start = 1, then starting weights are read from W, otherwise random
% weights should be generated
% eps is a tolerance threshold to stop learning process

% determine size of the Input matrix
[N, m] = size(Input);
% N input samples and m-1 inputs

% vector of the length m for storing weights
Weights = zeros(1, m);

if (Start == 1)
    if m ~= length(W)
        disp('Starting weighting vector and the number of inputs do not match');
    else
        Weights = W;
    end
else
    % randomization of the random numbers generator based on the current
    % time
    rng('shuffle');
    % Starting weights are generated as random nubbers from [-0.5, 0.5]
    Weights = rand(1, m) - 0.5;
    
end

% Shifting of all columns of Input by 1 to the right to release the 1st
% column
Input(:,2:m+1) = Input(:, 1:m);

% Generation of the column of ones
X0 = ones(N, 1);

% Aooending a column of ones as the 1st column to Input
Input(:, 1) = X0;

% Calculation of reciprocal inputs (last column of Input (m+1) contains
% desired outputs and should not be targeted)
Input_1(:, 1 : m) = Input(:, 1 : m).^-1;

% f - a vector of desired outputs
f = Input(:,m+1);
% Input will contain now only inputs, without desired outputs
Input = Input(:, 1:m);

%learningRate is equal to 1/(number of weights)
learningRate = 1/(N+1);

% learning is a flag (true - a neuron has to learn; false - learning
% finished
learning = true;
%counter of iterations
iterations = 0;
accumulatedRMSE=[];
while (learning == true)
    % flipping a flag at the beginning of every iteration
    learning = false;
    iterations = iterations + 1;
    
    % weighted sums for all learning samples, Input is taken transposed to
    % calculate all of them simultaneously (each row of input is a sample)
    Z = Weights * Input';
%     for i=1:height(Input)
%         Z(i)=sum(Weights.*Input(i,:));
%     end
    % actual outputs for all samples
    Y = tanh(Z);
    %MSE
    MSE = sum((f - Y').^2)/N;
    %RMSE
    RMSE = sqrt(MSE);
    accumulatedRMSE = [accumulatedRMSE;RMSE];
    if (RMSE <= eps)
        figure();
        plot(accumulatedRMSE);
        break  % exit a loop
    else
        learning = true; % has to learn
    end

    
    % A loop over all learning samples
    for j = 1 : N
        % z = w0 + w1*x1 +...+ wn1*xn - weighted sum
        z = dot(Input(j, :), Weights);
        % actual output = activation function of the weighted sum
        y = tanh(z);
        if (y ~= f(j))  % if actual output ~= desired output
            learning = true; % then flip learning flag 
            error = f(j) - y; % calculate the error
            % adjust the weights
            Weights = Weights + learningRate * error * Input_1(j, :)/m;
        end
    end
end

end

