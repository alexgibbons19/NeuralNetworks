clear;
x1 = [0,1,2];
x2 = [2,2,1];
x3 = [1,3,3];

D = [0,1,2];


Input = [x1;x2;x3;D]';
kMappedInputs = kMapping(Input);
[N,m] = size(Input);
k = max(max(Input))+1;
tol = 1/(3*k);

weights = zeros(m);
allIters = zeros(3,1);
actualOutputs = zeros(3,N);
z=zeros(3,N);
tol = zeros(3,1);
for j=1:3
    tol(j,1) = 1/(((j+1)^(j+1))*k);
    [weights, allIters(j,1)] = errorCorrectionRmse(kMappedInputs,tol(j),0);
    w0=weights(1);
    fixedWeights=weights(2:end);
    for i=1:N
        z(j,i) = sum(fixedWeights.*kMappedInputs(i,1:m-1));
        z(j,i)= z(j,i)+w0;
    end
    actualOutputs(j,:)=unmapInputs(z(j,:));
end
disp('Z values: ');
disp(z);
disp('Actual output: ');
disp(actualOutputs);
disp(['Desired output: ' num2str(D)]);
disp('Tolerances: ');
disp(tol);
disp('Iterations: '); 
disp(allIters);



