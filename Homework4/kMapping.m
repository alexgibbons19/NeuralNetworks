function [ kMappedInputs ] = kMapping(Input)
[N,m] = size(Input);
k = max(max(Input))+1;
subInt = 2/k;
kMappedInputs = zeros(N,m);
for j=1:N
    for i=1:m
        kMappedInputs(j,i)=Input(j,i)*subInt+(subInt/2)-1;
    end
end
