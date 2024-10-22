
%Read weights for a 36-7 network, stored in a *.wgt file

function [hidneur_weights, outneur_weights] = ReadWeights(fname)

vars_num = 18;
hidneur_num = 36;
outneur_num = 7;

hidneur_weights = zeros(vars_num+1, 36);
outneur_weights = zeros(hidneur_num+1, 7);

f = fopen(fname);
data_read = fread(f, 'double');
fclose(f);

data_ind = 1;

%Hidden neurons
for hh = 1 : hidneur_num
    
    for ww = 1 : vars_num+1
        
        hidneur_weights(ww, hh) = data_read(data_ind) + 1i * data_read(data_ind+1);
        data_ind = data_ind + 2;
    end
end

%Output neurons
for pp = 1 : outneur_num
    
    for ww = 1 : hidneur_num + 1
        
        outneur_weights(ww, pp) = data_read(data_ind) + 1i * data_read(data_ind+1);
        data_ind = data_ind + 2;
    end
end

