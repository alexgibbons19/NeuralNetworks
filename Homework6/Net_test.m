
function classif_rate = Net_test(Input, hidneur_weights, outneur_weights, win_ang)
% (c) E.Aizenberg & I.Aizenberg 2014-2018

% Input -   matrix N x m of testing samples. 
%           There are N testing samples in total
%           m is the number of columns in Input
%           Each sample consists of n = m-1 inputs followed by the desired
%           output in the m-th column. 
%           A desired output must be an integer - a class label (from 0 to
%           k-1 for k classes) 
%
%           Network inputs must be angular values in radians. It is assumed
%           that a classification problem to be solved has continuous
%           inputs. Hence, inputs must be transformed in advance in order
%           to fit them in the interval [0, fi] where fi<2pi radians.

% hidneur_weights -     Matrix of weights of all hidden neurons
% outneur_weights -     Matrix of weights of the output neuron
% win_ang   - a winning angle determining a bisector of the desired sector.
%             For example, if an output neuron has a 2-valued discrete
%             activation function and a sector 0 is a desired sector for a
%             correct output, then win_ang=pi/2. 
%             If an output neuron has a 2-valued discrete activation
%             function and a sector 1 is a desired sector for output, then
%             win_ang=3pi/2  

% If more than one output neuron recognized a testing sample as "its", the
% "winner takes it all" technique is used to determine the output.
% A neuron whose output is closer to the bisector of the desired sector is
% considered the winner

% Determine the number of learning samples (N) and the number of columns in
% the matrix Input containing inputs and desired output
[N, m] = size(Input); 
%Determine the number of network inputs (input variables n)
n = m-1; % n = # of inputs
%X = matrix of MVN inputs (N x n), where N=number of learning
%samples, n = number of input variables
X = Input(:,1:n); % X is a matrix of inputs
%y_d = (N x 1) vector of desired network outputs, expressed as class labels
y_d = Input(:,m); % y_d is a vector-column of desired outputs

%Determine the number of output neurons
outneur_num = size(outneur_weights, 2);

%Convert input values into complex numbers on the unit circle
X = exp(1i .* X);

%append a column of 1s to X from the left
%app_X
col_app(1:N) = 1;
col_app = col_app.';
app_X = [col_app X];

%Compute the output of hidden neurons for all samples
hid_outmat = app_X * hidneur_weights;

%Move outputs to the unit circle
hid_outmat = hid_outmat ./ abs(hid_outmat);

%append a column of 1s to hid_outmat
hid_outmat = [col_app hid_outmat];

%Compute the output of the network
z_outneur = hid_outmat * outneur_weights;

%We will now apply the "winner take it all" principle in the following
%manner: the output neuron whose output is closest to pi/2 determines the
%output class
%win_ang = pi/2;
current_phase = angle(z_outneur);
win_dist = zeros(N, outneur_num);
for ii=1:N
    
    for pp = 1 : outneur_num
       
        win_dist(ii, pp) = abs(current_phase(ii, pp) - win_ang);

        if (win_dist(ii, pp) > pi)

            win_dist(ii, pp) = 2*pi - win_dist(ii, pp);
        end

    end
end

% calculation of the network output based on the "winner takes it all"
% technique based on the closeness of the output to the bisector of the
% desired sector
[min_dist, current_labels] = min(win_dist, [], 2);
current_labels = current_labels - 1;

sovpad = 0;

for ii=1:N
    if (current_labels(ii) == y_d(ii))
        sovpad = sovpad + 1;
    end
end

classif_rate = sovpad / N;
