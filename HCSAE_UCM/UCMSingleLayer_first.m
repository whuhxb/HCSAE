% function linerdecoderUCM()
%% STEP 0: Initialization the first hidden layer
fprintf('STEP 0: Initialization the first hidden layer\n');
imageChannels = 3;     % number of channels (rgb, so 3)
patchDim = 10;          % patch dimension

visibleSizeL1 = patchDim * patchDim * imageChannels;  % number of input units
outputSizeL1 = visibleSizeL1;   % number of output units
%hiddenSizeL1 = 600;           % number of hidden units
sparsityParam = 0.5; % 0.035 desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter
beta = 5;              % weight of sparsity penalty term
epsilon = 0.1;	       % epsilon for ZCA whitening

HDL1=[700,800,900,1000,1100];
l=length(HDL1);
accuracy=[];
for i=1:l
hiddenSizeL1=HDL1(i);
UCMSingleLayer_second;
accuracy=[accuracy,acc2];
end

save accuracy accuracy;