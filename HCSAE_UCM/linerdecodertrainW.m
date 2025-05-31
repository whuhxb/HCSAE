function theta=linerdecodertrainW(visibleSize ,hiddenSize,patches,stepsize,numpatches, sparsityParam)
%stepsize是100,1000或者10000等，用于dropout。
% sparsityParam = 0.035; % 0.035 desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter
beta = 5;            
theta = initializeParameters(hiddenSize, visibleSize);
options.Method = 'lbfgs';
options.maxIter = 30;
options.display = 'on';

% Use minFunc to minimize the function
%matlabpool('open','local',2);   
for i=1:10  
        for j=1:numpatches/stepsize
                fprintf(1,'\nTraining : %d  Iteration',i);
                fprintf(1,'\nTraining : %d  Minipatch\n',j);
dropv=randperm(visibleSize);
visi1=fix(size(patches,1)*0.8);

drop=randperm(hiddenSize);
hidden1=hiddenSize/2;

W = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% displayColorNetwork(W22);

WD=W(drop(1:hidden1),dropv(1:visi1));
WD2=W2(dropv(1:visi1),drop(1:hidden1));
bD=b(drop(1:hidden1),:);
bD2=b2(dropv(1:visi1),:);

aaaa1=reshape(WD,hidden1*visi1,1);
aaaa2=reshape(WD2,hidden1*visi1,1);
theta2=[aaaa1 ;aaaa2 ;bD ;bD2];

[optTheta, cost] = minFunc( @(p) sparseAutoencoderLinearCost(p, ...
                                   visi1, hidden1, ...
                                   lambda, sparsityParam, ...
                                   beta, patches(dropv(1:visi1),1+stepsize*(j-1):stepsize*j)), theta2, options);
                              
theta2=optTheta;

WD = reshape(theta2(1:hidden1*visi1), hidden1, visi1);
WD2 = reshape(theta2(hidden1*visi1+1:2*hidden1*visi1), visi1, hidden1);                          
bD = theta2(2*hidden1*visi1+1:2*hidden1*visi1+hidden1);
bD2 = theta2(2*hidden1*visi1+hidden1+1:end);                          
                          
W(drop(1:hidden1),dropv(1:visi1))=WD;
W2(dropv(1:visi1),drop(1:hidden1))=WD2;
b(drop(1:hidden1),:)=bD;
b2(dropv(1:visi1),:)=bD2;
drawnow;
theta=[reshape(W,hiddenSize*visibleSize,1) ;reshape(W2,hiddenSize*visibleSize,1) ;b ;b2];

        end                                       
end
%matlabpool close;

end

% Use minFunc to minimize the function
% for i=1:10
%         for j=1:numpatches/100000
%                 fprintf(1,'\nTraining : %d  Iteration',i);
%                 fprintf(1,'\nTraining : %d  patch\n',j);
% clear('options');
% options.Method = 'lbfgs';
% options.maxIter = 20;
% options.display = 'on';
% 
% dropv=randperm(visibleSize);
% % visi1=240;
% 
% drop=randperm(hiddenSize);
% % hidden1=400;
% 
% W = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
% W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
% b = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
% b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
% 
% % displayColorNetwork(W22);
% 
% WD=W(drop(1:hidden1),dropv(1:visi1));
% WD2=W2(dropv(1:visi1),drop(1:hidden1));
% bD=b(drop(1:hidden1),:);
% bD2=b2(dropv(1:visi1),:);
% 
% aaaa1=reshape(WD,hidden1*visi1,1);
% aaaa2=reshape(WD2,hidden1*visi1,1);
% theta2=[aaaa1 ;aaaa2 ;bD ;bD2];
% 
% [optTheta, cost] = minFunc( @(p) sparseAutoencoderLinearCost(p, ...
%                                    visi1, hidden1, ...
%                                    lambda, sparsityParam, ...
%                                    beta, patches(dropv(1:visi1),1+100000*(j-1):100000*j)), ...
%                               theta2, options);
%                           theta2=optTheta;
% 
% WD = reshape(theta2(1:hidden1*visi1), hidden1, visi1);
% WD2 = reshape(theta2(hidden1*visi1+1:2*hidden1*visi1), visi1, hidden1);                          
% bD = theta2(2*hidden1*visi1+1:2*hidden1*visi1+hidden1);
% bD2 = theta2(2*hidden1*visi1+hidden1+1:end);                          
%                           
% W(drop(1:hidden1),dropv(1:visi1))=WD;
% W2(dropv(1:visi1),drop(1:hidden1))=WD2;
% b(drop(1:hidden1),:)=bD;
% b2(dropv(1:visi1),:)=bD2;
% displayColorNetwork(W');
% drawnow;
% theta=[reshape(W,hiddenSize*visibleSize,1) ;reshape(W2,hiddenSize*visibleSize,1) ;b ;b2];
% 
%         end                                       
% end
% save W.mat W;
% clear patches;