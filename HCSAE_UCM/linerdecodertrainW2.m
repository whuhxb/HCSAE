function theta12=linerdecodertrainW1(hiddenSizeL2 ,hiddenSizeL3,patches2,stepsize,numpatches, sparsityParam1,lambda1,beta1)

theta12 = initializeParameters(hiddenSizeL3, hiddenSizeL2);
addpath minFunc/;

for k=1:10
    for l=1:numpatches/stepsize
    fprintf(1,'\nTraining : %d  Iteration',k);
    fprintf(1,'\nTraining : %d  Minibatch',l);
    clear('options');
options.Method = 'lbfgs';
options.maxIter = 30;	 
options.display = 'on';

dropv2=randperm(hiddenSizeL2);
visi12=fix(size(patches2,1)*1);

drop2=randperm(hiddenSizeL3);
hidden12=hiddenSizeL3/2;

WO2 = reshape(theta12(1:hiddenSizeL3*hiddenSizeL2), hiddenSizeL3, hiddenSizeL2);
W22 = reshape(theta12(hiddenSizeL3*hiddenSizeL2+1:2*hiddenSizeL3*hiddenSizeL2), hiddenSizeL2, hiddenSizeL3);
bO2 = theta12(2*hiddenSizeL3*hiddenSizeL2+1:2*hiddenSizeL3*hiddenSizeL2+hiddenSizeL3);
b22 = theta12(2*hiddenSizeL3*hiddenSizeL2+hiddenSizeL3+1:end);

WDO2=WO2(drop2(1:hidden12),dropv2(1:visi12));
WD22=W22(dropv2(1:visi12),drop2(1:hidden12));
bD2=bO2(drop2(1:hidden12),:);
bD22=b22(dropv2(1:visi12),:);

aaaa12=reshape(WDO2,hidden12*visi12,1);
aaaa22=reshape(WD22,hidden12*visi12,1);
theta22=[aaaa12 ;aaaa22 ;bD2 ;bD22];

[sae2OptTheta, cost2] = minFunc( @(p) sparseAutoencoderLinearCost(p,visi12, hidden12,lambda1, sparsityParam1, ...
                                   beta1, patches2(dropv2(1:visi12),1+stepsize*(l-1):stepsize*l)),theta22, options);
                          theta22=sae2OptTheta;

WDO2 = reshape(theta22(1:hidden12*visi12), hidden12, visi12);
WD22 = reshape(theta22(hidden12*visi12+1:2*hidden12*visi12), visi12, hidden12);                          
bD2 = theta22(2*hidden12*visi12+1:2*hidden12*visi12+hidden12);
bD22 = theta22(2*hidden12*visi12+hidden12+1:end);                          
                          
WO2(drop2(1:hidden12),dropv2(1:visi12))=WDO2;
W22(dropv2(1:visi12),drop2(1:hidden12))=WD22;
bO2(drop2(1:hidden12),:)=bD2;
b22(dropv2(1:visi12),:)=bD22;
theta12=[reshape(WO2,hiddenSizeL3*hiddenSizeL2,1) ;reshape(W22,hiddenSizeL3*hiddenSizeL2,1) ;bO2 ;b22];

    end
end

save('theta12','theta12','-v7.3');
end