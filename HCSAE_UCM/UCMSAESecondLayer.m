%% STEP 0: Initialization the first hidden layer
fprintf('STEP 0: Initialization the first hidden layer\n');
imageChannels = 3;     % number of channels (rgb, so 3)
patchDim = 10;         % patch dimension
visibleSizeL1 = patchDim * patchDim * imageChannels;  % number of input units
outputSizeL1 = visibleSizeL1;   % number of output units
hiddenSizeL1 = 1200;           % number of hidden units
sparsityParam = 0.5; % 0.035 desired average activation of the hidden units.
epsilon = 0.1;	       % epsilon for ZCA whitening
% load pooled feature from the first CSAE layer to train the SAE feature
% for the second CSAE layer
load pooledFeaturesL1.mat;
pooledFeaturesL1=permute(pooledFeaturesL1, [3 4 1 2]);
pooledFeaturesL1=reshape(pooledFeaturesL1,4*4*1200,100,21);
pooledFeaturesL12=reshape(pooledFeaturesL1,4,4,1200,100*21);
picturenum=size(pooledFeaturesL12,4);
%随机选取onepicnum个patch
patchDim1=3;
onepicnum=4;
numpatches1=picturenum*onepicnum;
patches2=zeros(patchDim1*patchDim1*hiddenSizeL1,numpatches1);
 for imageNum = 1:picturenum
     [rowNum colNum] = size(pooledFeaturesL12(:,:,1,imageNum));%这里是4*4
     for patchNum = 1:onepicnum%从每个patch中选取4个小块
         xPos = randi([1,rowNum-patchDim1+1]);%随机获取一个坐标，这里randi([a,b])表示随机取一个范围为[a,b]的整数
         yPos = randi([1,colNum-patchDim1+1]);
         temp=pooledFeaturesL12(xPos:xPos+patchDim1-1,yPos:yPos+patchDim1-1,:,imageNum);
         reshapetemp=reshape(temp,patchDim1*patchDim1*hiddenSizeL1,1);
         patches2(:,(imageNum-1)*onepicnum+patchNum) =reshapetemp;
     end
 end
clear temp reshapetemp clear pooledFeaturesL1 pooledFeaturesL11 pooledFeaturesL12;
patches2=normalizeData1(patches2);
patches2=ZCAwhitenData1(patches2);
save('patches2','patches2','-v7.3');
clear patches2;

hiddenSizeL2 = 3*3*hiddenSizeL1;
hiddenSizeL3 = 2400;
sparsityParam1=0.7;
lambda1=3e-3;
beta1=5;
load patches2;
theta12=linerdecodertrainW2(hiddenSizeL2 ,hiddenSizeL3,patches2,200,numpatches1,sparsityParam1,lambda1,beta1);
WL2=reshape(theta12(1:hiddenSizeL3*hiddenSizeL2), hiddenSizeL3, hiddenSizeL2);
bL2=theta12(2*hiddenSizeL3*hiddenSizeL2+1:2*hiddenSizeL3*hiddenSizeL2+hiddenSizeL3);
displayColorNetwork(WL2(:,1:3*patchDim1*patchDim1)');
print -djpeg weights2.jpg;
save WL2.mat WL2;
save bL2.mat bL2;
clear patches2 WL2 bL2;

load ZCAWhite1;
load WL2.mat;
load bL2.mat;
load pooledFeaturesL1.mat;
meanPatch1=zeros(hiddenSizeL2,1);
imageT1=4;
patchDim1=3;
pp4=2;
bbb=(imageT1-patchDim1+1)/pp4;  %image size after conv
PN=2100;
pooledFeaturesL2=zeros(hiddenSizeL3,PN,bbb,bbb);
psize=4;
pooledFeaturesL1=permute(pooledFeaturesL1,[3 4 1 2]);
[im1,in1,imagechanne1l]=size(pooledFeaturesL1(:,:,:,1));

tic;
for jj=1:PN
    image=pooledFeaturesL1(:,:,:,jj);
    convlocation=myconvlocation(image,patchDim1);
    testX=(image(convlocation));% 0.14s
    testX=ZCAWhite1*testX;% 0.20s
    Z=sigmoid(WL2*testX+repmat( bL2,[1 size(testX,2)])  );% 1.2s  
    Z=Z';
    Z=reshape(Z,im1 - patchDim1 + 1,in1 - patchDim1 + 1,size(WL2,1));% 0.23s
    Z1=permute(Z,[3 1 2]);
    %Z在这里表示的是卷积之后的结果，还没有进行pooling
    pooledFeaturesThis=cnnPool(pp4,Z1);
    pooledFeaturesL2(:,jj,:,:)=pooledFeaturesThis;
    jj
end
toc;

clear WL2 bL2;
clear ZCAWhite1;
clear image convlocation testX Z Z1 pooledFeaturesThis pooledFeaturesL1;
save pooledFeaturesL2.mat pooledFeaturesL2;
tempfeature = permute(pooledFeaturesL2, [3 1 2 4]);
featureL1 = reshape(tempfeature, numel(pooledFeaturesL2) / PN,PN); %numel(pooledFeaturesTrain) / numTrainImages            
clear tempfeature pooledFeaturesL2;
save featureL1.mat featureL1;

%%  classification
fprintf('classification\n');
clear;
load featureL1;    
softmaxLambda = 1e-4;
numClasses = 21;
numTrainImages=2100;
featureL2=reshape(featureL1,numel(featureL1)/numTrainImages,100,21);
% two seperation methods: 1. using the fixed random number for the first and second layer
load A;
%A=cell(21,1);
for i=1:21
randnum=A{i};
pooledFeaturesTrain(:,:,i)=featureL2(:,randnum(1:80),i);
pooledFeaturesTest(:,:,i)=featureL2(:,randnum(81:100),i);
end
%save A A;
% % 2. using the different random number for the first and second layer
% %similar to initially seperate different training set and testing set
% for i=1:21
% randtemp1=randperm(100);
% pooledFeaturesTrain(:,:,i)=featureL2(:,randtemp1(1:80),i);
% pooledFeaturesTest(:,:,i)=featureL2(:,randtemp1(81:100),i);
% end
numTrain=1680;
numTest=420;
pooledFeaturesTrain=reshape(pooledFeaturesTrain,numel(pooledFeaturesTrain)/numTrain,numTrain);
pooledFeaturesTest=reshape(pooledFeaturesTest,numel(pooledFeaturesTest)/numTest,numTest);
codeI=featureL1;
softmaxX=codeI;  
load label.mat;

numClasses=21;
options = struct;
options.maxIter = 400;
softmaxModel = softmaxTrain(numel(codeI) / numTrainImages,numClasses, softmaxLambda,pooledFeaturesTrain, trainLabels, options);%                            
[pred] = softmaxPredict(softmaxModel, pooledFeaturesTest);
save pred pred;
acc1 = (pred(:) == testLabels(:));
acc3 = sum(acc1) / size(acc1, 1);
save acc3 acc3;
fprintf('Accuracy: %2.3f%%\n', acc3 * 100);

