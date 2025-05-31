%% STEP 0: Initialization the first hidden layer
fprintf('STEP 0: Initialization the first hidden layer\n');
imageChannels = 3;   % number of channels (rgb, so 3)
patchDim = 10;       % patch dimension
%% GET the first layer train patches
fprintf('GET the first layer train patches\n');
load colorIMAGES;    % load images from disk
% picturenum=size(colorIMAGES,4);%表示train的时候，一共有多少图片
IMAGES=reshape(colorIMAGES,256*256*imageChannels ,100,21);
IMAGES2=reshape(IMAGES,256,256,3,100*21);
picturenum=size(IMAGES2,4);
patchonepic=1000;%表示一幅图取多少patch
numpatches = picturenum*patchonepic;
patches = zeros(patchDim*patchDim*imageChannels, numpatches);
%随机选取patchonepic个patch
for imageNum = 1:picturenum
    [rowNum colNum] = size(IMAGES2(:,:,1,imageNum));%这里是256*256
    for patchNum = 1:patchonepic
        xPos = randi([1,rowNum-patchDim+1]);%随机获取一个坐标，这里randi([a,b])表示随机取一个范围为[a,b]的整数
        yPos = randi([1,colNum-patchDim+1]);
        temp=IMAGES2(xPos:xPos+patchDim-1,yPos:yPos+patchDim-1,:,imageNum);
        reshapetemp=reshape(temp,patchDim*patchDim*3,1);
        patches(:,(imageNum-1)*patchonepic+patchNum) =reshapetemp;
    end
end
clear rowNum colNum xPos yPos temp reshapetemp;
clear colorIMAGES IMAGES IMAGES1 IMAGES2;
visibleSizeL1 = patchDim * patchDim * imageChannels;  % number of input units
hiddenSizeL1 = 1200;           % number of hidden units
sparsityParam = 0.5; % 0.035 desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter
beta = 5;              % weight of sparsity penalty term
epsilon = 0.1;	       % epsilon for ZCA whitening
%% preprocessing the train data
fprintf('prepocessing the train data\n');
patches=normalizeData(patches);
patches=ZCAwhitenData(patches);%在这里保存了针对训练数据得到的ZCAWhite矩阵
save('patches.mat', 'patches','-v7.3');
clear patches ZCAWhite;
%% train W
fprintf('train W\n');
addpath minFunc/;
load patches;
thetaL1=linerdecodertrainW(visibleSizeL1,hiddenSizeL1,patches,10000,numpatches, sparsityParam);
WL1 = reshape(thetaL1(1:hiddenSizeL1*visibleSizeL1), hiddenSizeL1, visibleSizeL1);
bL1= thetaL1(2*hiddenSizeL1*visibleSizeL1+1:2*hiddenSizeL1*visibleSizeL1+hiddenSizeL1);
save WL1.mat WL1;
save bL1.mat bL1;
displayColorNetwork(WL1');
print -djpeg weights.jpg;
clear WL1 bL1 patches sparsityParam lambda beta epsilon;
%% convolution and pool  data
fprintf('convolution and pool  data\n');
poolDim =61;          % dimension of pooling region
imageDim=256;
pp4=61;
bbb=floor((imageDim-patchDim+1)/poolDim);

numTrainImages=2100;
PN=2100;
load ZCAWhite;
load colorIMAGES;
load WL1;
load bL1;
meanPatch=zeros(visibleSizeL1,1);
trainImages=colorIMAGES*0.8+0.1;%把输入图像归一到0.1-0.9，这是为了保证训练和编码的输入是一样的。
clear colorIMAGES;
pooledFeaturesL1 = zeros(hiddenSizeL1,PN,bbb,bbb);
[im1,in1,imagechanne1l]=size(trainImages(:,:,:,1));
tic;
for jj=1:PN
    image=trainImages(:,:,:,jj);
    convlocation=myconvlocation(image,patchDim);
    testX=(image(convlocation));% 0.14s
    testX=ZCAWhite*testX;% 0.20s
    Z=sigmoid(  WL1*testX+repmat( bL1,[1 size(testX,2)])  );% 1.2s  
    Z=Z';
    Z=reshape(Z,im1 - patchDim + 1,in1 - patchDim + 1,size(WL1,1));% 0.23s
    Z1=permute(Z,[3 1 2]);
    %Z在这里表示的是卷积之后的结果，还没有进行pooling
    pooledFeaturesThis=cnnPool(pp4,Z1);
    pooledFeaturesL1(:,jj,:,:)=pooledFeaturesThis;
    jj
end
toc;
clear ZCAWhite bL1 image convlocation testX Z Z1 pooledFeaturesThis trainImages;
clear poolDim imageDim patchDim pp4 bbb im1 in1 imagechannel;
save('pooledFeaturesL1.mat', 'pooledFeaturesL1');
tempfeature = permute(pooledFeaturesL1, [1 3 4 2]);%permute,原先pooledfeature是400*2100*3*3的，现在改为400*3*3*2100
featureL1 = reshape(tempfeature,numel(pooledFeaturesL1)/numTrainImages,numTrainImages); %numel(pooledFeaturesTrain) / numTrainImages            
clear tempfeature;
save featureL1.mat featureL1;

%%  classification
fprintf('classification\n');
clear;
load featureL1;    
softmaxLambda = 1e-4;
numClasses = 21;
numTrainImages=2100;
featureL2=reshape(featureL1,numel(featureL1)/numTrainImages,100,21);
% Generate fixed random number
A=cell(21,1);
for i=1:21
randnum=randperm(100);
A{i}=randnum;
end
save A A;
% Two seperation methods. 1. Using the fixed random number to train the first and second layer
load A;
for i=1:21
randnum=A{i};
pooledFeaturesTrain(:,:,i)=featureL2(:,randnum(1:80),i);
pooledFeaturesTest(:,:,i)=featureL2(:,randnum(81:100),i);
end
% % 2. Using the random number to train the first and second layer
% for i=1:21
% randnum1=randperm(100);
% pooledFeaturesTrain(:,:,i)=featureL2(:,randnum1(1:80),i);
% pooledFeaturesTest(:,:,i)=featureL2(:,randnum1(81:100),i);
% end
clear randnum1 A featureL2;
numTrain=1680;
numTest=420;
pooledFeaturesTrain=reshape(pooledFeaturesTrain,numel(pooledFeaturesTrain)/numTrain,numTrain);
pooledFeaturesTest=reshape(pooledFeaturesTest,numel(pooledFeaturesTest)/numTest,numTest);
codeI=featureL1;
softmaxX=codeI;  
load label.mat;
clear featureL1;

numClasses=21;
options = struct;
options.maxIter = 400;
softmaxModel = softmaxTrain(numel(codeI) / numTrainImages,numClasses, softmaxLambda,pooledFeaturesTrain, trainLabels, options);%                            
[pred] = softmaxPredict(softmaxModel, pooledFeaturesTest);
acc2 = (pred(:) == testLabels(:));
acc2 = sum(acc2) / size(acc2, 1);
fprintf('Accuracy: %2.3f%%\n', acc2 * 100);
save acc2 acc2;

