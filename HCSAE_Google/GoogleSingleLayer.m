%% STEP 0: Initialization the first hidden layer
fprintf('STEP 0: Initialization the first hidden layer\n');
imageChannels = 3;     % number of channels (rgb, so 3)
patchDim = 10;         % patch dimension
%% GET the first layer train patches
fprintf('GET the first layer train patches\n');
load IMAGES_Google.mat;    % load images from disk 
IMAGES2=reshape(IMAGES,200,200,imageChannels,200*12);
picturenum=size(IMAGES2,4);
patchonepic=1000;%表示一幅图取多少patch
numpatches=picturenum*patchonepic;
patches = zeros(patchDim*patchDim*imageChannels,numpatches);
%随机选取patchonepic个patch
for imageNum = 1:picturenum
    [rowNum colNum] = size(IMAGES2(:,:,1,imageNum));%这里是256*256
    for patchNum = 1:patchonepic
        xPos = randi([1,rowNum-patchDim+1]);%随机获取一个坐标，这里randi([a,b])表示随机取一个范围为[a,b]的整数
        yPos = randi([1, colNum-patchDim+1]);
        temp=IMAGES2(xPos:xPos+patchDim-1,yPos:yPos+patchDim-1,:,imageNum);
        reshapetemp=reshape(temp,patchDim*patchDim*3,1);
        patches(:,(imageNum-1)*patchonepic+patchNum) =reshapetemp;
    end
end
clear rowNum colNum xPos yPos temp reshapetemp IMAGES2;
clear  picturenum patchonepic;
visibleSizeL1 = patchDim * patchDim * imageChannels;  % number of input units
hiddenSizeL1 = 1000;           % number of hidden units
sparsityParam = 0.7; % 0.035 desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter
beta = 5;              % weight of sparsity penalty term
epsilon = 0.1;	       % epsilon for ZCA whitening
%% prepocessing the train data
fprintf('preprocessing the train data\n');
patches=normalizeData(patches);
patches=ZCAwhitenData(patches);%在这里保存了针对训练数据得到的ZCAWhite矩阵
save('patches.mat', 'patches','-v7.3');
clear patches ZCAWhite;
%% train W
fprintf('train W\n');
addpath minFunc/;
load patches;
theta2=linerdecodertrainW(visibleSizeL1,hiddenSizeL1,patches,10000,numpatches, sparsityParam);
WL1 = reshape(theta2(1:hiddenSizeL1*visibleSizeL1), hiddenSizeL1, visibleSizeL1);
bL1 = theta2(2*hiddenSizeL1*visibleSizeL1+1:2*hiddenSizeL1*visibleSizeL1+hiddenSizeL1);
%displayColorNetwork(WL2');
save WL1.mat WL1;
save bL1.mat bL1;
displayColorNetwork(WL1');
print -djpeg weights.jpg;
clear WL1 bL1 patches sparsityParam lambda beta epsilon;

%% convolution and pooling data
fprintf('convolution and pooling data\n');
poolDim =47;          
imageDim=200;
patchDim=10;

numTrainImages=2400;
load ZCAWhite;
load IMAGES;
load WL1;
load bL1;
trainImages=IMAGES*0.8+0.1;%把输入图像归一到0.1-0.9，这是为了保证训练和编码的输入是一样的。
clear IMAGES;
[im,in,imagechannel]=size(trainImages(:,:,:,1));
pp3=47;
bbb1=floor((imageDim-patchDim+1)/poolDim);
pooledFeaturesL1 = zeros(hiddenSizeL1,numTrainImages,bbb1,bbb1);%%

tic;
for i=1:numTrainImages 
    image=trainImages(:,:,:,i);
    convlocation=myconvlocation(image,patchDim);
    testX=(image(convlocation));% 0.14s
    testX=ZCAWhite*testX;% 0.20s
    Z=sigmoid(WL1*testX+repmat( bL1,[1 size(testX,2)])  );% 1.2s  
    Z=Z';
    Z=reshape(Z,im-patchDim+1,in-patchDim+1,size(WL1,1));% 0.23s
    %Z在这里表示的是卷积之后的结果，还没有进行pooling
    Z1=permute(Z,[3 1 2]);
    pooledFeaturesThis=cnnPool(pp3,Z1);
    pooledFeaturesL1(:,i,:,:)=pooledFeaturesThis;
    i
end
toc;
clear ZCAWhite bL1 image convlocation testX Z Z1 pooledFeaturesThis trainImages;
clear poolDim imageDim patchDim pp3 bb1 im in imagechannel;
fprintf('The first layer convolution and pooling done!');
pooledFeaturesL1=permute(pooledFeaturesL1,[3 4 1 2]);%4*4*hiddenSizeL1*PN
save pooledFeaturesL1.mat pooledFeaturesL1;
PN=2400;
tempfeature = permute(pooledFeaturesL1, [3 1 2 4]);
featureL1 = reshape(tempfeature, numel(pooledFeaturesL1) / PN,PN); %numel(pooledFeaturesTrain) / numTrainImages            
clear tempfeature;
save featureL1.mat featureL1;
clear pooledFeaturesL1 featureL1;

load featureL1;    
softmaxLambda = 1e-4;
numClasses = 12;
featureL2=reshape(featureL1,numel(featureL1)/numTrainImages,200,12);
clear pooledFeaturesTrain pooledFeaturesTest;
% Generate random number, can fixed for training the first and second layer 
randnum=cell(12);
for i=1:12
randtemp=randperm(200);
randnum{i}=randtemp;
end
save randnum randnum;
% two seperation methods: 1. using the fixed random number for the first and second layer
load randnum;
for i=1:12
randtemp1=randnum{i};
pooledFeaturesTrain(:,:,i)=featureL2(:,randtemp1(1:160),i);
pooledFeaturesTest(:,:,i)=featureL2(:,randtemp1(161:200),i);
end
% % 2. using the different random number for the first and second layer
% for i=1:12
% randnum1=randperm(200);
% %randnum1=A{1};
% pooledFeaturesTrain(:,:,i)=featureL2(:,randnum1(1:160),i);
% pooledFeaturesTest(:,:,i)=featureL2(:,randnum1(161:200),i);
% end
clear randnum1 featureL2;
numTrain=1920;
numTest=480;
pooledFeaturesTrain=reshape(pooledFeaturesTrain,numel(pooledFeaturesTrain)/numTrain,numTrain);
pooledFeaturesTest=reshape(pooledFeaturesTest,numel(pooledFeaturesTest)/numTest,numTest);
codeI=featureL1;
load label.mat;
clear featureL1;
numClasses=12;
options = struct;
options.maxIter = 400;
softmaxModel = softmaxTrain(numel(codeI) / numTrainImages,numClasses, softmaxLambda,pooledFeaturesTrain, trainLabels, options);%                            
[pred] = softmaxPredict(softmaxModel, pooledFeaturesTest);
acc2 = (pred(:) == testLabels(:));
acc2 = sum(acc2) / size(acc2, 1);
fprintf('Accuracy: %2.3f%%\n', acc2 * 100);
save acc2 acc2;
save pred1 pred;