%% STEP 0: Initialization the first hidden layer
fprintf('STEP 0: Initialization the first hidden layer\n');
imageChannels = 3;     % number of channels (rgb, so 3)
patchDim = 10;         % patch dimension
visibleSizeL1 = patchDim * patchDim * imageChannels;  % number of input units
outputSizeL1 = visibleSizeL1;   % number of output units
hiddenSizeL1 = 1000;           % number of hidden units
sparsityParam = 0.5; % 0.035 desired average activation of the hidden units.
lambda = 3e-3;         % weight decay parameter
beta = 5;              % weight of sparsity penalty term
epsilon = 0.1;	       % epsilon for ZCA whitening
%% GET the first layer train patches
fprintf('GET the first layer train patches\n');
load IMAGES;    % load images from disk 
picturenum=size(IMAGES,4);%表示train的时候，一共有多少图片
patchonepic=1000;%表示一幅图取多少patch
numpatches = 2400*patchonepic;
patches = zeros(patchDim*patchDim*3, numpatches);
%随机选取patchonepic个patch
for imageNum = 1:picturenum
    [rowNum colNum] = size(IMAGES(:,:,1,imageNum));%这里是256*256
    for patchNum = 1:patchonepic
        xPos = randi([1,rowNum-patchDim+1]);%随机获取一个坐标，这里randi([a,b])表示随机取一个范围为[a,b]的整数
        yPos = randi([1, colNum-patchDim+1]);
        temp=IMAGES(xPos:xPos+patchDim-1,yPos:yPos+patchDim-1,:,imageNum);
        reshapetemp=reshape(temp,patchDim*patchDim*3,1);
        patches(:,(imageNum-1)*patchonepic+patchNum) =reshapetemp;
    end
end
clear temp reshapetemp;

%% prepocessing the train data
fprintf('preprocessing the train data\n');
patches=normalizeData(patches);
patches=ZCAwhitenData(patches);%在这里保存了针对训练数据得到的ZCAWhite矩阵
save('patches.mat', 'patches','-v7.3');

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
meanPatch=zeros(visibleSizeL1,1);
trainImages=IMAGES*0.8+0.1;%把输入图像归一到0.1-0.9，这是为了保证训练和编码的输入是一样的。
clear IMAGES;
[im,in,imagechannel]=size(trainImages(:,:,:,1));
convolvedFeaturesThis=zeros(size(WL1,1),1,im-patchDim+1,in-patchDim+1);
pooledFeaturesL1 = zeros(hiddenSizeL1, numTrainImages,  floor((imageDim - patchDim + 1) / poolDim),   floor((imageDim - patchDim + 1) / poolDim) );%%
tic;
for i=1:numTrainImages 
    image=trainImages(:,:,:,i);
    [im,in,imagechannel]=size(image);
    convlocation=myconvlocation(image,patchDim);
    testX=(image(convlocation));% 0.14s
    testX=ZCAWhite*testX;% 0.20s
    Z=sigmoid(WL1*testX+repmat( bL1,[1 size(testX,2)])  );% 1.2s  
    Z=Z';
    Z=reshape(Z,im - patchDim + 1,in - patchDim + 1,size(WL1,1));% 0.23s
    %Z在这里表示的是卷积之后的结果，还没有进行pooling
    Z1=permute(Z,[3 1 2]);
    convolvedFeaturesThis(:,i,:,:)=Z1;
    pooledFeaturesL1(:,i,:,:)=cnnPool(poolDim,convolvedFeaturesThis(:,i,:,:));
    i
end
toc;
clear bL1;
clear ZCAWhite;
fprintf('The first layer convolution and pooling done!');
pooledFeaturesL1=permute(pooledFeaturesL1,[3 4 1 2]);%4*4*hiddenSizeL1*PN
save pooledFeaturesL1.mat pooledFeaturesL1;

%随机选取onepcinum个patch
patchDim1=3;
onepicnum=4;
numpatches1=2400*onepicnum;
patches2=zeros(patchDim1*patchDim1*hiddenSizeL1,numpatches1);
for imageNum = 1:2100
    [rowNum colNum] = size(pooledFeaturesL1(:,:,1,imageNum));%这里是4*4
    for patchNum = 1:onepicnum%从每个patch中选取4个小块
        xPos = randi([1,rowNum-patchDim1+1]);%随机获取一个坐标，这里randi([a,b])表示随机取一个范围为[a,b]的整数
        yPos = randi([1,colNum-patchDim1+1]);
        temp=pooledFeaturesL1(xPos:xPos+patchDim1-1,yPos:yPos+patchDim1-1,:,imageNum);
        reshapetemp=reshape(temp,patchDim1*patchDim1*hiddenSizeL1,1);
        patches2(:,(imageNum-1)*onepicnum+patchNum) =reshapetemp;
    end
end
clear temp reshapetemp;
patches2=normalizeData1(patches2);
patches2=ZCAwhitenData1(patches2);
save('patches2','patches2','-v7.3');

hiddenSizeL2 = 3*3*hiddenSizeL1;
hiddenSizeL3 = 2000;
sparsityParam1=0.1;
lambda1=3e-3;
beta1=5;
load patches2;
theta12=linerdecodertrainW2(hiddenSizeL2 ,hiddenSizeL3,patches2,200,numpatches1,sparsityParam1);
WL2=reshape(theta12(1:hiddenSizeL3*hiddenSizeL2), hiddenSizeL3, hiddenSizeL2);
bL2=theta12(2*hiddenSizeL3*hiddenSizeL2+1:2*hiddenSizeL3*hiddenSizeL2+hiddenSizeL3);
%displayColorNetwork(WL2');
save WL2.mat WL2;
save bL2.mat bL2;
clear patches2;

load ZCAWhite1;
meanPatch1=zeros(hiddenSizeL2,1);
imageT1=4;
patchDim1=3;
bbb=imageT1-patchDim1+1;  %image size after conv
PN=2400;
convolvedFeaturesThis1=zeros(size(WL2,1),1,imageT1-patchDim1+1,imageT1-patchDim1+1);
pooledFeaturesL2=zeros(hiddenSizeL3,PN,bbb,bbb);
psize=4;
pp4=1;
tic;
for jj=1:PN
    image=pooledFeaturesL1(:,:,:,jj);
    convlocation=myconvlocation(image,patchDim1);
    testX=(image(convlocation));% 0.14s
    testX=ZCAWhite1*testX;% 0.20s
    Z=sigmoid(  WL2*testX+repmat( bL2,[1 size(testX,2)])  );% 1.2s  
    Z=Z';
    Z=reshape(Z,im - patchDim1 + 1,in - patchDim1 + 1,size(WL2,1));% 0.23s
    Z1=permute(Z,[3 1 2]);
    convolvedFeaturesThis1(:,jj,:,:)=Z1;
    pooledFeaturesL1(:,jj,:,:)=cnnPool(pp4,convolvedFeaturesThis1(:,jj,:,:));
    jj
end
toc;

clear bL2;
clear ZCAWhite1;
save pooledFeaturesL2.mat pooledFeaturesL2;
tempfeature = permute(pooledFeaturesL2, [3 1 2 4]);
featureL1 = reshape(tempfeature, numel(pooledFeaturesL2) / PN,PN); %numel(pooledFeaturesTrain) / numTrainImages            
clear tempfeature;
save featureL1.mat featureL1;

%%  classification
fprintf('classification\n');
load featureL1;    
softmaxLambda = 1e-4;
numClasses = 12;
sortindex=1:1:200;
for i=1:40
    sortindex(4*(i-1)+1)=5*(i-1)+1;
    sortindex(4*(i-1)+2)=5*(i-1)+2;
    sortindex(4*(i-1)+3)=5*(i-1)+3;
    sortindex(4*(i-1)+4)=5*(i-1)+4;
end
for i=1:40
    sortindex(160+i)=5*i;
end

codeI=featureL1;
softmaxX=codeI;   
load trainLabels.mat;

 for i=0:numClasses-1
       softmaxXTRAIN(:,160*i+1:160*i+160)=softmaxX(:,200*i+sortindex(1:160));                                 
       softmaxYTRAIN(160*i+1:160*i+160)=trainLabels(200*i+sortindex(1:160));
       softmaxXTEST(:,40*i+1:40*i+40)=softmaxX(:,200*i+sortindex(161:200));                                 
       softmaxYTEST(40*i+1:40*i+40)=trainLabels(200*i+sortindex(161:200));
 end
 
options = struct;
options.maxIter = 400;
PN=2400;
softmaxModel = softmaxTrain(numel(codeI) / PN,numClasses, softmaxLambda, softmaxXTRAIN, softmaxYTRAIN, options);                           
[pred] = softmaxPredict(softmaxModel, softmaxXTEST);
acc2 = (pred(:) == softmaxYTEST(:));
acc2 = sum(acc2) / size(acc2, 1);
save accuracy acc2;

fprintf('Accuracy: %2.3f%%\n', acc2 * 100);