load pooledFeaturesL2.mat
load label.mat;
PN=2400;
softmaxLambda = 1e-4;
numClasses = 12;
numTrainImages=2400;
numTrain=1920;
numTest=480;
tempfeature = permute(pooledFeaturesL2, [3 1 2 4]);
featureL1 = reshape(tempfeature, numel(pooledFeaturesL2) / PN,PN); %numel(pooledFeaturesTrain) / numTrainImages            
featureL2=reshape(featureL1,numel(featureL1)/numTrainImages,200,12);

% two seperation methods: 1. using the fixed random number for the first and second layer
load randnum;
for i=1:12
randtemp1=randnum{i};
pooledFeaturesTrain(:,:,i)=featureL2(:,randtemp1(1:160),i);
pooledFeaturesTest(:,:,i)=featureL2(:,randtemp1(161:200),i);
end
% % 2. using the different random number for the first and second layer
% for i=1:12
% randtemp1=randperm(200);
% pooledFeaturesTrain(:,:,i)=featureL2(:,randtemp1(1:160),i);
% pooledFeaturesTest(:,:,i)=featureL2(:,randtemp1(161:200),i);
% end
save pooledFeaturesTrain pooledFeaturesTrain '-v7.3';
save pooledFeaturesTest pooledFeaturesTest '-v7.3';
clear randtemp1 randnum featureL2;
numTrain=1920;
numTest=480;
pooledFeaturesTrain=reshape(pooledFeaturesTrain,numel(pooledFeaturesTrain)/numTrain,numTrain);
pooledFeaturesTest=reshape(pooledFeaturesTest,numel(pooledFeaturesTest)/numTest,numTest);
codeI=featureL1;
softmaxX=codeI;  
load label.mat;
clear featureL1 featureL2;
numClasses=12;
options = struct;
options.maxIter = 400;
softmaxModel = softmaxTrain(numel(codeI) / numTrainImages,numClasses, softmaxLambda,pooledFeaturesTrain, trainLabels, options);%                            
[pred] = softmaxPredict(softmaxModel, pooledFeaturesTest);
save pred pred;
acc2 = (pred(:) == testLabels(:));
acc2 = sum(acc2) / size(acc2, 1);
fprintf('Accuracy: %2.3f%%\n', acc2 * 100);