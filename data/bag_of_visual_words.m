clearvars;
setDir  = fullfile('office_caltech_10','office_caltech_10', 'webcam');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'labelsource','foldernames');
trainingSet = imds;
%imshow(preview(imds));
%trainingSets = partition(imgSets,2);
%[trainingSet,ValidateSet,testSet] = splitEachLabel(imds,1.0,0,'randomize');bag = bagOfFeatures(imageSet(trainingSet.Files),'VocabularySize',100,'PointSelection','Detector', 'StrongestFeatures',0.1);
bag = bagOfFeatures(imageSet(trainingSet.Files),'VocabularySize',800, 'PointSelection','Detector', 'StrongestFeatures',1);

trainingSet.Labels;
%Y_label = ['car';'cat';'dog';'tra'];
Y_train = zeros(length(trainingSet.Files),1);
X_train = encode(bag,imageSet(trainingSet.Files));

X_train(X_train>0)=1;
X_train(X_train<=0)=-1;
label = grp2idx(trainingSet.Labels);
save('webcam.mat','X_train','label','bag');
[pc,score,latent,tsquare] = pca(X_train);
size(score);
mappedX=score(:,1:2);
%mappedX = tsne(X_train)
gscatter(mappedX(:,1), mappedX(:,2), trainingSet.Labels);

%imgSets = imageSet(setDir,'recursive');
%img = read(imgSets(1),1);
%featureVector = encode(bag,img)
%categoryClassifier = trainImageCategoryClassifier(trainingSet,bag);
%confMatrix = evaluate(categoryClassifier,testSet)
