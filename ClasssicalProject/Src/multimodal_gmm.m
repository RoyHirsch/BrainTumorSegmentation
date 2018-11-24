%%  ---------- Train gmm for multi-modal input data: -------------------
clear all;
%% load data
addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc'))
% load the image matrix named Im
load('BRATS_HG0001/dataBN.mat','im')
% load the label matrix, named gt4
load('BRATS_HG0001/gt4.mat')

im = double(im);
% im = image_adjustment(im,0,0,1,2);
label = double(gt4);
%% Pre-process the data:
% Training data in [n,d] struct where:
% n - number of voxals
% d - number of modulities.
% 
% The model will calculate the probability of the i's voxal
% (1X4 double vector) to be a part of the 4D geussian distribiution.
% 
% 
[H, W, D, C] = size(im);
dataMatrix = zeros(H*W*D,3); % dataMatrix - a reshape of input data to shape(H*W*D,c) struct
for i=1:C
numChannels = 4;
[H, W, D, C] = size(im);
dataMatrix = zeros(H*W*D,numChannels); % dataMatrix - a reshape of input data to shape(H*W*D,c) struct
for i=1:numChannels
    modeImg = im(:,:,:,i);
    minMode = min(modeImg(:));
    maxMode = max(modeImg(:));
    dataMatrix(:,i) = modeImg(:) / maxMode;
%     dataMatrix(:,i) = modeImg(modeImg>minMode);
end
end

%% moltimodal - only T2 and FLAIR:
[H, W, D, C] = size(im);
dataMatrix = zeros(H*W*D,2);
modeImg = im(:,:,:,2);
maxMode = max(modeImg(:));
dataMatrix(:,1) = modeImg(:) / maxMode;
modeImg = im(:,:,:,4);
maxMode = max(modeImg(:));
dataMatrix(:,2) = modeImg(:) / maxMode;


%% Train GMM:
gm = fitgmdist(dataMatrix,3,'RegularizationValue',0.001,'Options',statset('MaxIter',400));

%% Create PDF matrix, dimentions[n,classes].
% In each (i,j) cell: the distribiuotion the the i'th voxal to be a part of
% the j's class.
% 
% The function 'mvnpdf' (mvnpdf(X,mu,sigma)) creats a multimodat guessian distribioution with:
% X - [n,d], d - the modalities
% mu = [classes,d] vector of means per modality (dimention)
% sigma - [d,d] covarience matrix for the class
% 

meanV = gm.mu;
sigmaV = gm.Sigma;
propV = gm.ComponentProportion;
classes = length(meanV(:,1));

% Sort by proportion (the smallest is the tumor class). 
% For getting a consistent classes for every run

[val,ind] = intersect(propV,sort(propV));
meanV = meanV(ind,:);
sigmaV = sigmaV(:,:,ind);
propV = propV(ind);

pdfM = zeros(H*W*D,classes); 
sumM = zeros(H*W*D,1);
for i=1:classes
    pdfM(:,i) = mvnpdf(dataMatrix,meanV(i,:),sigmaV(:,:,i));
    sumM = sumM + pdfM(:,i)*propV(i);
end

for i=1:classes
     pdfM(:,i) =  pdfM(:,i)*propV(i) ./ sumM;
end

%% Reshape pdfM to 3D probability matrices

% Each classes dim is a 3D probability matrix for this class.
pdfM = reshape(pdfM,H,W,D,classes);
for i=1:classes
        figure;
        imagesc(pdfM(:,:,80,i)); colorbar;title(['PDF matrix of geussian num ',num2str(i),': '])
end

%% Create segmentation matrix
% By max pooling for all classes in each voxal (i,j).
segMetrix = zeros(H,W,D);

for i=1:H
    for j=1:W
        for k=1:D
            [x,IND]=max(pdfM(i,j,k,:));
             segMetrix(i,j,k)=IND;
        end
    end
end
figure; imagesc(segMetrix(:,:,80)); colorbar;

%% Evaluate the segmentation:
predict = zeros(size(segMetrix));
predict(segMetrix==1) = 1;
predict = cleanSegmentaionMask(predict,30);
diceScore = dice(predict,label);

%% Print boundries:
B = bwboundaries(predict(:,:,80));
L = bwboundaries(label(:,:,80));
im_n = im(:,:,:,3);
im_n = im_n / max(im_n(:));
figure; imshow(im_n(:,:,80));hold on; visboundaries(B);visboundaries(L,'Color','b');
diceScore = dice(label,predict);
title(['Prediction vs. ground true, dice: ', num2str(diceScore),' :']);
