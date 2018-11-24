%% Test function for gmm of BARTS dataset for brain tumor segmentaiton:
% 29.11.17 (c) Roy Hirsch
% matObj = matfile('dataBN.mat');

clear all;
addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/classic_project'))

% Load all data and labels:
% data = load_all_data();
% label = load_all_labels()
%% Load single image:
% load the image matrix named Im
load('BRATS_HG0002/dataBN.mat','im')
% load the label matrix, named gt4
load('BRATS_HG0002/gt4.mat')

%% Parameters:
params.mod = 3;
params.sliceZ = 80;
params.backThs = exp(-4);

%% Regular gmm on a single modality
% Select and pre-prepare a modality for gmm:
img = im(:,:,:,params.mod);
maxMatrix = max(img(:));
im_n = img / maxMatrix;
im_n(im_n<params.backThs) = 0;
im_n_vec = im_n(im_n~=0);

%% Train gmm with bakround voxal and regularization
%im_n = loadMRImage('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc/Data/BRATS_HG0004/dataBN.mat',2,exp(-4),0);
gm = fitgmdist(im_a(:),5,'RegularizationValue',0.003,'Options',statset('MaxIter',400));
pdfM = createPDFMatrix(gm, im_a, 0,1);
segMatrix = createSegmentetionMatrix(pdfM,1,80);
figure; imshow(segMatrix(:,:,60)/5);
%figure; imshow(im_n(:,:,105));

%% 4. generating few GMM models for each modality
% Generate four different vectors for each modality to train GMM model:
im_n = zeros(size(im));
for i=1:4
    temp = im(:,:,:,i);
    minMatrix = min(temp(:));
    im_n(:,:,:,i) = im(:,:,:,i)/maxMatrix;
    im_n(im_n<params.backThs) = 0;
    % for pre-processing of FL 
    if i == 4
       imFL = im_n(:,:,:,4);
       im_a = zeros(size(imFL));
       for i=1:length(imFL(1,1,:))
           temp = imFL(:,:,i);
           temp = imadjust(temp);
           im_a(:,:,i) = temp;
       end
       im_n(:,:,:,4) = im_a;
    end
end
im_T1_vec = im_n(:,:,:,1);
im_T2_vec = im_n(:,:,:,2);
im_T1t_vec = im_n(:,:,:,3);
im_FL_vec = im_n(:,:,:,4);

%% gmm train
%% Multimudat gmm
% gmm of T1
% gmT1 = fitgmdist(im_T1_vec,3,'Options',statset('MaxIter',400));
% pdfMT2 = createPDFMatrix(gmT1, im_n(:,:,:,1), 0,1);
% createSegmentetionMatrix(pdfMT2,1,80);

% gmm of T2
gmT2 = fitgmdist(im_T2_vec(:),5,'RegularizationValue',0.003,'Options',statset('MaxIter',400));
pdfMT2 = createPDFMatrix(gmT2, im_n(:,:,:,2), 0,1);
segMatrixT2 = createSegmentetionMatrix(pdfMT2,1,90);

%gmm of T1t
% gmT1t = fitgmdist(im_T1t_vec(:),5,'RegularizationValue',0.003,'Options',statset('MaxIter',200));
% pdfMT1t = createPDFMatrix(gmT1t, im_n(:,:,:,3), 0,1);
% createSegmentetionMatrix(pdfMT1t,1,80);

% gmm of FLAIR
gmFL = fitgmdist(im_FL_vec(:),4,'RegularizationValue',0.005,'Options',statset('MaxIter',200));
pdfMFL = createPDFMatrix(gmFL, im_n(:,:,:,4), 0,1);
segMatrixFL = createSegmentetionMatrix(pdfMFL,1,90);

%% print T2 - for print:
segMatrixT2(segMatrixT2 == 5) = 4;
figure;
imagesc(segMatrixT2(:,:,90)); colorbar;
figure;
imagesc(segMatrixFL(:,:,90)); colorbar;
%% train model for T1t
% histogram
imhist(im_T1t_vec(im_T1t_vec~=0),256);
%% train model for T1g
gmT1t = fitgmdist(im_T1t_vec(im_T1t_vec~=0),3,'RegularizationValue',0.004,'Options',statset('MaxIter',400));
pdfMT1t = createPDFMatrix(gmT1t, im_n(:,:,:,3), 0,1);
segMatrixT1g = createSegmentetionMatrix(pdfMT1t,1,80);

%% Accuracy 
% Create the prediction mask of T2 and FL
predictT2 = zeros(size(segMatrixT2));
predictFL = zeros(size(segMatrixT2));
% predictT1g = zeros(size(segMatrixT2));

predictT2(segMatrixT2 == 4 | segMatrixT2 == 5) = 1;
predictFL(segMatrixFL == 4) = 1;
%%
figure; imshow3D(predictT2);
figure; imshow3D(predictFL);
predict = and(predictT2,predictFL);
figure; imshow3D(predict);

%%
% predictT1g(segMatrixT1g == 3) = 1;

% predict = and(and(predictT2,predictFL),predictT1g);
predict = and(predictT2,predictFL);

% close holes
[X, Y, Z] = size(predict);
for z=1:Z
    predict(:,:,z) = imfill(predict(:,:,z),'holes');
end
predict = cleanSegmentaionMask(predict, 30);
% Figure
% figure; subplot(1,3,1); imshow(predictT2(:,:,80));subplot(1,3,2); imshow(predictFL(:,:,80));subplot(1,3,3); imshow(predict(:,:,80));
% Calculate dice score
labels = double(gt4);
% figure; imsho w(labels(:,:,80)/4);
labels(labels~=0) = 1;
% print_boundary(predict,im_n,params);
B = bwboundaries(predict(:,:,80));
L = bwboundaries(labels(:,:,80));
figure; imshow(im_n(:,:,80,3));hold on; visboundaries(B);visboundaries(L,'Color','b');
diceScore = dice(labels,predict);
title(['Prediction vs. ground true, dice: ', num2str(diceScore),' :']);
%% Chan and Vese active contours

% fill 'holes'
predict_fill_holes = zeros(size(predict));
for z=1:Z    
    predict_fill_holes(:,:,z) = imfill(predict(:,:,z),'holes');
end

% removes small objects
object_size = zeros(Z,1);
object_size(1:40) = 20;
object_size(41:60) = 60;
object_size(61:90) = 100;
object_size(91:110) = 50;
object_size(111:end) = 20;

for z=1:Z    
    predict_fill_holes_without_small(:,:,z) = bwareaopen(predict(:,:,z),object_size(z));
end

%% Chan Vese level set
maxIterations = 199;
img = im_n(:,:,:,2);
CVpredict = zeros(size(predict));
for z=1:Z
    CVpredict(:,:,z) = activecontour(img(:,:,z),predict_fill_holes(:,:,z),maxIterations,'Chan-Vese');
end
%% help:
figure; subplot(2,2,1); imshow(im_n(:,:,100,1));title('T1');
subplot(2,2,2); imshow(im_n(:,:,100,2));title('T2');
subplot(2,2,3); imshow(im_n(:,:,100,3));title('T1t');
subplot(2,2,4); imshow(im_n(:,:,100,4));title('FL');
figure; imshow(double(gt4(:,:,100))/4);

%%
testim = predict(:,:,60);
fill = imfill(testim,'holes');
figure; imshow(testim);figure; imshow(fill);
%%
figure;imshow3D(predict);figure;imshow3D(double(gt4)/4);