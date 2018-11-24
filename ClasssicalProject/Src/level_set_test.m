% Level-set test

clear all;
addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc'))
% load the image matrix named Im
load('/Data/BRATS_HG0001/dataBN.mat','im')
% load the label matrix, named gt4
load('/Data/BRATS_HG0001/gt4.mat')


% Initial segmentaion mask
[diceScore, predict] = quantizationT2andFLSegmentation(im,gt4);
predict = cleanSegmentaionMask(predict,20);
figure; imshow3D(predict);


%% 