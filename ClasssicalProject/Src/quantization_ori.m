%% Using quantization for BARTS HG tumor segmentation:

% 1. Load the data

clear all;
addpath(genpath('\Users\אורי\Documents\GitHub\Final-Project\ProjectSrc'))
% load the image matrix named Im
load('/Data/BRATS_HG0001/dataBN.mat','im')
% load the label matrix, named gt4
load('/Data/BRATS_HG0001/gt4.mat')

label = double(gt4);
label(label~=0) = 1;

zSlice = 80;
im = double(im);
thr = exp(-4);

% Separate to modalities:
imT1 = im(:,:,:,1);
imT2 = im(:,:,:,2);
imT1g = im(:,:,:,3);
imFL = im(:,:,:,4);
%normalization
imT2 = imT2/max(max(max(imT2)));
imFL = imFL/max(max(max(imFL)));
% % %color adjustment %%%%%%Didnt help  :(
% imT2 = image_adjustment(imT2,0,0,1,2);
% imFL = image_adjustment(imFL,0,0,1,2);

%smooth3
imT2=smooth3(imT2,'box');
imFL=smooth3(imFL,'box');
% Quantization for single example:
[X, Y, Z] = size(imT2);
threshT2 = multithresh(imT2, 2);

quantImage = zeros(X, Y, Z);
for z=1:Z
    quantImage(:,:,z) = imquantize(imT2(:,:,z),threshT2);
end

threshFL = multithresh(imFL, 2);
quantImageFL = zeros(X, Y, Z);
for z=1:Z
    quantImageFL(:,:,z) = imquantize(imFL(:,:,z),threshFL);
end

% Ceate predict mask
quantImage(quantImage~=3) = 0;
quantImageFL(quantImageFL~=3) = 0;

% Calculate predict mask and dicescore:
predict = quantImage & quantImageFL;
diceScore = dice(predict,label);
%%
[bias,variance]=BiasVariance(predict);
newpredict = BiasVarianceOpt(predict,0,bias,variance);
newDice=dice(newpredict,label);
%%
newpredict(newpredict>10^-7)=1
newpredict(newpredict<=10^-7)=0;
%% Plot slices:

maxT1 = max(imT1(:));
maxT2 = max(imT2(:));
maxT1g = max(imT1g(:));
maxFL = max(imFL(:));

imT1 = imT1 ./ maxT1;
imT2 = imT2 ./ maxT2;
imT1g = imT1g ./ maxT1g;
imFL = imFL ./ maxFL;

imT1(imT1<=thr) = 0;
imT2(imT2<=thr) = 0;
imT1g(imT1g<=thr) = 0;
imFL(imFL<=thr) = 0;

figure;
subplot(2,2,1);
imshow(imT1(:,:,zSlice));
title(' T1 image:')
subplot(2,2,2);
imshow(imT2(:,:,zSlice));
title('T2 image:')
subplot(2,2,3);
imshow(imT1g(:,:,zSlice));
title('T1g image:')
subplot(2,2,4);
imshow(imFL(:,:,zSlice));
title('FL image:')

%% Plot histograms and thresholds of quantizations:

imT1_vec = imT1(imT1~=0);
imT2_vec = imT2(imT2~=0);
imT1g_vec = imT1g(imT1g~=0);
imFL_vec = imFL(imFL~=0);

threshT2 = threshT2 / maxT2;
threshFL = threshFL / maxFL;

figure;

subplot(2,2,1);
imhist(imT1_vec,256);
title('Histogram of normalized T1 image:')
subplot(2,2,2);
imhist(imT2_vec,256);hold on;
line([threshT2(1),threshT2(1)],ylim,'color','r');
line([threshT2(2),threshT2(2)],ylim,'color','r');
title('Histogram of normalized T2 image:')
subplot(2,2,3);
imhist(imT1g_vec,256);
title('Histogram of normalized T1g image:')
subplot(2,2,4);
imhist(imFL_vec,256);hold on;
line([threshFL(1),threshFL(1)],ylim,'color','r');
line([threshFL(2),threshFL(2)],ylim,'color','r');
title('Histogram of normalized FL image:')

%% Evaloation on many examples:

data = load_all_data();
label = load_all_labels();

numOfExamples = 20;
diceScoreArray = zeros(numOfExamples, 1);
sensScoreArray = zeros(numOfExamples, 1);
specScoreArray = zeros(numOfExamples, 1);
for i=1:numOfExamples
    [predict] = quantizationT2andFLSegmentation_ori(data(i).f,label(i).f);
    lables=double(label(i).f);
    lables(lables~=0)=1;
    predict = cleanSegmentaionMask(predict,30);
%     [bias,variance]=BiasVariance(predict);
%     predict = BiasVarianceOpt(predict,0,bias,variance);
    diceScoreArray(i)= dice(lables,predict);
    specScoreArray(i)= specificity(lables,predict);
    sensScoreArray(i)= sensitivity(lables,predict);
end
avarageDicScor = sum(diceScoreArray) / numOfExamples; 
avarageSensScor = sum(sensScoreArray) / numOfExamples; 
avarageSpecScor = sum(specScoreArray) / numOfExamples; 