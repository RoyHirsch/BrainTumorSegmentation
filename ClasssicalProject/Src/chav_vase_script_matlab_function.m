%% Using quantization for BARTS HG tumor segmentation:
% Senity check code
clear all;
%addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc'))
% load the image matrix named Im
load('dataBN.mat','im')
% load the label matrix, named gt4
load('gt4.mat')

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
predictClean = cleanSegmentaionMask(predict,30);
predictClean = fillSegmentaionMask(predictClean);
diceScore = dice(predictClean,label);

imT2 = imT2 / max(imT2(:));

%% Chane Vase run
% ref:
% https://sites.google.com/site/rexstribeofimageprocessing/chan-vese-active-contours/wubiaotitiezi
clear all;

% Chane-Vase parameters:
smooth_weight = 1; 
image_weight = 1e-6; 
delta_t = 4; 
num_of_iter = 10;

% Load all data
data = load_all_data()
label = load_all_labels()

%% Run the model on multiple examples:

% Init parameters
numOfExamples = 5;      
measureBeforeCV = initMeasureBeforCV(numOfExamples);
measureAfterCV = initMeasureAfterCV(numOfExamples);
CVpredictCell = {}; % cell array for the CV predict masks

tic

for i=1:numOfExamples
    
    % Initial segmentaion:
    [measureBeforeCV.diceArray(i), predict] = quantizationT2andFLSegmentation(data(i).f,label(i).f);
    predictClean = cleanSegmentaionMask(predict,30);
    predictClean = fillSegmentaionMask(predictClean);
   
    labels = double(label(i).f);
    labels(labels~=0) = 1;
    orgImg = data(i).f;
    orgMod = orgImg(:,:,:,2); % extract T2 mod
    orgMod = orgMod / max(orgMod(:)); % image adjustments
    orgMod = image_adjustment(orgMod,0,0,1,2);
    
    % measure parameters before CV
    measureBeforeCV.diceArray(i) = dice(predictClean,labels);
    measureBeforeCV.sensitivityArray(i) = sensitivity(labels, predictClean);
    measureBeforeCV.specificityArray(i) = specificity(labels, predictClean);
    
    % Chan Vase method
%     phi = ac_reinit(predictClean-.5); 
%     phi = ac_ChanVese_model(orgMod, phi, smooth_weight, image_weight, delta_t, num_of_iter); 
    
%     from phi to bineary mask
%     CVpredict = zeros(size(phi));
%     CVpredict(phi>0) = 1;
    CVpredict = activecontour(orgMod,predict,10,'Chan-Vese','SmoothFactor',3,'ContractionBias',-0.7);
    CVpredictCell{i} = CVpredict;
    
    % measure parameters after CV
    measureAfterCV.diceArray(i) = dice(CVpredict,labels);
    measureAfterCV.sensitivityArray(i) = sensitivity(labels, CVpredict);
    measureAfterCV.specificityArray(i) = specificity(labels, CVpredict);
end

measureBeforeCV = sumMeasureStruct(measureBeforeCV, numOfExamples);
measureAfterCV = sumMeasureStruct(measureAfterCV, numOfExamples);

toc

%% view the prediction results
for i=1:numOfExamples
   img = CVpredictCell{i};
   figure; imshow3D(img);
end
%% Single example run:
phi = ac_reinit(predictClean-.5); 

for i = 1:50
    phi = ac_ChanVese_model(imT2, phi, smooth_weight, image_weight, delta_t, 1); 
    if mod(i,10) == 0
        figure; imshow(phi(:,:,80));
    end
end
