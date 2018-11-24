%% Quantization 2
% The purpose of this program is to invastigate the quantization methode
% against the gmm method.
% The final section contains outputs of all the graphs together.
% The code is modular - need to choose:
% 1. data example
% 2. modality
%

clear all;
addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc'))
% load the image matrix named Im
load('BRATS_HG0001/dataBN.mat','im')
% load the label matrix, named gt4
load('BRATS_HG0001/gt4.mat')

label = double(gt4);
label(label~=0) = 1;

im = double(im);
thr = exp(-4);

% Separate to modalities:
imT1 = im(:,:,:,1);
imT2 = im(:,:,:,2);
imT1g = im(:,:,:,3);
imFL = im(:,:,:,4);


% Select the relevat modality:
%
%
img = imFL;
%
%

% Quantization for single example:
% Seperate into 3 tisues of background, healthy and tumor.
[X, Y, Z] = size(img);
threshImg = multithresh(img, 2);

quantImage = zeros(X, Y, Z);
for z=1:Z
    quantImage(:,:,z) = imquantize(img(:,:,z),threshImg);
end

%% Plot the treshold lines obsereved using Outo's method ovet the histogram:
maxVal = max(img(:));
imgNorm = img / maxVal;
imgNorm(imgNorm<thr) = 0;
imgVect = imgNorm(imgNorm~=0);
threshNorm = threshImg / maxVal;
figure;
histogram(imgVect,256);hold on;
line([threshNorm(1),threshNorm(1)],ylim,'color','r');
line([threshNorm(2),threshNorm(2)],ylim,'color','r');
title('Histogram of normalized image without the background voxals with thresholds:')

%% Plot a histogram of the labeled image:
tumorVoxals = imgNorm(label==1);
backVoxals = imgNorm(label==0);
backVoxals = backVoxals(backVoxals~=0);

h1 = histogram(tumorVoxals(:),256,'FaceColor','r');
h2 = histogram(backVoxals(:),256,'FaceColor','g');
bc1 = log10(h1.BinCounts)
bc2 = log10(h2.BinCounts)*10
x=0:1:255
figure;

bar(bc1);
hold on;
bar(bc2,'r');
legend({'tumor voxals','healthy voxals'},'FontSize',16);
title('Histogram healthy voxals and tumor voxals:','FontSize',16)

%% Match gaussian distribution to the histogram of the labeled image:
x = 0:0.001:1;
factor = 10^4;
pdTumor=fitdist(tumorVoxals,'normal');
pdBack=fitdist(backVoxals,'normal');
PDFTumor=pdf(pdTumor,x)*factor;
PDFBack=pdf(pdBack,x)*factor;
figure;
histogram(tumorVoxals(:),256,'FaceColor','r');
hold on;
histogram(backVoxals(:),256,'FaceColor','g');
plot(x,PDFTumor,'r');plot(x,PDFBack,'g');
% title('A matched gaussian distribution to the labeled histogram')
title('Matched gaussians to the two classes histograms, FLAIR modality:')
legend(['Tumor class: mean-',num2str(pdTumor.mean), ' variance-',num2str(pdTumor.sigma)],['Background class: mean-',num2str(pdBack.mean), ' variance-',num2str(pdBack.sigma)])
%% Calculate simple gmm to img:
maxVal = max(img(:));
imgNormTwo = img / maxVal;
imgNormVect = imgNormTwo(:);
% imgVect - the normalized image without the background
gmm = fitgmdist(imgVect,2,'RegularizationValue',0.005,'Options',statset('MaxIter',600));

meansVector = gmm.mu;
variencesVector = gmm.Sigma;
factor = 800;
gausOne = normpdf(x,meansVector(1),variencesVector(:,:,1))*factor;
gausTwo = normpdf(x,meansVector(2),variencesVector(:,:,2))*factor;
% gausThree = normpdf(x,meansVector(3),variencesVector(:,:,3))*factor;
figure; imhist(imgVect,256);hold on;
plot(x,gausOne,'color','r');plot(x,gausTwo,'color','g');
% plot(x,gausThree,'color','b');
title('Results of simple gmm over the data:')

%% plot all 4 graphs together:
figure;

subplot(2,2,1);
histogram(tumorVoxals(:),256,'FaceColor','r');
hold on;
histogram(backVoxals(:),256,'FaceColor','g');
legend('tumor histogrm','healthy tissue histogram');
title('Histogram of ground true labeling, without the background voxals:')

subplot(2,2,2);
histogram(tumorVoxals(:),256,'FaceColor','r');
hold on;
histogram(backVoxals(:),256,'FaceColor','g');
plot(x,PDFTumor,'r');plot(x,PDFBack,'g');
title('A matched gaussian distribution to the labeled histogram')

subplot(2,2,3);
histogram(imgVect,256);hold on;
line([threshNorm(1),threshNorm(1)],ylim,'color','r');
line([threshNorm(2),threshNorm(2)],ylim,'color','r');
title('Histogram of normalized image without the background voxals with thresholds:')

subplot(2,2,4);
imhist(imgVect,256);hold on;
plot(x,gausOne,'color','r');plot(x,gausTwo,'color','g');
% plot(x,gausThree,'color','b');
title('Results of simple gmm over the data:');

%%
tmp = imFL(:,:,80);
edgeImg = edge(tmp,'Canny');
figure;imshow(edgeImg)