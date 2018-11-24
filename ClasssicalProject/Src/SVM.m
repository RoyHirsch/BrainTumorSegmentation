% SVM model for brain tumor classification on BARTS data
%
% Simple tryout to train SVM model over BARTS dataset in matlab.
%
% SVM: creating an optimazid hyperplane to separate between two classes [1,-1]
% the optimal hyperplane have a maximum margin, the hyperplane equation:
% f(x) =x'beta + b
%   x - the training data
%   beta - the hyperplane normal (hyperplane parameters)
%   b - bias
% 
% See more info:
% https://www.mathworks.com/help/stats/support-vector-machines-for-binary-classification.html#bs3tbev-16
% https://www.mathworks.com/help/stats/fitcsvm.html#bt8v_23-1

% ideas:
% soft edges for nonseparable data
% adding OutlierFraction

clear all;

% 1. Load the data
addpath(genpath('/Users/royhirsch/Documents/GitHub/Final-Project/ProjectSrc'))
% load the image matrix named Im
load('/Data/BRATS_HG0001/dataBN.mat','im')
% load the label matrix, named gt4
load('/Data/BRATS_HG0001/gt4.mat')

% 2. Pre-process and re-shape the data
X = double(im);
[H, W, D, C] = size(im);
X = reshape(X,[H*W*D,C]);

Y = double(gt4);
Y = reshape(Y,[H*W*D,1]);
Y(Y~=0) = 1;

%% 3. Reduce for train and validation data:

% parameters:
params.train = 100000;
params.val = 1000;

ind = randi([1 5000000],1,params.train);
Xtrain = X(ind,:);
Ytrain = Y(ind);

% Extract unique training data:
% use temp data struct for unique values only
tempMatrix = zeros(params.train,5);
tempMatrix(:,1:4) = Xtrain;
tempMatrix(:,5) = Ytrain;
tempMatrixUnique = unique(tempMatrix,'rows');

Xtrain = tempMatrixUnique(:,1:4);
Ytrain =  tempMatrixUnique(:,5);

% Extract validation data:
indVal = randi([1 5000000],1,params.train);
Xval = X(indVal,:);
Yval = Y(indVal,:);
%% 3. Train simple SVM model
SVMModel = fitcsvm(Xtrain,Ytrain,'RemoveDuplicates','on');

%% 4. Validation Accuracy:
[Ypredict,score] = predict(SVMModel,Xval);
accuracy = sum(Ypredict==Yval)/length(Yval);

%% 5. Predict for the whole data:
[label,score] = predict(SVMModel,X);
Ypredict =  reshape(label,[H,W,D]);

%% 5.5 Predict for a different image
load('/Data/BRATS_HG0004/dataBN.mat','im')
% load the label matrix, named gt4
load('/Data/BRATS_HG0004/gt4.mat')
Xtest = double(im);
Xtest = reshape(Xtest,[],C);
Ytest = double(gt4);
Ytest(Ytest~=0) = 1;

[label,score] = predict(SVMModel,Xtest);
Ypredict =  reshape(label,H,W,[]);

%% 6. dice score
dice = dice(Ypredict,double(gt4));

%% 7. Interactive test:
figure;imshow3D(double(gt4)/4);
figure;imshow3D(Ypredict);