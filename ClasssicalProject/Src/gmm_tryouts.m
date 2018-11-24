%% tryout for gmm models
[H, W, D] = size(im_n);
im_a = zeros(size(im_n));
for i=1:D
    im_2d = im_n(:,:,i);
    [C, S] = wavedec2(im_2d,1,'haar');
    A = appcoef2(C,S,'haar',1);
    im_a(:,:,i) = imresize(A,2,'bilinear')-im_n(:,:,i);
end
figure;imshow3D(im_a);

figure;imshow3D(im_n);
%%
mu = [1 -1]; 
SIGMA = [.9 .4; .4 .3]; 
X = mvnrnd(mu,SIGMA,10); 
p = mvnpdf(X,mu,SIGMA); 
%%
gmT2 = fitgmdist(im_a(:),5,'RegularizationValue',0.003,'Options',statset('MaxIter',400));
pdfMT2 = createPDFMatrix(gmT2, im_n(:,:,:,2), 0,1);
segMatrixT2 = createSegmentetionMatrix(pdfMT2,1,80);

%%
im_g = imgaussfilt3(im_n);
im_g_vec = im_g(im_g~=0);
%%

gm = fitgmdist(im_g_vec(:),3,'RegularizationValue',0.001,'Options',statset('MaxIter',400));
pdfM = createPDFMatrix(gm, im_g, 0);
segMatrix = createSegmentetionMatrix(pdfM,1,80);
%%
%WDEC = wavedec3(im_n,3,'haar');
WT = dwt3(im_n,'haar');
val = WT.dec;
val2 = val(1,1);
%[H,V,D] = detcoef2('all',C,S,1);
%ALL = H + V + D;
%figure; imshow(ALL(:,:,80));
%%
figure;subplot(2,2,1);imshow(H/max(max((H))));title(['Horizontal coefficients, level: ',num2str(level)]);
subplot(2,2,2);imshow(H/max(max((V))));title(['Vertical coefficients, level: ',num2str(level)]);
subplot(2,2,3);imshow(H/max(max((D))));title(['Diagonal coefficients, level: ',num2str(level)]);