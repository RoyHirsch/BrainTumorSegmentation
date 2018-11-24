function pdfM = createPDFMatrix(gm,img, isPrint, isSort)
% This function returns a 4D matrix that conatins a set of n pdf
% matrixes. Each matrix is a propability matrix:
% p(gi/img) = (p(img/gi) * (gi)) / sum for i(p(img/gi) * (gi)) 
%
% Input:
% gm - a GMM object
% img - normalized input image, without zeros
% isPrint - boolean value for printing the maps.
%
%
% Output:
% A 4D matrix, dims: [n,H,W,D]
% n - number of geussians, (H,W,D) = size (img)

meanV = gm.mu';
sigmaV = gm.Sigma;
propV = gm.ComponentProportion;

if isSort == 1
    [val,ind] = intersect(meanV,sort(meanV));
    meanV = meanV(ind);
    sigmaV = sigmaV(ind);
    propV = propV(ind);
end

[H W D] = size(img);
pdfM = zeros(H,W,D,length(meanV)); % creat array for the pdf
sumM = zeros(size(img));
for i=1:length(meanV)
    pdfM(:,:,:,i) = normpdf(img,meanV(i),sigmaV(i));
    sumM = sumM + pdfM(:,:,:,i)*propV(i);
end

for i=1:length(meanV)
     pdfM(:,:,:,i) =  pdfM(:,:,:,i)*propV(i) ./ sumM;
end
if isPrint ==1
    for i=1:length(meanV)
        figure;
        imagesc(pdfM(:,:,80,i)); colorbar;title(['PDF matrix of geussian num ',num2str(i),': '])
    end
end
end