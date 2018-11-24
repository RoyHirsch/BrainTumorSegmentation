function out = loadMRImage(dir,mod,backThs,zeroOut)
% if zeroOut is set the output is vector, else a matrix
%
load(dir);
im = double(im);
% filter modality
if mod > 0
    im = im(:,:,:,mod);
end
% normalize
maxMatrix = max(im(:));
im_n = im / maxMatrix;

im_n(im_n<backThs) = 0;

if zeroOut == 1
    im_n_vec = im_n(im_n~=0);
    out = im_n_vec;
else
    out =  im_n;
end
end
