function plot3DPointCload(img)
%   plots a 3D graph of the input image
%   
%   input: img - a binary 3D image, 1 are the dots to plot
%   
    [X, Y, Z] = size(img);
    numOfNoneZero = sum(img(:));
    tmp = zeros(numOfNoneZero,3);
    ind = 1;
    for i=1:X
        for j=1:Y
            for k=1:Z
                if img(i,j,k) == 1
                    tmp(ind,:) = [i,j,k];
                    ind = ind + 1;
                end
            end
        end
    end
    centerOfMass = round(mean(tmp,1));
    figure;pcshow(tmp); hold on; pcshow(centerOfMass);

end