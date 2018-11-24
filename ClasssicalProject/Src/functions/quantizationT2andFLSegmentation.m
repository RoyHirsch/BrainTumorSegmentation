function [dice, predict] = quantizationT2andFLSegmentation(img, label)
    label = double(label);
    label(label~=0) = 1;

    zSlice = 80;
    im = double(img);
    thr = exp(-4);

    % Separate to modalities:
    imT1 = im(:,:,:,1);
    imT2 = im(:,:,:,2);
    imT1g = im(:,:,:,3);
    imFL = im(:,:,:,4);

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

    predict = double(quantImage & quantImageFL);

    % imshow(quantImage)
    dice =  2*sum(sum(sum(and(label,predict)))) / sum(sum(sum((label + predict))));

end