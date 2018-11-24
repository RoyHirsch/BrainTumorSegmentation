function noNoisePredict = cleanSegmentaionMask(predict, minArea)
    [X, Y, Z] = size(predict);
    noNoisePredict = zeros(size(predict));
    for z=1:Z    
        noNoisePredict(:,:,z) = bwareaopen(predict(:,:,z),minArea);
    end
end