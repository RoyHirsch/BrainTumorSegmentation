function noHolesPredict = fillSegmentaionMask(predict)
    [X, Y, Z] = size(predict);
    noHolesPredict = zeros(size(predict));
    for z=1:Z    
        noHolesPredict(:,:,z) = imfill(predict(:,:,z),'holes');
    end
end