function noHolesPredict = fillSegmentaionMask3D(predict)
    [X, Y, Z] = size(predict);
    noHolesPredict = zeros(size(predict));
    for z=1:Z    
        noHolesPredict(:,:,z) = imfill(predict(:,:,z),'holes');
    end
    
    noHolesPredict2 = zeros(size(predict));
    for y=1:Y    
        noHolesPredict2(:,y,:) = imfill(noHolesPredict(:,y,:),'holes');
    end
    
    noHolesPredict3 = zeros(size(predict));
    for x=1:X    
        noHolesPredict3(x,:,:) = imfill(noHolesPredict2(x,:,:),'holes');
    end
end