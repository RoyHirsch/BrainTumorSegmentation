function print_boundary(predict_img,original_img,params)
% print boundary to the hole matrix
B = bwboundaries(predict_img(:,:,params.sliceZ))
figure; imshow(original_img(:,:,params.sliceZ,params.mod));hold on; visboundaries(B);
title(['The segmenteg img, mode: ', num2str(params.mod), '  sliceZ: ', num2str(params.sliceZ)]);

end