function segMetrix = createSegmentetionMatrix(pdfM,isPrint,zSlice)

[H,W,D,C] = size(pdfM);
segMetrix = zeros(H,W,D);

for i=1:H
    for j=1:W
        for k=1:D
            [x,IND]=max(pdfM(i,j,k,:));
             segMetrix(i,j,k)=IND;
        end
    end
end
if isPrint
    figure; imagesc(segMetrix(:,:,zSlice)); colorbar;
end
end