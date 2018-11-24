function plot_images(list_of_images,list_of_titles,isNorm)
figure;
len = length(list_of_images);

if len == 1
    if isNorm == 1
        maxVal = max(max(list_of_images));
        imshow(list_of_images / maxVal); title(list_of_titles);
    else
        imshow(list_of_images);
    end
    
elseif len % 2 == 0
    
    for i=1:len
        if isNorm == 1
            maxVal = max(max(list_of_images(i)));
            subplot(2,len / 2, i); imshow(list_of_images(i) / maxVal); title(list_of_titles(i));
        else
            subplot(2,len / 2, i); imshow(list_of_images(i));
        end
    end
    
else
    
    for i=1:len
    if isNorm == 1
        maxVal = max(max(list_of_images(i)));
        subplot(1,len, i); imshow(list_of_images(i) / maxVal); title(list_of_titles(i));
    else
        subplot(1,len, i); imshow(list_of_images(i));
    end
    end
end
    
end