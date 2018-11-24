function im_a = image_adjustment(img,isGamma,gamma,isAlpha,Alpha)
% The function performs 3D image adjustment over every slice at Z

im_a = zeros(size(img));
len = length(img(1,1,:));
if isGamma
    for i=1:len
    im_a(:,:,i) = imadjust(img(:,:,i),[],[],gamma);
    end
elseif isAlpha
    im_a=img;
    im_a(im_a<0.5)=0.5*((im_a(im_a<0.5)/0.5).^Alpha);
    im_a(im_a>=0.5)=1-0.5*(((1-im_a(im_a>=0.5))/0.5).^Alpha);
else 
    for i=1:len
        im_a(:,:,i) = imadjust(img(:,:,i));
    end
end
end