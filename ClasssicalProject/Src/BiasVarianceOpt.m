function [ Y ] = BiasVarianceOpt( X,thrsh,bias,covariance)
%This function removes segmentation based on bias and variance
[H,W,D]=size(X);
M=abs(det(covariance));
V=covariance^-1;
K=sqrt(((2*pi)^3)*M);
filter=zeros(H,W,D);

for i=1:H
    for j=1:W
        for k=1:D
            if X(i,j,k)==1
            filter(i,j,k)=-0.5*([i,j,k]-bias)*V*(([i,j,k]-bias)');
            end
        end
    end
end

if thrsh==0
thrsh=1/(10*((covariance(1,1)^2)+(covariance(2,2)^2)+(covariance(3,3)^2)));
end

filter=exp(filter)/K;

Y=(filter.*X);
Y(Y>thrsh)=1;
Y(Y<=thrsh)=0;

end

