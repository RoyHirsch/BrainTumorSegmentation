function [ bias,covariance ] = BiasVariance( X )
%This function finds variance and covariance matrix
%bias calculation
[H,W,D]=size(X);
N=sum(sum(sum(X)));
bias=zeros(1,3);
for i=1:H
    for j=1:W
        for k=1:D
            if X(i,j,k)==1
            bias=bias+[i,j,k];
            end
        end
    end
end

bias=round(bias/N);

covariance=0;
for i=1:H
    for j=1:W
        for k=1:D
            if X(i,j,k)==1
            covariance=covariance+([i,j,k]-bias)'*([i,j,k]-bias);
            end
        end
    end
end
covariance=covariance/N;

end

