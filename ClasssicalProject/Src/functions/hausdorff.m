function dH = hausdorff( A, B) 

edgeA = edge(A,'Canny');
edgeB = edge(B,'Canny');

[H, W, D] = size(A);
flatA = zeros(H*W*D,4);
flatB = zeros(H*W*D,4);
for i=1:H
    for j=1:W
        for k=1:D
            flatA(i,j,k) = edgeA(i,j,k);
        end
    end
end
for i=1:H
    for j=1:W
        for k=1:D
            flatB(i,j,k) = edgeB(i,j,k);
        end
    end
end            
% out = A(all(A,2),:);
flatA = flatA(all(flatA,2),:);
flatB = flatB(all(flatB,2),:);

[L1 W1] = size(flatA); 
[L2 W2] = size(flatB);

%
end



% Compute distance
function dist = compute_dist(A, B) 

m = size(A);
n = size(B);

d_vec = [];
D = [];

% dim= size(A, 2); 
for j = 1:m(1)
    
    for k= 1: n(2)
        
    D(k) = abs((A(j)-B(k)));
      
    end ;
    
    d_vec(j) = min(D); 
      
end;

% keyboard

 dist = max(d_vec);

end