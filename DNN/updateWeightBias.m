function [W,b] = updateWeightBias(X,Y,eta,W,b,n,l) 
    nabla_W = cell([1,l-1]);
    nabla_b = cell([1,l-1]);
    lambda=0;
    for i = 1:l-1
        nabla_W{i} = zeros(n(i+1),n(i));
        nabla_b{i} = zeros(n(i+1),1);
    end
 
    for i = 1:size(X,2)
        [dnabla_W, dnabla_b] = backprop(X(:,i),Y(:,i),W,b,l);
        for j = 1:l-1
            nabla_W{j} = nabla_W{j}+dnabla_W{j};
            nabla_b{j} = nabla_b{j}+dnabla_b{j};
        end
    end
        
    % update the value of weight and bias
    for i = 1:l-1
        W{i} = W{i} - ((eta/size(X,2)).*nabla_W{i}+lambda*(W{i}));
        b{i} = b{i} - (eta/size(X,2)).*nabla_b{i};
    end
end

