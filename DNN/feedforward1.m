function [y1] = feedforward1(x,W,b,l)
%     y = x;
    for j=1:size(x,2)
        y = x(:,j); 
        for i = 1:l-1
        y=sigmoid(W{i}*y+b{i});
        end
        y1(:,j)=y;
    end
end
 