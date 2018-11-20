function [y,y_dot] = feedforward(x,W,b,l)
    y = x;
    y_dot = cell(1,l-1);

    for i = 1:l-1
        y{i+1} = sigmoid(W{i}*y{i}+b{i});
        y_dot{i} = sigmoidPrime(W{i}*y{i}+b{i});
    end
end
