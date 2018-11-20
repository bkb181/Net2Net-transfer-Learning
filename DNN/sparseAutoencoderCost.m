function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

numImages = size(data,2);
numImages_inv = 1./numImages;
% size(W1)
% size(data)
activations1 = sigmoid(W1*data+repmat(b1,[1,numImages]));
output = sigmoid(W2*activations1+repmat(b2,[1,numImages]));
mean_act1 = mean(activations1,2);
% kkkk=size(output)
squared_error = 0.5.*(output - data).^2;
cost = numImages_inv .* sum(squared_error(:));
cost = cost + 0.5 .* lambda .* (sum(W1(:).^2) + sum(W2(:).^2));
cost = cost + beta .* sum(kl_div(sparsityParam, mean_act1));

delta2 = -(data-output) .* output .* (1-output);
b2grad = mean(delta2,2);
W2grad = numImages_inv .* delta2 * activations1' + lambda .* W2;

delta_sparsity = repmat(beta.*(-sparsityParam./mean_act1+(1-sparsityParam)./(1-mean_act1)),[1,numImages]);
delta1 = (W2' * delta2 + delta_sparsity) .* activations1 .* (1-activations1);
b1grad = mean(delta1,2);
W1grad = numImages_inv .* delta1 * data' + lambda .* W1;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.
% k=size(W1grad)
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function divergence = kl_div(a,b)
    divergence = a.*log(a./b) + (1-a).*log((1-a)./(1-b));
end
