function theta = initializeParameters(n,l)

%% Initialize parameters randomly based on layer sizes.
% r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   % we'll choose weights uniformly from the interval [-r, r]
% W1 = rand(hiddenSize, visibleSize) * 2 * r - r;
% W2 = rand(visibleSize, hiddenSize) * 2 * r - r;
% 
% b1 = zeros(hiddenSize, 1);
% b2 = zeros(visibleSize, 1);
for i=1:l-1
    r  = sqrt(6) / sqrt(n(i+1)+n(i)); 
    W{i}=rand(n(i+1),n(i))*2*r-r;
    b{i}=randn(n(i+1),1);
end
theta=[W{1}(:);W{2}(:);W{3}(:);b{1}(:);b{2}(:);b{3}(:)];
% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
% theta = [W1(:) ; W2(:) ; b1(:) ; b2(:)];

end

