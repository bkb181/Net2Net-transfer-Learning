function [cost,grad] = sparseAutoencoderCostAll(theta,n, ...
                                             lambda, sparsityParam, beta, data,data1)
%data1= desired output
k=1;
for i=1:length(n)-1
    
    W{i}=reshape(theta(k:k+n(i+1)*n(i)-1),n(i+1),n(i));
    k=k+n(i+1)*n(i);
%     b{i}=theta(2*n(i+1)*n(i)+n(i+1)+1:end);
    Wgrad{i}=zeros(size(W{i}));
%     bgrad{i}=zeros(size(b{i}));
end
for i=1:length(n)-1
    b{i}=theta(k:k+n(i+1)-1);
    k=k+n(i+1);
%     b{i}=theta(2*n(i+1)*n(i)+n(i+1)+1:end); 
    bgrad{i}=zeros(size(b{i}));
end

cost = 0;      l=length(n);                                                          
% size(b{1})
numImages = size(data,2);
numImages_inv = 1./numImages;
% size((W{1}))
% size(data)
H{1}=sigmoid(W{1}*data+repmat(b{1},[1,numImages]));
for i=2:l-1
    H{i}=sigmoid(W{i}*H{i-1}+repmat(b{i},[1,numImages]));
end
output=H{l-1};
mean_act{1}=mean(H{1},2);
mean_act{2}=mean(H{2},2);
% size(H{l-1})
% size(data)
squared_error = 0.5.*(output - data1).^2;
cost = numImages_inv .* sum(squared_error(:));
cost = cost + 0.5 .* lambda .* (sum(W{1}(:).^2)+sum(W{2}(:).^2)+ sum(W{3}(:).^2));
cost = cost + beta .* (sum(kl_div(sparsityParam, mean_act{1}))+sum(kl_div(sparsityParam, mean_act{2})));

delta3 = -(data1-output) .* output .* (1-output);
bgrad{3} = mean(delta3,2);
Wgrad{3} = numImages_inv .* delta3 * H{2}' + lambda .* W{3};

% delta_sparsity = repmat(beta.*(-sparsityParam./mean_act{2}+(1-sparsityParam)./(1-mean_act{2})),[1,numImages]);
% delta2 = (W{3}' * delta3 + delta_sparsity) .* H{2} .* (1-H{2});
% bgrad{2} = mean(delta2,2);
% Wgrad{2} = numImages_inv .* delta2 * H{1}' + lambda .* W{2};
% 
% delta_sparsity = repmat(beta.*(-sparsityParam./mean_act{1}+(1-sparsityParam)./(1-mean_act{1})),[1,numImages]);
% delta1 = (W{2}' * delta2 + delta_sparsity) .*H{1} .* (1-H{1});
% bgrad{1} = mean(delta1,2);
% Wgrad{1} = numImages_inv .* delta1 * data' + lambda .* W{1};
V{1}=data;
for i=1:l-1
V{i+1}=H{i};
end
delta{l-1}=delta3;
for i=l-2:-1:1
    delta_sparsity = repmat(beta.*(-sparsityParam./mean_act{i}+(1-sparsityParam)./(1-mean_act{i})),[1,numImages]);
    delta{i} = (W{i+1}' * delta{i+1} + delta_sparsity) .*H{i} .* (1-H{i});
    bgrad{i} = mean(delta{i},2);
    Wgrad{i} = numImages_inv .* delta{i} * V{i}' + lambda .* W{i};
end
grad = [Wgrad{1}(:) ; Wgrad{2}(:);Wgrad{3}(:) ; bgrad{1}(:) ; bgrad{2}(:);bgrad{3}(:)];

end 
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function divergence = kl_div(a,b)
    divergence = a.*log(a./b) + (1-a).*log((1-a)./(1-b));
end
