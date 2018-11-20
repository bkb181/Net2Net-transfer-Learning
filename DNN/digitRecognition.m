close all
% clear all   %  working DNN
clc 
nh = [60,45,35,25];                                                      % one hidden layer with 30 nodes
epoch =1;                                                      % training epochs
mini_batch_size =10;                                           % mini bacth size
eta1=3;
sparsityParam = 0.07;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 0.05;     % weight decay parameter      , regularization 
beta = 0.8;             % k l divergence coefficient
load('mnist.mat')

training_results=vectorizeData(training_results,10);

l  = 1+length(nh)+1;                                            % number of layer     
           
n = zeros(l,1);                                                 % initialize the nodes
n(1) = size(training_inputs,1);                                      % nodes at the input layer, number of pixel square

for i = 1:length(nh)   
    n(1+i) = nh(i);                                             % nodes at the hidden layers
end

n(l) = size(training_results,1);                                                      % nodes at the output layer, number of digit                                                     % nodes at the output layer, number of digit
 
W = cell([1,l-1]);                                              % Weight  l=3
b = cell([1,l-1]);                                              % Bias

for i = 1:l-1
    W{i} = randn(n(i+1),n(i));
    b{i} = randn(n(i+1),1);
end
 
corr_val = zeros(epoch,1);                                      % number of correct output for each epoch

[~,col] = size(training_inputs);                                % access the number of column
 
for i = 1:epoch 
    col_prime = randperm(col);
    training_inputs_prime = training_inputs(:,col_prime);
    training_results_prime = training_results(:,col_prime); 
    mini_batches = [];
    for j = 1:mini_batch_size:col
         mini_batches{end+1} = {training_inputs_prime(:,j:j+min(mini_batch_size-1,col-j)), training_results_prime(:,j:j+min(mini_batch_size-1,col-j))};
    end 
%     for j = 1:length(mini_batches)
%         [W,b] = updateWeightBias(mini_batches{j}{1}, mini_batches{j}{2},eta,W,b,n,l);
%     end
%%   LBFGS using minFunc

patches=training_inputs_prime;
for ii=1:l-2
    hiddenSize=n(ii+1);
    visibleSize=n(ii);
    
    theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; 
options.maxIter =4;	  
options.display = 'on'; 
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options); 
W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W{ii}=W1;
patches=feedforward1(training_inputs_prime,W,b,ii+1);
end

%% back-propagation
    for t=1:100
for j = 1:length(mini_batches)
        [W,b] = updateWeightBias(mini_batches{j}{1}, mini_batches{j}{2},eta1,W,b,n,l);
end
%% dataset for other classifier , svm ,rf,rbf
    test_inputs1=(feedforward1(test_inputs,W,b,l-1));
    test_inputs1=test_inputs1';
    test_results1=test_results';
    trainData=feedforward1(training_inputs_prime,W,b,l-1);
    trainData=trainData';
    trainLabel=training_results_prime';

%% testing using softmax
corr_val(i) = validateNetwork(test_inputs, test_results, W, b, l);
% test_results1=validate(test_inputs,test_results,W,b,l);
disp(['Epoch {',num2str(t),'} out of ',num2str(100),': ', num2str(corr_val(i)),'/',num2str(length(test_results))]);
    end 
end
% save('CWRUTr','training_inputs','training_results') 