 
close all
clear all
clc

 
nh = [30];                                                      % one hidden layer with 30 nodes
epoch = 5;                                                      % training epochs
mini_batch_size = 10;                                           % mini bacth size
eta = 3.0;                                                      % learning rate 

load('mnist.mat');
 

training_inputs = double(mnist{1,1}');                          % transpose the matrix to make the pixel data as the row element
training_results = vectorizeData(mnist{1,2});                   % convert the digit into the activation value of the neural network

validation_inputs = double(mnist{2,1}');
validation_results = mnist{2,2};

test_inputs = double(mnist{3,1}');
test_results = mnist{3,2}; 

l  = 1+length(nh)+1;                                            % number of layer     
          
n = zeros(l,1);                                                 % initialize the nodes
n(1) = size(mnist{1,1},2);                                      % nodes at the input layer, number of pixel square

for i = 1:length(nh)   
    n(1+i) = nh(i);                                             % nodes at the hidden layers
end

n(l) = 10;                                                      % nodes at the output layer, number of digit 

W = cell([1,l-1]);                                              % Weight  l=3
b = cell([1,l-1]);                                              % Bias

for i = 1:l-1
    W{i} = randn(n(i+1),n(i));
    b{i} = randn(n(i+1),1);
end 

corr_val = zeros(epoch,1);                                      % number of correct output for each epoch

[~,col] = size(training_inputs);                                % access the number of column 
for i = 1:epoch
    
    %%
    % First, the training data and training results are randomly shuffled.
    
    col_prime = randperm(col);
    training_inputs_prime = training_inputs(:,col_prime);
    training_results_prime = training_results(:,col_prime); 
    
    mini_batches = [];
    for j = 1:mini_batch_size:col
         mini_batches{end+1} = {training_inputs_prime(:,j:j+min(mini_batch_size-1,col-j)), training_results_prime(:,j:j+min(mini_batch_size-1,col-j))};
    end
     
    for j = 1:length(mini_batches)
        [W,b] = updateWeightBias(mini_batches{j}{1}, mini_batches{j}{2},eta,W,b,n,l);
    end 
    corr_val(i) = validateNetwork(test_inputs, test_results, W, b, l);
    disp(['Epoch {',num2str(i),'} out of ',num2str(epoch),': ', num2str(corr_val(i)),'/',num2str(length(test_results))]);
    
end  

function [W,b] = updateWeightBias(X,Y,eta,W,b,n,l) 
    nabla_W = cell([1,l-1]);
    nabla_b = cell([1,l-1]);
    for i = 1:l-1
        nabla_W{i} = zeros(n(i+1),n(i));
        nabla_b{i} = zeros(n(i+1),1);
    end

    % compute the gradient of the weights and biases using back
    % progpagation
    for i = 1:size(X,2)
        [dnabla_W, dnabla_b] = backprop(X(:,i),Y(:,i),W,b,l);
        for j = 1:l-1
            nabla_W{j} = nabla_W{j}+dnabla_W{j};
            nabla_b{j} = nabla_b{j}+dnabla_b{j};
        end
    end
        
    % update the value of weight and bias
    for i = 1:l-1
        W{i} = W{i} - (eta/size(X,2)).*nabla_W{i};
        b{i} = b{i} - (eta/size(X,2)).*nabla_b{i};
    end
end
function [dnabla_W,dnabla_b] = backprop(X,Y,W,b,l) 
    a{1} = X;
    
    % fill in the hidden layer nodes up to the output layers
    % 
    [a,a_dot] = feedforward(a,W,b,l);
    
    % gradient descent
    delta = (a{end}-Y).*a_dot{end};
    
    dnabla_W{l-1} = delta*a{l-1}';
    dnabla_b{l-1} = delta;
    
    for i = l-2:-1:1
        delta = W{i+1}'*delta.*a_dot{i};
        dnabla_W{i} = delta*a{i}';
        dnabla_b{i} = delta;
    end
end
function [y,y_dot] = feedforward(x,W,b,l) 
    
    y = x;
    y_dot = cell(1,l-1);

    for i = 1:l-1
        y{i+1} = sigmoid(W{i}*y{i}+b{i});
        y_dot{i} = sigmoidPrime(W{i}*y{i}+b{i});
    end
end
function y = sigmoid(X) 

y = 1./(1+exp(-X));
end
function y = sigmoidPrime(X) 
y = sigmoid(X).*(1-sigmoid(X));
end
function correct = validateNetwork(X,Y,W,b,l) 
    correct = 0;
    
    for i = 1:size(Y,2)
        % fill in the input layers
        a{1} = X(:,i);
        
        % fill in the hidden layer nodes up to the output layers
        [a,~] = feedforward(a,W,b,l);
    
        % find the maximum index value of the output layer
        [~,idx] = max(a{l});
    
        if idx-1 == Y(i)
            correct = correct + 1;
        end
    end  
end
function y = vectorizeData(results) 
    y = zeros(10,length(results));

    for i = 1:length(results)
        y(results(i)+1,i) = 1.0;
    end
end

