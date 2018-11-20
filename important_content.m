%% svm classifier , linear and rbf
    

%     model = svmtrain(trainLabel, trainData, '-c 3 -h 0 -t 0 -g 2'); 
%     
%     [predict_label, accuracy, dec_values] = svmpredict(test_results1, test_inputs1, model); % test the training data

%% rf classifier
%       B = TreeBagger(200,trainData,trainLabel);
%  	    [Yfit,scores,stdevs] = predict(B,test_inputs1);
%       Yfit=cell2mat(Yfit);
%       Yfit=str2num(Yfit);
%       acc_rf(i) = mean(test_results1(:) == Yfit(:));
%       [~,~,T,auc_rf(i)] = perfcurve(test_results1,(max(scores'))',2);


%% STEP 5: Visualization 
% kkk=1;
% for k=1:l-1
%     kk=n(k+1)*n(k);
%     W{k}=reshape(opttheta(kkk:(kkk+kk-1)), n(k+1), n(k));
%     kkk=kkk+kk;
% end 
% theta = initializeParametersG(n,l);
% 
% %  Use minFunc to minimize the function
% addpath minFunc/
% options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
%                           % function. Generally, for minFunc to work, you
%                           % need a function pointer with two outputs: the
%                           % function value and the gradient. In our problem,
%                           % sparseAutoencoderCost.m satisfies this.
%                           % Maximum number of iterations of L-BFGS to run 
% options.maxIter = 400;
% options.display = 'on';
% 
% 
% [opttheta, cost] = minFunc( @(p) sparseAutoencoderCostAll(p, ...
%                                    n, ...
%                                    lambda, sparsityParam, ...
%                                    beta,training_inputs, training_results), ...
%                               theta, options);
%% svm classifier , linear and rbf
%         A = div_data_cell(trainData,trainLabel');
%         [Bin_clsfr, l_node, r_node] = gen_classifier_OAO(A);
%         L_Node = l_node; R_Node = r_node;
% % Testing Phase
%         [res_class] = classify_clsfr_OAO(test_inputs1, Bin_clsfr, l_node, r_node);
%         err_number = sum(res_class ~= test_results1);
%         Accuracy = (length(test_results1) - err_number) / length(test_results1);

%% rf classifier
%       B = TreeBagger(1000,trainData,trainLabel);
%  	    [Yfit,scores,stdevs] = predict(B,test_inputs1);
%       Yfit=cell2mat(Yfit);
%       Yfit=str2num(Yfit);
%       acc_rf(i) = mean(test_results1(:) == Yfit(:));
%       [~,~,T,auc_rf(i)] = perfcurve(test_results1,(max(scores'))',2);
%     plot(MSE1)
%     %% svm classifier , linear and rbf
%    trainLabel=data(:,1001);
%    trainData=data(:,1:1000);
%    test_results1=trainLabel;
%    test_inputs1=trainData;
%     model = svmtrain(trainLabel, trainData, '-c 3 -h 0 -t 2 -g 2'); 
%     
%     [predict_label, accuracy, dec_values] = svmpredict(test_results1, test_inputs1, model); % test the training data

%% rf classifier
      B = TreeBagger(1000,trainData,trainLabel);  % 1000 no of tree
 	    [Yfit,scores,stdevs] = predict(B,test_inputs1);
      Yfit=cell2mat(Yfit);
      Yfit=str2num(Yfit);
      acc_rf(i) = mean(test_results1(:) == Yfit(:));
      [~,~,T,auc_rf(i)] = perfcurve(test_results1,(max(scores'))',2);
%%
%% mnist dataset
% load('mnist.mat'); 
% training_inputs = double(mnist{1,1}');                          % transpose the matrix to make the pixel data as the row element
% training_results =double(mnist{1,2});                   % convert the digit into the activation value of the neural network
% test_inputs = double(mnist{3,1}');
% test_results = double(mnist{3,2});
%% training dataset
load('105.mat')
load('97.mat')
load('118.mat')
load('130.mat') 
input=[X097_DE_time(1:121000);X105_DE_time(1:121000);X118_DE_time(1:121000);X130_DE_time(1:121000)];
j=1;k=1210;is=100;
% training_inputs=reshape(input,is,2*k);
for i=1:is:(size(input,1)-is+1)
    training_inputs(:,j)=input(i:(i+is-1),1);
    j=j+1;
end
for i=1:size(training_inputs,2)
    if(i<=k)
    training_results(1,i)=0;
    elseif(i<=k*2)
    training_results(1,i)=1;
    elseif(i<=(k*3))
    training_results(1,i)=2;
    else
    training_results(1,i)=3;
    end
end
% test_inputs=training_inputs;
% test_results=training_results;
% training_results=vectorizeData(training_results,size(training_results,1));
%% Testing dataset_0  same both
% load('105.mat')
% load('97.mat')
% load('118.mat')
% load('130.mat') 
% input=[X097_DE_time(1:121000);X105_DE_time(1:121000);X118_DE_time(1:121000);X130_DE_time(1:121000)];
% j=1;
% for i=1:100:size(input,1)
%     test_inputs(:,j)=input(i:(i+99),1);
%     j=j+1;
% end
% for i=1:size(test_inputs,2)
%     if(i<=1210)
%     test_results(1,i)=0;
%     elseif(i<=1210*2)
%     test_results(1,i)=1;
%     elseif(i<=1210*3)
%     test_results(1,i)=2;
%     else
%     test_results(1,i)=3;
%     end
% end
%% testing dataset_1
load('98.mat')
load('106.mat')
load('119.mat')
load('131.mat')
input=[X098_DE_time(1:121000);X106_DE_time(1:121000);X119_DE_time(1:121000);X131_DE_time(1:121000)];
k=121000/100;is=100;
test_inputs=reshape(input,is,4*k);

for i=1:size(test_inputs,2)
    if(i<=k)
    test_results(1,i)=0;
    elseif(i<=k*2)
    test_results(1,i)=1;
    elseif(i<=k*3)
    test_results(1,i)=2;
    else
    test_results(1,i)=3;
    end
end
%% testing dataset_2
% load('99.mat')
% load('107.mat')
% load('120.mat')
% load('132.mat')
% input=[X099_DE_time(1:120000);X107_DE_time(1:120000);X120_DE_time(1:120000);X132_DE_time(1:120000)];
% j=1;k=1200;
% for i=1:100:size(input,1)-99 
%     test_inputs(:,j)=input(i:(i+99),1);
%     j=j+1;
% end
% for i=1:size(test_inputs,2)
%     if(i<=k)
%     test_results(1,i)=0;
%     elseif(i<=k*2)
%     test_results(1,i)=1;
%     elseif(i<=k*3)
%     test_results(1,i)=2;
%     else
%     test_results(1,i)=3;
%     end
% end 
%%test dataset_3
% load('100.mat')
% load('108.mat')
% load('121.mat')
% load('133.mat')
% input=[X100_DE_time;X108_DE_time;X121_DE_time;X133_DE_time];
% for i=1:100:size(input,1)-99 
%     test_inputs(:,i)=input(i:(i+99),1);
% end
% for i=1:size(test_inputs,2)
%     if(i<size(X100_DE_time,1)/100)
%     test_results(1,i)=0;
%     elseif(i<(size(X108_DE_time,1)+size(X100_DE_time,1))/100)
%     test_results(1,i)=1;
%     elseif(i<(size(X121_DE_time)+size(X108_DE_time,1)+size(X100_DE_time,1))/100)
%     test_results(1,i)=2;
%     else
%     test_results(1,i)=3;
%     end
% end 

%% test_dataset_4_diameter diff load same
load('97.mat')
load('169.mat')
load('185.mat')
load('197.mat')
input=[X097_DE_time(1:120000);X169_DE_time(1:120000);X185_DE_time(1:120000);X197_DE_time(1:120000)];
j=1;
for i=1:100:size(input,1)-99 
    test_inputs(:,j)=input(i:(i+99),1);
    j=j+1;
end
for i=1:size(test_inputs,2)
    if(i<=1200)
    test_results(1,i)=0;
    elseif(i<=1200*2)
    test_results(1,i)=1;
    elseif(i<=1200*3)
    test_results(1,i)=2;
    else
    test_results(1,i)=3;
    end
end

%% test_dataset_4_diameter diff load diff
% load('98.mat')
% load('170.mat')
% load('186.mat')
% load('198.mat')
% input1=[X098_DE_time(1:120000);X170_DE_time(1:120000);X186_DE_time(1:120000);X198_DE_time(1:120000)];
% j=1;
% test_inputs=reshape(input1,100,4*1200);
% for i=1:size(test_inputs,2)
%     if(i<=1200)
%     test_results(1,i)=0;
%     elseif(i<=2*1200)
%     test_results(1,i)=1;
%     elseif(i<=3*1200)
%     test_results(1,i)=2;
%     else
%     test_results(1,i)=3;
%     end
% end
% whitening
%% NASA
% load('trainData3');
% load('testData3');
% inputs=[Untitled;Untitled1;Untitled2;Untitled3;Untitled4;Untitled5;Untitled6;Untitled7;Untitled8;Untitled9;Untitled10;Untitled11;Untitled12;Untitled13;...
%     Untitled14;Untitled15;Untitled16;Untitled17;Untitled18;Untitled19];
% inputs1=[inputs(:,1);inputs(:,3)];
% inputs1=inputs1(1:819000);
% training_inputs=reshape(inputs1,150,5460);
% k=2730;
% for i=1:size(training_inputs,2)
%     if(i<=k)
%     training_results(1,i)=0;
%     else
%     training_results(1,i)=1; 
%     end
% end
%% test of NASA
% input=[Untitled21(:,1);Untitled22(:,1);Untitled23(:,1);Untitled24(:,1);Untitled20(:,1);Untitled21(:,3);Untitled22(:,3);...
%     Untitled23(:,3);Untitled24(:,3);Untitled20(:,3)];%;Untitled21(:,7);Untitled22(:,7);Untitled23(:,7);Untitled24(:,7);Untitled20(:,7)];
% input=input(1:204000);
% test_inputs=reshape(input,150,1360);
% k=682;
% for i=1:size(test_inputs,2)
%     if(i<=k)
%     test_results(1,i)=0;
%     else
%     test_results(1,i)=1; 
%      
%     end
% end



% test_results=vectorizeData(test_results,3);
% whitening  
% x=training_inputs;
% sigma = x * x' / size(x, 2);
% [U,S,V] = svd(sigma);
% training_inputs = diag(1./sqrt(diag(S)+eta)) * U' * x;        %pcaWhitening
% training_inputs = U * diag(1./sqrt(diag(S) + eta)) * U' * x;  %zcaWhitening

%     test_inputs=test_inputs1;
%     test_results=test_results1;
    
%     training_inputs=test_inputs1;
%     training_results=test_results1;
%     training_inputs=test_inputs1;
%     training_results=test_results1;
%     k1=test_inputs1;
%     k2=test_results1;