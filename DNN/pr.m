clc
clear all

%training dataset
load('105.mat')
load('97.mat')
load('118.mat')
load('130.mat') 
input=[X105_DE_time;X097_DE_time;X118_DE_time;X130_DE_time];
for i=1:200:size(input,1)-199 
    training_inputs(:,i)=input(i:(i+199),1);
end
for i=1:size(training_inputs,2)
    if(i<size(X105_DE_time,1)/200)
    training_results(1,i)=0;
    elseif(i<(size(X105_DE_time,1)+size(X097_DE_time,1))/200)
    training_results(1,i)=1;
    elseif(i<(size(X118_DE_time)+size(X105_DE_time,1)+size(X097_DE_time,1))/200)
    training_results(1,i)=2;
    else
    training_results(1,i)=3;
    end
end
training_results=vectorizeData(training_results);
%testing dataset
load('98.mat')
load('106.mat')
load('119.mat')
load('131.mat')
input=[X098_DE_time;X106_DE_time;X119_DE_time;X131_DE_time];
for i=1:200:size(input,1)-199 
    test_inputs(:,i)=input(i:(i+199),1);
end
for i=1:size(training_inputs,2)
    if(i<size(X098_DE_time,1)/200)
    test_results(1,i)=0;
    elseif(i<(size(X106_DE_time,1)+size(X098_DE_time,1))/200)
    test_results(1,i)=1;
    elseif(i<(size(X119_DE_time)+size(X106_DE_time,1)+size(X098_DE_time,1))/200)
    test_results(1,i)=2;
    else
    test_results(1,i)=3;
    end
end















