function correct = validateNetwork(X,Y,W,b,l)

    correct = 0; 
    for i = 1:size(Y,2)
        % fill in the input layers
        a{1} = X(:,i); 
        % fill in the hidden layer nodes up to the output layers
        [a1,~] = feedforward(a,W,b,l); 
        % find the maximum index value of the output layer
        [~,idx] = max(a1{l});
    
        if idx-1 == Y(i)
            correct = correct + 1;
        end
%         az=exp(a1{1});
%         s=sum(az);
%         az=az./s; 
    end   
end

