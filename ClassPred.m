function [testIndices,yTest] = ClassPred(X,y,Xtest,neuron)
%% Train a customized NN to predict policies with two categories
if (size(y,1)==2)
    %Model construction using Hyperbolic tangent sigmoid transfer function
    %and sigmoid transfer function for first layer and second layer
    %Scaled conjugate gradient backpropagation was used as training
    %algorithm
    net=newff(X,y,neuron,{'tansig','logsig'},'trainscg');
    view(net)
    
    %Divide training data into train, validation and test sets
    net.divideFcn='divideblock';
    net.divideParam.trainRatio = 0.6;
    net.divideParam.valRatio   = 0.2;
    net.divideParam.testRatio  = 0.2;
    
    %train the NN
    [net,tr] = train(net,X,y);
    %nntraintool
    
    %New data prediction
    yTest = net(Xtest);
    %Convert vectors to one indices that represent the vectors
    testIndices = vec2ind(yTest);
    
elseif (size(y,1)>2)
    %Select pattern recognition model in the NN toolbox as the model
    net = patternnet(neuron);
    view(net)
    %Divide training data into train, validation and test sets
    %net.divideFcn='dividerand';
    net.divideFcn='divideblock';
    net.divideParam.trainRatio = 0.6;
    net.divideParam.valRatio   = 0.2;
    net.divideParam.testRatio  = 0.2;
    
    %train the NN
    [net,tr] = train(net,X,y);
    nntraintool
    %New data prediction
    yTest = net(Xtest);
    %Convert vectors to one indices that represent the vectors
    testIndices = vec2ind(yTest);
end





end
