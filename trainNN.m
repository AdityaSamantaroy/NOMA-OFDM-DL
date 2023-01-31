% This script is to launch the training of the neural network, based on the
% training data.
% =========================================================================

clear variables;
close all;

% Load training data and essential parameters
load('trainData.mat','XTrain','YTrain');

numSC = 64;

% Batch size
miniBatchSize = 4000;

% Iteration
maxEpochs = 50;

% Sturcture
inputSize = 2*numSC*3;
numHiddenUnits = 128; 
numHiddenUnits2 = 64;
numHiddenUnits3 = numSC;
numClasses = 16;

% DNN Layers
layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% Training options
options = trainingOptions('adam',...
    'InitialLearnRate',0.01,...
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',1, ...
    'LearnRateDropFactor',0.1,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Verbose',1,...
    'Plots','training-progress'); 

% Train the neural network
tic;
net = trainNetwork(XTrain,YTrain,layers,options);
toc;

save('NN.mat','net');

