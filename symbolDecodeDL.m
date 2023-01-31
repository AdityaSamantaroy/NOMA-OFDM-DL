function [numErr,rLabel] = symbolDecodeDL(labelClass,receivePacket,dataLabel,net,decOrder_sc,symClass,constQPSK, numSC, numSym)       
% This function is to detect received symbols for 2 users simultaneously 
% using trained neural network.

[numUE,numPacket] = size(decOrder_sc);
% fprintf("%d, %d\n",numUE, numPacket);
idx_1 = decOrder_sc(1,:); 
idx_2 = decOrder_sc(2,:); 

% Construct feature vectors and estimated labels
XTest = zeros(64, size(dataLabel,2), 6, 1);
YTest = zeros(1, size(dataLabel,2));
for n = 1:length(labelClass)
    % Construct feature vector and labels
    % Labels for the target subcarrier
    [feature,label,inx] = getFeatureAndLabel(real(receivePacket),imag(receivePacket),dataLabel,n);
    featureVec = cell2mat(mat2cell(feature,size(feature,1),ones(1,size(feature,2)))); % cell, 1 x #perClass, each cell, 384 x 1
        
    featureVec = reshape(featureVec, numSC, [], 2*numSym, 1);
    fprintf("dataLabel -> %s\n",mat2str(size(dataLabel))); % [1 1000]
    fprintf("receivePacket -> %s\n",mat2str(size(receivePacket))); % [3 64 1000]
    fprintf("featureVec -> %s\n",mat2str(size(featureVec)));
    fprintf("label -> %s\n",mat2str(size(label)));
    fprintf("XTest(inx) -> %s\n",mat2str(size(XTest(:, inx, :))));
     
    XTest(:, inx, :) = featureVec;
    YTest(:, inx) = label;

%     [feature,label,idx] = getFeatureAndLabel(real(receivePacket),imag(receivePacket),dataLabel,n);
%     featureVec = mat2cell(feature,size(feature,1),ones(1,size(feature,2)));
%     XTest(idx) = featureVec;
%     YTest(idx) = label; 
end

% XTest = XTest.';
XTest = permute(XTest, [1 3 4 2]);
YTest = categorical(YTest.');
fprintf("%s\n",mat2str(size(XTest)));
fprintf("%s\n",mat2str(size(YTest)));
% save("testData.mat", 'XTest', 'YTest');

% load("testData.mat", 'XTest', 'YTest');

% Signal detection (prediction)
YPred = classify(net,XTest);
disp(size(YPred))
disp(size(YTest))
disp(size(numPacket))

% Obtain indices for misclassified packets    
wrongPred = YPred(logical(YPred~=YTest)); 
wrongPred = str2double(string(wrongPred)); % Index (postion) for misclassified frames in YPred
numWrongPred = length(wrongPred);

% Obtain packets of wrong predictions
% (perhaps either UE 1 or UE 2 is wrong, perhaps both are wrong)    
packetIdx = 1:numPacket;
wrongPacket = packetIdx(logical(YPred~=YTest));

% Calculate correct detection for each user  
correctPred_1 = sum(YPred == YTest);
correctPred_2 = sum(YPred == YTest);
for n = 1:numWrongPred
        
    % Correct labels for misclassified packets
    correctLabel = dataLabel(wrongPacket(n)); 
    correctSym = symClass(correctLabel,:); % 1 x 2, 2 users
    correct_1 = correctSym(idx_1(wrongPacket(n)));
    correct_2 = correctSym(idx_2(wrongPacket(n)));
        
    % Decoded symbols
    decodeSym = symClass(wrongPred(n),:); % 1 x 2, 2 users
    decodeSym_1 = decodeSym(idx_1(wrongPacket(n)));
    decodeSym_2 = decodeSym(idx_2(wrongPacket(n)));

    if correct_1 == decodeSym_1
        correctPred_1 = correctPred_1+1;
    end
    if correct_2 == decodeSym_2
        correctPred_2 = correctPred_2+1;
    end
        
end

% Error rate per user
numErr_1 = 1-correctPred_1/numPacket;
numErr_2 = 1-correctPred_2/numPacket;
numErr = [numErr_1;numErr_2];

% Obtain received constellation for each user
symLabel(logical(symClass == constQPSK(1))) = 1;
symLabel(logical(symClass == constQPSK(2))) = 2;
symLabel(logical(symClass == constQPSK(3))) = 3;
symLabel(logical(symClass == constQPSK(4))) = 4; 
symLabel = reshape(symLabel,length(labelClass),numUE); % 16 x 2
estimateLabel = str2double(string(YPred));
rLabel = zeros(numPacket,numUE);
for p = 1:numPacket
    rLabel(p,:) = symLabel(estimateLabel(p),:);
end
rLabel = rLabel.';
