function [feature,label,idx] = getFeatureAndLabel(realData,imagData,labelData,targetLabel)
% This function is to construct real-valued feature vector 
% (training samples) and corresponding labels for the training. 

[numSym,numSC,~] = size(realData);
dimFetureVec = numSym*numSC*2; 


idx = find(labelData == targetLabel);
numSample = length(idx);

% Labels
label = targetLabel*ones(1,numSample);

% Real-valued feature vectors
RealCollection = realData(:,:,idx); % 2 x 64 x #
RealCollection = permute(RealCollection,[2,1,3]); % 64 x 2 x #
RealCollection = reshape(RealCollection,numSC*numSym,numSample); % 128 x #

ImagCollection = imagData(:,:,idx);
ImagCollection = permute(ImagCollection,[2,1,3]); % 64 x 2 x #
ImagCollection = reshape(ImagCollection,numSC*numSym,numSample); % 128 x #

% Collect real and imaginary parts of training data as feature vectors 
feature = zeros(dimFetureVec,numSample);
feature(1:2:end,:) = RealCollection;
feature(2:2:end,:) = ImagCollection;

end



