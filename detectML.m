function [errorML,rLabel] = detectML(H,randomPhase,constQPSK,pF,rData,idx_1,idx_2,tData,symClass)
% This function is to perform maximum likelihood (ML) detection for 2
% users, assuming perfect channel estimation. 

[numUE,numPacket] = size(H); 
numLabel = length(constQPSK).^2; % 16

symLabel(logical(symClass == constQPSK(1))) = 1;
symLabel(logical(symClass == constQPSK(2))) = 2;
symLabel(logical(symClass == constQPSK(3))) = 3;
symLabel(logical(symClass == constQPSK(4))) = 4; 
symLabel = reshape(symLabel,numLabel,numUE);

% All possible transmitted data
allSym = repmat(1/sqrt(2)*symClass,1,1,numPacket); 
powerFactor = reshape(pF,1,numUE,numPacket); 
powerFactor =  repmat(powerFactor,numLabel,1,1);
allData = allSym.*sqrt(powerFactor); 

% Compute all possible received data
H = reshape(H,1,numUE,numPacket); 
H_all = repmat(H,numLabel,1,1); 
phase_all = reshape(randomPhase,1,numUE,numPacket); 
phase_all = repmat(phase_all,numLabel,1,1);
restoreData = allData.*H_all.*phase_all;
restoreDataSum = squeeze(sum(restoreData,2)); % Add up signals from 2 users

% Compute mean square error
Y = permute(rData,[2,1]); 
Y = repmat(Y,numLabel,1); 
err = abs(Y-restoreDataSum).^2;

% Find the transmitted signals for the minumum mean square error
[~,idx] = min(err,[],1);

% Obtain labels for estimated symbols
estLabel = zeros(numPacket,numUE); 
for p = 1:numPacket    
    estLabel(p,:) = symLabel(idx(p),:);
end

% Labels for estimated symbols
estLabel_1 = zeros(1,numPacket); 
estLabel_2 = zeros(1,numPacket);
for p = 1:numPacket
    estLabel_1(p) = estLabel(p,idx_1(p));
    estLabel_2(p) = estLabel(p,idx_2(p));    
end

% Labels for transmitted symbols
tLabel = zeros(size(tData)); 
for c = 1:numel(constQPSK)
    tLabel(logical(tData==1/sqrt(2)*constQPSK(c))) = c;
end

% Error rate
errorNum_1 = 1-sum(estLabel_1 == tLabel(1,:))/numPacket;
errorNum_2 = 1-sum(estLabel_2 == tLabel(2,:))/numPacket;
errorML = [errorNum_1;errorNum_2];
rLabel = [estLabel_1;estLabel_2];


