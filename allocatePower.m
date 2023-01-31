function  [powerFactor,decOrder] = allocatePower(symPower,gainH,targetSNR_1,targetSNR_2,nVar)
% This function is to allocate transmit power to users based on their
% channel gain. The power is allocated to make both users achieve a target
% SNR. 

[numUE,numSC] = size(gainH); 

% Channel gain levels
gainDiff = diff(gainH,[],1); % column2-column1
highGain(logical(gainDiff<0)) = 1;
highGain(logical(gainDiff>0)) = 2;
lowGain(logical(gainDiff>0)) = 1;
lowGain(logical(gainDiff<0)) = 2;
highGain = highGain.';
lowGain = lowGain.';

% Calculate power allocation factor
powerFactor = zeros(numSC,numUE);
for sc = 1:numSC
    
    lowPower = targetSNR_2*nVar./(symPower*gainH(highGain(sc),sc)); 
    highPower = (targetSNR_1*(lowPower.*symPower*gainH(highGain(sc),sc)+nVar))./symPower*gainH(lowGain(sc),sc); 
    highPowerFactor = highPower./(highPower+lowPower);
    lowPowerFactor = lowPower./(highPower+lowPower);
    powerFactor(sc,highGain(sc)) = lowPowerFactor;
    powerFactor(sc,lowGain(sc)) = highPowerFactor;
    
end

decOrder = zeros(numSC,numUE);
[~,decOrder(:,1)] = max(powerFactor,[],2);
[~,decOrder(:,2)] = min(powerFactor,[],2);
