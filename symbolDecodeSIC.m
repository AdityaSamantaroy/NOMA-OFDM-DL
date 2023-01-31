function [numErr,rLabel] = symbolDecodeSIC(rData,H,decOrder,powerFactor,const,tData)
% This function is to perform successive interference cancellation at the
% receiver for the 2-user NOMA system. 

[numUE,numPacket] = size(tData);
H = squeeze(H);

% Indices for strong user and weak user
idx_1 = decOrder(1,:); % 1 x 10
idx_2 = decOrder(2,:); 

% User 1 (strong user)
zfSym_1 = zeros(1,numPacket);
for p = 1:numPacket
    zfSym_1(p) = rData(p)/H(idx_1(p),p); % zero-forcing
end
% Hard decoding
decSym_1 = 1/sqrt(2)*complex(sign(real(zfSym_1)),sign(imag(zfSym_1)));

% User 2 (weak user)
zfSym_2 = zeros(1,numPacket);
for p = 1:numPacket
    resData = rData(p)-H(idx_1(p),p)*sqrt(powerFactor(idx_1(p),p))*decSym_1(p);
    zfSym_2(p) = resData/H(idx_2(p),p); % zero-forcing
end
% Hard decoding
decSym_2 = 1/sqrt(2)*complex(sign(real(zfSym_2)),sign(imag(zfSym_2)));
decSym = [decSym_1;decSym_2];

% Obtain labels for transmitted symbols
tLabel = zeros(size(tData));
for c = 1:numel(const)
    tLabel(logical(tData==1/sqrt(2)*const(c))) = c;
end

% Obtain labels for detected symbols
rLabel = zeros(size(decSym));
for c = 1:numel(const)
    rLabel(logical(decSym==1/sqrt(2)*const(c))) = c;
end

% Error rate
numErr = zeros(numUE,1);
for u = 1:numUE
    numErr(u) = 1-sum(rLabel(u,:)==tLabel(u,:))/numPacket;
end


end
