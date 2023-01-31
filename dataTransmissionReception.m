function [receivePacket,randomPhase] = dataTransmissionReception(transmitPacket,powerFactor,lengthCP,h,nVar)
% This function is to model the OFDM signal transmission and reception
% process.  

[numSym,numSC,numUE,numPacket] = size(transmitPacket);
powerScale = reshape(powerFactor,1,numSC,numUE,numPacket);
powerScale = repmat(powerScale,numSym,1,1,1);

% Transmitter
randomPhase = exp(-1j*rand(numUE,numPacket)*2*pi);
% randomPhase = ones(numUE,numPacket);
for u = 1:numUE                
    for p = 1:numPacket
        
        % 1. IFFT, along the row (subcarrier)
        x1 = ifft(sqrt(powerScale(:,:,u,p)).*transmitPacket(:,:,u,p),numSC,2);
    
        % 2. Inserting CP
        x1_CP = [x1(:,numSC-lengthCP+1:end) x1]; 
        
        % 3. Parellel to serial
        x2 = x1_CP.';
        x = x2(:);
    
        % 4. Multipath channel convolution
        y_conv = conv(h(:,u,p),x);
        y(:,u,p) = randomPhase(u,p)*y_conv(1:length(x));
    
    end
end 

% 5. Add up signals from 2 users
y_total = squeeze(sum(y,2));

% 6. Add Gaussion noise to time-domain channel
sigLength = size(y_total,1);
nFre = sqrt(nVar)/sqrt(2).*(randn(numPacket,numSC)+1j*randn(numPacket,numSC)); % Frequency domain
nTime = sqrt(sigLength)*sqrt(sigLength/numSC)*ifft(nFre,sigLength,2); % Time domain
y_total = y_total+nTime.';

% Receiver
Y = zeros(numPacket,numSym,numSC); % Frequency-domain receive signal     
for p = 1:numPacket
    
    % 1. Serial to parallel
    block = reshape(y_total(:,p),numSC+lengthCP,numSym).'; 
    % 2. Removing CP
    y_block = block(:,lengthCP+1:lengthCP+numSC);
    % 3. FFT
    Y(p,:,:) = fft(y_block,numSC,2);
        
end 

receivePacket = permute(Y,[2,3,1]);




