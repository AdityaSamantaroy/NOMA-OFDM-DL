% This script is to generate testing data for signal detection in a
% two-user NOMA system.
% 
% Essential assumptions are as follows.
% 1. The procedure of generating testing data is the same as the training
% data. 
% 2. The same pilot sequences are used in the testing stage.
% 3. The test is for the same simulation conditions: the same number of
% pilot subcarriers and the same length of cyclib prefix.
% 4. The performance is compared with maximum likelihood (ML), least square
% (LS) and minimum mean square error (MMSE) estimations. The implementation
% of LS and MMSE estimations are based on [1]. 
%
% [1] O. Edfors, M. Sandell, J. -. van de Beek, S. K. Wilson and 
% P. Ola Borjesson, "OFDM channel estimation by singular value 
% decomposition," VTC, Atlanta, GA, USA, 1996, pp. 923-927 vol.2.
% =========================================================================

clear variables;
close all;

% Load training data and essential parameters
load('trainData.mat','h','numPSC','lengthCP','idx_sc','fixedPilot');
% Load neural network
load('NN.mat','net');
% Load channel covariance matrix
% load('Rhh.mat','Rhh');

% System parameters
[numPath,numUE] = size(h);
numSC = 64; % number of subcarriers
numPSym = numUE; % number of pilot OFDM symbols per packet
numDSym = 1; % number of data OFDM symbol per packet
numSym = numPSym+numDSym; % number of OFDM symbols per packet
pilotSpacing = numSC/numPSC;
pilotStart = [1,1]; % pilot starting subcarrier for two users 

% QPSK modulation
constQPSK = [1-1j;1+1j;-1+1j;-1-1j]; % QPSK constellation
a = constQPSK(1);
b = constQPSK(2);
c = constQPSK(3);
d = constQPSK(4);
% Labels
symClass = [a a;a b;a c;a d;b a;b b;b c;b d;c a;c b;c c;c d;d a;d b;d c;d d]; % 16 x 2
labelClass = 1:1:size(symClass,1);

% Testing data size
numPacket = 1000;
fixedPilot = repmat(fixedPilot,1,1,1,numPacket);

% Power allocations
targetSNR_1 = 12; % dB, target SNR for strong user
targetSNR_2 = 12; % dB, target SNR for weak user
targetSNR_linear_1 = 10^(targetSNR_1/10);
targetSNR_linear_2 = 10^(targetSNR_2/10);
H = fft(h,numSC,1); 
gainH = (abs(H).^2).';

% Noise computation
EsN0_dB = 4:2:28; % total received SNR
EsN0 = 10.^(EsN0_dB./10);
symRate = 2; % symbol rate, 2 symbol/s, sent from 2 users at the same time
Es = 1; % symbol energy, joules/symbol
sigPower = Es*symRate; % total signal power (2 symbols/s), joules/s = watts
symPower = sigPower/numUE; % signal power per symbol 
N0 = sigPower./EsN0; % noise power in watts/Hz, assuming subcarrier spacing = 1 Hz
bw = 1; % bandwidth per subcarrier, Hz
nPower = N0*bw; % total noise power in watts, freqency domain
nVar = nPower./2; % noise variance, frequency domain

% Generate channel covariance matrix (or load saved data)
Rhh = getRhh(numPath,numSC,1e5);

% Testing stage
ITER = 1; % number of monte-carlo iterations
numErr_ML = zeros(numUE,numel(EsN0_dB),ITER);
numErr_LS = zeros(numUE,numel(EsN0_dB),ITER);
numErr_MMSE = zeros(numUE,numel(EsN0_dB),ITER);
numErr_DL = zeros(numUE,numel(EsN0_dB),ITER);
tic;
for it = 1:ITER
    for snr = 1:numel(nVar)
        
        % Transmit packets (same procedure as generating training data)
        pilotFrame = zeros(numPSym,numSC,numUE,numPacket);
        pilotFrame(1,:,1,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacket)-0.5),sign(rand(1,numSC,1,numPacket)-0.5));
        pilotFrame(2,:,2,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacket)-0.5),sign(rand(1,numSC,1,numPacket)-0.5));
        pilotFrame(:,pilotStart(1):pilotSpacing:end,1,:) = fixedPilot(:,:,1,:);
        pilotFrame(:,pilotStart(2):pilotSpacing:end,2,:) = fixedPilot(:,:,2,:);
        dataFrame = complex(sign(rand(numDSym,numSC,numUE,numPacket)-0.5),sign(rand(numDSym,numSC,numUE,numPacket)-0.5));
        transmitPacket = zeros(numSym,numSC,numUE,numPacket);
        transmitPacket(1:2,:,:,:) = pilotFrame;
        transmitPacket(end,:,:,:) = 1/sqrt(2)*dataFrame;

        % Collect labels for transmitted datas symbols
        tLabel = zeros(1,numPacket);
        for b = 1:numel(labelClass)
            tLabel(logical(squeeze(dataFrame(1,idx_sc,1,:))==symClass(b,1) & squeeze(dataFrame(1,idx_sc,2,:))==symClass(b,2))) = b;
        end
        
        % Allocate power
        [powerFactor,decOrder] = allocatePower(symPower,gainH,targetSNR_linear_1,targetSNR_linear_2,nVar(snr));
        h_all = repmat(h,1,1,numPacket);
        powerFactor_all = repmat(powerFactor,1,1,numPacket);
        decOrder_all = repmat(decOrder,1,1,numPacket);
        
        % Received packets
        [receivePacket,randomPhase] = dataTransmissionReception(transmitPacket,powerFactor_all,lengthCP,h_all,nVar(snr));
        receivePilot = receivePacket(1:2,:,:); 
        receiveData = receivePacket(end,:,:);

        % ML detection (assuming perfect channel estimation)
        decOrder_sc = squeeze(decOrder_all(idx_sc,:,:));
        idx_1 = decOrder_sc(1,:).';
        idx_2 = decOrder_sc(2,:).';
        H_sc = repmat(H(idx_sc,:),numPacket,1).';
        pF_sc = squeeze(powerFactor_all(idx_sc,:,:)); 
        rData = squeeze(receiveData(1,idx_sc,:)); % numPacket x 1
        tData = squeeze(transmitPacket(end,idx_sc,:,:)); % numUE x numPacket
        [numErr_ML(:,snr,it),rLabel_ML] = detectML(H_sc,randomPhase,constQPSK,pF_sc,rData,idx_1,idx_2,tData,symClass);
        
        % LS and MMSE estimation
        [H_LS,H_MMSE] = channelEstimation(receivePilot,pilotFrame,powerFactor_all,pilotStart,Rhh,nVar(snr),numPSC,H);
        [numErr_LS(:,snr,it),rLabel_LS] = symbolDecodeSIC(rData,H_LS(idx_sc,:,:),decOrder_sc,pF_sc,constQPSK,tData);
        [numErr_MMSE(:,snr,it),rLabel_MMSE] = symbolDecodeSIC(rData,H_MMSE(idx_sc,:,:),decOrder_sc,pF_sc,constQPSK,tData);
    
        % DL detection
        numErr_DL(:,snr,it) = symbolDecodeDL(labelClass,receivePacket,tLabel,net,decOrder_sc,symClass,constQPSK);
        
    end
    
end
toc;
 
numErr_LS = mean(numErr_LS,3);
numErr_MMSE = mean(numErr_MMSE,3);
numErr_ML = mean(numErr_ML,3);
numErr_DL = mean(numErr_DL,3);

figure();
semilogy(EsN0_dB,numErr_DL(1,:),'r-o');hold on;
semilogy(EsN0_dB,numErr_DL(2,:),'r-x');hold on;
semilogy(EsN0_dB,numErr_LS(1,:),'b--o');hold on;
semilogy(EsN0_dB,numErr_LS(2,:),'b--x');hold on;
semilogy(EsN0_dB,numErr_MMSE(1,:),'k--o');hold on;
semilogy(EsN0_dB,numErr_MMSE(2,:),'k--x');hold on;
semilogy(EsN0_dB,numErr_ML(1,:),'g-o');hold on;
semilogy(EsN0_dB,numErr_ML(2,:),'g-x');hold off;
legend('UE 1 - DL','UE 2 - DL','UE 1 - LS','UE 2 - LS','UE 1 - MMSE','UE 2 - MMSE','UE 1 - ML','UE 2 - ML');

function Rhh = getRhh(numPaths,numSC,numChan)
Rhh = zeros(numSC,numSC,numChan);
for i = 1:numChan    
    h = 1/sqrt(2)/sqrt(numPaths)*complex(randn(numPaths,1),randn(numPaths,1)); % L x 1
    H = fft(h,numSC); % numSC x 1
    Rhh(:,:,i) = H*H';    
end
Rhh = mean(Rhh,3); % numSC x numSC
end
