% This script is to generate training data for signal detection in a
% two-user NOMA system [1].
% 
% Essential assumptions are as follows.
% 1. One OFDM packet contains 3 blocks: 2 pilot blocks and 1 data block. 
% 2. Based on QPSK moduation, the total number of symbol combinations
% for 2 users (number of distinct labels) is 4^2 = 16.
% 3. For the neural network to learn the pattern between transmitted signal
% and received signal, the same pilot sequence is used throughout the
% entire simulation. 
% 4. The neural network is trained for a static channel with a specific 
% number of pilot subcarriers NUMPSC and a specific length of cyclic 
% prefix LENGTHCP. To test the robustness of the neural network, a random 
% phase shift is added to each packet. 
% 
% [1] Narengerile and J. Thompson, "Deep Learning for Signal Detection 
% in Non-Orthogonal Multiple Access Wireless Systems," 2019 UK/ China 
% Emerging Technologies (UCET), Glasgow, United Kingdom, 2019, pp. 1-4.
% =========================================================================

clear variables;
close all;

% Random seed for reproducing static channel
s = RandStream('mt19937ar','Seed',1921164231);
RandStream.setGlobalStream(s);

% System parameters
lengthCP = 20; % length of cyclic prefix
numPSC = 64; % 64 or 16, number of pilot subcarriers
numUE = 2;
numSC = 64; % number of subcarriers
numPSym = numUE; % number of pilot OFDM symbols per packet
numDSym = 1; % number of data OFDM symbol per packet
numSym = numPSym+numDSym; % number of OFDM symbols per packet
pilotSpacing = numSC/numPSC;
pilotStart = [1,1]; % pilot starting subcarrier for two users 

% Data symbol modulation
constQPSK = [1-1j;1+1j;-1+1j;-1-1j];
a = constQPSK(1);
b = constQPSK(2);
c = constQPSK(3);
d = constQPSK(4);
% Symbol combination class
symComb = [a a;a b;a c;a d;b a;b b;b c;b d;c a;c b;c c;c d;d a;d b;d c;d d]; 
labelClass = 1:1:size(symComb,1);
numLabel = length(labelClass);

% Noise computation
EsN0_dB = 40;
EsN0 = 10.^(EsN0_dB./10);
symRate = 2; % symbol rate, 2 symbol/s, sent from 2 users at the same time
Es = 1; % symbol energy, joules/symbol
sigPower = Es*symRate; % total signal power (2 symbols/s), joules/s = watts
symPower = sigPower/numUE; % signal power per symbol 
N0 = sigPower./EsN0; % noise power in watts/Hz, assuming subcarrier spacing = 1 Hz
bw = 1; % bandwidth per subcarrier, Hz
nPower = N0*bw; % total noise power in watts, freqency domain
nVar = nPower./2; % noise variance, frequency domain

% Power allocation in frequency domain
targetSNR_1 = 12; % dB, target SNR for strong user
targetSNR_2 = 12; % dB, target SNR for weak user
targetSNR_linear_1 = 10^(targetSNR_1/10);
targetSNR_linear_2 = 10^(targetSNR_2/10);
% Static channel realisation
numPath = 20;
h = 1/sqrt(2)/sqrt(numPath)*complex(randn(numPath,numUE),randn(numPath,numUE));
H = fft(h,numSC,1); 
gainH = (abs(H).^2).';
% Calculate power allocation factor and obtain decoding order
[powerFactor,decOrder] = allocatePower(symPower,gainH,targetSNR_1,targetSNR_2,nVar);

% Training data generation
numPacketClass = 3e4; % number of OFDM packets per label
% Fixed pilot symbols (BPSK modulation)
fixedPilot = zeros(numPSym,numPSC,numUE);
fixedPilot(1,:,1) = complex(sign(rand(1,numPSC,1)-0.5)); 
fixedPilot(2,:,2) = complex(sign(rand(1,numPSC,1)-0.5));
fixedPilotPacket = repmat(fixedPilot,1,1,1,numPacketClass); % use the same pilot for all packets
% Target subcarrier for signal detection
idx_sc = 20; 
XTrain = []; % training samples
YTrain = []; % labels

tic;
for n = 1:numLabel % generate training data for each class
    
    % Pilot symbols
    pilotFrame = zeros(numPSym,numSC,numUE,numPacketClass);
    pilotFrame(1,:,1,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacketClass)-0.5),sign(rand(1,numSC,1,numPacketClass)-0.5));
    pilotFrame(2,:,2,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacketClass)-0.5),sign(rand(1,numSC,1,numPacketClass)-0.5)); 
    % Replace pilot subcarriers with fixed pilot symbols
    pilotFrame(:,pilotStart(1):pilotSpacing:end,1,:) = fixedPilotPacket(:,:,1,:);
    pilotFrame(:,pilotStart(2):pilotSpacing:end,2,:) = fixedPilotPacket(:,:,2,:);

    % Data symbols
    dataFrame = 1/sqrt(2)*complex(sign(rand(numDSym,numSC,numUE,numPacketClass)-0.5),sign(rand(numDSym,numSC,numUE,numPacketClass)-0.5));
    % Replace random data symbols with current data combination on the
    % target subcarrier
    currentData = repmat(symComb(n,:),1,1,numPacketClass); 
    currentData = reshape(currentData,1,1,numUE,numPacketClass); 
    dataFrame(:,idx_sc,:,:) = 1/sqrt(2)*currentData;
    
    % Data transmission and reception
    hAll = repmat(h,1,1,numPacketClass);
    powerFactorAll = repmat(powerFactor,1,1,numPacketClass);
    decOrderAll =  repmat(decOrder,1,1,numPacketClass);
    transmitPacket = zeros(numSym,numSC,numUE,numPacketClass);
    transmitPacket(1:2,:,:,:) = pilotFrame;
    transmitPacket(end,:,:,:) = dataFrame;
    [receivePacket,~] = dataTransmissionReception(transmitPacket,powerFactorAll,lengthCP,hAll,nVar);
    
    % Construct feature vector and labels
    % Labels for the target subcarrier
    dataLabel = n*ones(1,numPacketClass);
    [feature,label,~] = getFeatureAndLabel(real(receivePacket),imag(receivePacket),dataLabel,n);
    featureVec = mat2cell(feature,size(feature,1),ones(1,size(feature,2))); % cell, 1 x #perClass, each cell, 384 x 1
    XTrain = [XTrain featureVec];
    YTrain = [YTrain label];
    
end
toc;

XTrain = XTrain.';
YTrain = categorical(YTrain.');

save('trainData.mat','XTrain','YTrain','h','numPSC','lengthCP','idx_sc','fixedPilot');


