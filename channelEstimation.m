function [H_LS,H_MMSE] = channelEstimation(rData,pilotFrame,powerFactor,pStart,RHH,nVar,numPSC,H_perf)
% This function is to perform least square (LS) and minimum mean square
% error (MMSE) channel estimation based on [1].
% [1] O. Edfors, M. Sandell, J. -. van de Beek, S. K. Wilson and 
% P. Ola Borjesson, "OFDM channel estimation by singular value 
% decomposition," VTC, Atlanta, GA, USA, 1996, pp. 923-927 vol.2.

[~,numSC,numUE,numPacket] = size(pilotFrame);
pilotSpacing = numSC/numPSC;
H_LS = zeros(numSC,numUE,numPacket);
H_MMSE = zeros(numSC,numUE,numPacket);

for u = 1:numUE
    
    % Pilot symbols and data symbols
    pilot_sc = pStart(u):pilotSpacing:numSC;
    pilot = squeeze(pilotFrame(u,pilot_sc,u,:)); 
    pF = squeeze(powerFactor(pilot_sc,u,:));
    data = squeeze(rData(u,pilot_sc,:)); 
    
    intfPower = squeeze(powerFactor(:,setdiff(1:numUE,u),:)); % power of interfering user

    for p = 1:numPacket
                
        pl = sqrt(pF(:,p)).*pilot(:,p);
        hLS = data(:,p)./pl;
        hLS_interp = interp1(1:pilotSpacing:numSC,hLS,1:numSC,'spline','extrap');
        hLS_interp = hLS_interp.';
        H_LS(:,u,p) = hLS_interp;
        H_MMSE(:,u,p) = RHH*inv(RHH+eye(numSC)*nVar.*diag(intfPower(:,p)))*hLS_interp;

    end
    
end

% % Plot frequency-domain channel gains compared with the actual channel
% hLS = mean(abs(H_LS).^2,3);
% hMMSE = mean(abs(H_MMSE).^2,3);
% figure();plot(abs(H_perf(:,1)).^2,'-o');hold on;plot(hLS(:,1),'-x');hold on;plot(hMMSE(:,1),'-d');
% title('User 1');legend('Actual channel','LS','MMSE');
% figure();plot(abs(H_perf(:,2)).^2,'-o');hold on;plot(hLS(:,2),'-x');hold on;plot(hMMSE(:,2),'-d');
% title('User 2');legend('Actual channel','LS','MMSE');

