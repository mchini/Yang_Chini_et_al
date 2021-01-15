function ASS = AutomatedSleepScoring(LFP_EMG, movement, fsLFP, block_size)
% function to automatically sleep-score based on EEG & EMG in
% block_size * s epochs, with block_size/2 overlap

% inputs: LFP_EMG: block_size channels, first 3 are LFP, last two are EMG
%         fsLFP: sampling rate of LFP
%         block_size: size of scoring block (in seconds)

num_blocks = floor(size(LFP_EMG, 2) / (block_size * fsLFP));
%%%% now compute PSD %%%%
% set sampling rate
params.Fs = fsLFP;
% only analyze LFP until round number of instances
% second input is window size and overlap
[PSD, ~, freqs] = mtspecgramc(LFP_EMG(1:3, 1 : num_blocks * block_size * fsLFP)', ...
    [block_size block_size/2], params);
[PSD_EMG, ~, ~] = mtspecgramc(LFP_EMG(4, 1 : num_blocks * block_size * fsLFP)', ...
    [block_size block_size/2], params);
% average movement over block
avg_mov = NaN(size(PSD, 1), 1);
for idx = 1 : size(PSD, 1)
    avg_mov(idx) = sum(movement(block_size / 2 * fsLFP * (idx - 1) + 1 : block_size / 2 * fsLFP * (idx + 1)));
end


% define freq bands
deltaF = freqs > 1 & freqs < 4;
thetaF = freqs > 6 & freqs < 12;
EMGF = freqs > 30 & freqs < 300;
% compute delta, theta, their ratio and EMG power
deltaP = sum(mean(PSD(:, deltaF, :), 3), 2);
thetaP = sum(mean(PSD(:, thetaF, :), 3), 2);
EMGP = sum(mean(PSD_EMG(:, EMGF, :), 3), 2);
theta_delta = thetaP ./ deltaP;
% find REM & NREM states
REM = theta_delta > prctile(theta_delta, 75) & EMGP < prctile(EMGP, 25) & avg_mov < 5e6;
NREM = deltaP > prctile(deltaP, 70) & EMGP < prctile(EMGP, 50) & avg_mov < 5e6;
WAKE = EMGP > prctile(EMGP, 80) | avg_mov > 1e7;

% put stuff into a structure
ASS.REM = REM;
ASS.NREM = NREM;
ASS.WAKE = WAKE;
ASS.score = WAKE + NREM * 2 + REM * 3;
ASS.deltaP = deltaP;
ASS.thetaP = thetaP;
ASS.theta_delta = theta_delta;
ASS.EMGP = EMGP;
ASS.deltaF = deltaF;
ASS.thetaF = thetaF;
ASS.PSD = PSD;
ASS.PSD_EMG = PSD_EMG;
ASS.avg_mov = avg_mov;
ASS.freqs = freqs;

end