clear

% take out names of animals/files from directory in which stuff is saved
directory = 'xxx';
files = dir(directory);
files = files(3 : end);
% load reversal +- 3 channels
ch2load = {[7 10 13]; [4 7 10]; [10 13 16]; ...
    [8 11 14]; [6 9 13]; [4 7 10]; [4 7 10]};
ExtractMode = 1; % extract from neuralynx into matlab
fs = 32000; % from data
downsampling_factor = 100; % downsample for LFP analysis
fsLFP = fs / downsampling_factor; % sampling rate LFP
fsVideo = 30; % sampling rate eye video
high_cut = fsLFP / 2; % nyquist
EMGfreqs = [30 300]; % freqs for EMG filtering
% set stuff for interpreter
block2start = 1; 
block_size = 10;
to_plot = 0;

for file_idx = 1
    disp(['loading animal ' num2str(file_idx)])
    % extract the subfolder (1st of non empty entries)
    subfolder = dir([directory files(file_idx).name]);
    subfolder = subfolder(3).name;
    % load 3 channels of LFP (reversal +- 3)
    idx = 1;
    for channel = ch2load{file_idx}
        file_to_load = [directory, files(file_idx).name, '\', subfolder, '\CSC', num2str(channel), '.ncs'];
        [~, signal, ~] = load_nlx_Modes(file_to_load, ExtractMode, []);
        signal = ZeroPhaseFilter(signal, fs, [0.1 high_cut]);
        LFP_EMG(idx, :) = signal(1 : downsampling_factor : end);
        idx = idx + 1;
    end
    % load EMG
    file_to_load = [directory, files(file_idx).name, '\', subfolder, '\EMG.ncs'];
    [~, signal, ~] = load_nlx_Modes(file_to_load, ExtractMode, []);
    signal = ZeroPhaseFilter(signal, fs, EMGfreqs);
    LFP_EMG(idx, :) = signal(1 : downsampling_factor : end); clear signal
    % load movement
    file_to_load = [directory, files(file_idx).name, '\', subfolder, '\MHC.ncs'];
    [~, signal, ~] = load_nlx_Modes(file_to_load, ExtractMode, []);
    movement = abs(signal(1 : downsampling_factor : end)); clear signal
    % load sync ephys
    file_to_load = [directory, files(file_idx).name, '\', subfolder, '\Sync.ncs'];
    [~, signal, ~] = load_nlx_Modes(file_to_load, ExtractMode, []);
    % downsample the sync signal & extract sync 
    signal = signal(1 : downsampling_factor : end);
    sync_ephys = find(signal > 0, 1, 'last');
    % cut LFP, EMG & movement
    LFP_EMG = LFP_EMG(:, sync_ephys : end);
    movement = movement(:, sync_ephys : end);
    disp(['analyzing animal ' num2str(file_idx)])
    ASS = AutomatedSleepScoring(LFP_EMG, movement, fsLFP, block_size, ...
        [directory files(file_idx).name '\SleepScoring'], to_plot);
    disp(['animal ' num2str(file_idx) ' NREM epochs ' num2str(nnz(ASS.NREM)) '/' num2str(numel(ASS.NREM))])
    disp(['animal ' num2str(file_idx) ' REM epochs ' num2str(nnz(ASS.REM)) '/' num2str(numel(ASS.REM))])
    save([directory files(file_idx).name '\SleepScoring'], 'ASS')
    clear LFP_EMG
end

