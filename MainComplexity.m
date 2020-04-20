%%
% This script is adapted from the repo here below, which refers to the
% paper by Wenzel et al., 2019 also linked here
% https://github.com/hanshuting/tsne_neural_states/blob/master/embedding_state_analysis.m
% https://www.cell.com/cell-systems/pdf/S2405-4712(19)30077-8.pdf

%% set a few global variables

clear
experiments = load_meta_data_lower_thr; % load meta data
traces_folder = 'Folder with 2P traces here';
folder2save = 'Folder in which to save results here';
number_reps = 10; % how many times to repeat if n° of neurons is > than min n° of neurons for that mouse
samplingFreq = 30; % in Hz
downsamp_factor = samplingFreq / 10; % downsample to 10 Hz
perplexities = 5 : 10 : 45; % range of tSNE perplexities

%% compute and save parameters into a structure

for exp_idx = 1 : size(experiments, 1)
    
    experiment = experiments(exp_idx);
    % if the n° of neurons in the recording is > than the minimun n° of
    % neurons for that animal, random sample a n° of neurons = min n° of
    % neurons and repeat (same applies for length of recording)
    to_shorten = 0;
    done_with_recording = 0;
    idx_rep = 1;
    
    % keep looping if recording has to be shortened and rep < number_reps
    while done_with_recording < 1 
        
        disp(strcat('analyzing__', num2str(experiment.Mouse), '__', ...
            experiment.Condition, num2str(experiment.Rec_idx_cum), '__', ...
            'repetition__', num2str(idx_rep)))
        
        % load traces/spikes
        load(strcat(traces_folder, num2str(experiment.Mouse), filesep, ...
            experiment.Condition, num2str(experiment.Rec_idx_cum), '.mat'))
        % if there are more neurons that in the lowest number of neurons
        % for this animal, reduce the number of neurons accodingly
        if size(Traces, 1) > experiment.MinCellsAllCond
            idx2keep = randsample(1 : size(Traces, 1), experiment.MinCellsAllCond);
            Traces = Traces(idx2keep, :);
            to_shorten = 1;
        end
        % if there are more recording epochs that in the lowest number of 
        % rec epochs for this animal, reduce the number accordingly
        if size(Traces, 2) > experiment.MinLength
            idx2keep = randsample(1 : size(Traces, 2), experiment.MinLength);
            Traces = Traces(:, idx2keep);
            to_shorten = 1;
        end
        % downsample to 100ms - take care that not always the
        % matrix is disible by three (30Hz -> 10Hz)
        try
            DownsampTraces = squeeze(mean(reshape(Traces, ...
                size(Traces, 1), downsamp_factor, []), 2));
        catch
            numRows = floor(size(Traces, 2) ./ downsamp_factor);
            DownsampTraces = squeeze(mean(reshape(Traces(:, 1 : numRows * downsamp_factor), ...
                size(Traces, 1), downsamp_factor, []), 2));
        end
        
        % z-score Traces as this is needed for both PCA and t-SNE
        DownsampTraces = zscore(DownsampTraces, [], 2);
        
        % search for perplexity value that minimizes the reconstruction
        % error
        perplexities = perplexities(perplexities < size(DownsampTraces, 2));
        idx = 1;
        error = zeros(1, numel(perplexities));
        % limit the number of PCA components to speed up execution
        if size(DownsampTraces, 1) < 30
            PCAcomp = size(DownsampTraces, 1);
        else
            PCAcomp = 30;
        end
        for perplexity = perplexities
            [~, error(idx)] = tsne(DownsampTraces', 'Perplexity', perplexity, 'NumPCAComponents', PCAcomp);
            idx = idx + 1;
        end
        % select the perplexity with smallest reconstruction error
        embedded = tsne(DownsampTraces', 'Perplexity', perplexities(error == min(error)), ...
            'NumPCAComponents', PCAcomp);
        
        maxVal = round(max(abs(embedded(:))) * 1.1); % as per github page above
        sigma = maxVal / 40; % as per Wenzel paper. sigma for gaussian smoothing
        numPoints = 501; % as per github page
        rangeVals = [-maxVal maxVal]; % as per github page
        
        % creates a 2d probability distribution, gaussian convolved, starting from
        % 2d tSNE coordinates
        [xx, densAll] = findPointDensity(embedded, sigma, numPoints, rangeVals);
        maxDensity = max(densAll(:));
        minDensity = min(densAll(:));
        
        % threshold the image by first subtracting background with standard rolling
        % ball method, then using a median filter and finally thresholding.
        rolling_ball = strel('disk', 25);
        background = imopen(densAll, rolling_ball);
        FiltNoBack = medfilt2(densAll - background, [5 5]);
        GrayFiltNoBack = mat2gray(FiltNoBack, double([minDensity maxDensity]));
        BW = imbinarize(GrayFiltNoBack);
        
        % watershed the image and take care of over-segmentation
        DistImage = - bwdist(~ BW);
        WatershedBoundaries = watershed(DistImage);
        mask = imextendedmin(DistImage, 1);
        boundaries = bwboundaries(mask);
        % number of clusters is equal to number of segmented boundaries
        num_clustTSNE = numel(boundaries);
        
        %% ap cluster
        
        % create distance map
        S = 1 - pdist2(DownsampTraces', ...
            DownsampTraces', 'cosine');
        % cluster
        idx_ap = apcluster(S, median(S));
        num_clustAP = nnz(unique(idx_ap));
        
        %% pca
        
        [coeffs, score, ~, ~, explainedVar] = pca(DownsampTraces');
        explainedVar = cumsum(explainedVar);
        explainedVar = explainedVar / explainedVar(end);
        explained_states = find(explainedVar > 0.9, 1);
        
        %% put stuff into a structure
        
        Complexity.TSNEclust(idx_rep) = num_clustTSNE;
        Complexity.ProbDensity(idx_rep, :, :) = densAll;
        Complexity.mask(idx_rep, :, :) = mask;
        Complexity.APclust(idx_rep) = num_clustAP;
        Complexity.IdxAPclust(idx_rep, :) = idx_ap;
        Complexity.PCAclust(idx_rep, :) = explained_states;
        Complexity.explainedVar(idx_rep, :) = explainedVar;
        Complexity.score(idx_rep, :, :) = score;
        Complexity.coeffs(idx_rep, :, :) = coeffs;
        
        % continue looping if rep < 5 and rec had to be shortened
        if to_shorten > 0
            idx_rep = idx_rep + 1;
            if idx_rep > number_reps
                done_with_recording = 1;
            end
        else
            done_with_recording = 1;
        end
        % save only if done with recording
        if done_with_recording > 0
            save(strcat(folder2save, num2str(experiment.Mouse), '_', ...
                experiment.Condition, num2str(experiment.Rec_idx_cum), '_Traces'), 'Complexity')
            clear Complexity
        end
    end
end
