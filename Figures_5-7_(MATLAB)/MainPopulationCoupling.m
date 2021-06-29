
%% clean up set global variables

clear

experiments = load_meta_data_lower_thr(1); % load meta data
traces_folder = 'Folder with 2P traces here';
folder2save = 'Folder in which to save results here';
number_reps = 10; % how many times to repeat if n° of neurons is > than min n° of neurons for that mouse
max_lag = 1200; % in ms, max lag for which to compute pop coupling
fs = 30; % sampling rate
repeat_calc = 0;
save_data = 0;

%% load dataset and downsample it if cells > min cells, resample and loop

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
        
        if exist(strcat(folder2save, num2str(experiment.Mouse), '_', ...
                experiment.Condition, num2str(experiment.Rec_idx_cum), '_Traces.mat'), 'file')
            disp('already calculated')
            done_with_recording = 1;
        else
            
            disp(strcat('analyzing__', num2str(experiment.Mouse), '__', ...
                experiment.Condition, num2str(experiment.Rec_idx_cum), '__', ...
                'repetition__', num2str(idx_rep)))
            
            % load spikes
            load(strcat(traces_folder, num2str(experiment.Mouse), filesep, ...
                experiment.Condition, num2str(experiment.Rec_idx_cum), '.mat'))
            clear Traces SpikesSEQ centroid
            % if there are more neurons that in the lowest number of neurons
            % for this animal, reduce the number of neurons accodingly
            num_neurons = size(Spikes, 1);
            if num_neurons > experiment.MinCellsAllCond
                idx2keep = randsample(1 : size(Spikes, 1), experiment.MinCellsAllCond);
                Spikes = Spikes(idx2keep, :);
                to_shorten = 1;
            end           
                      
            %% compute pop coupling
            
            [stPR, pop_coupling] = get_stPR(Spikes, max_lag, fs, [], repeat_calc, save_data, folder2save);
            
            %% put stuff into a structure that will be saved
            
            Complexity(idx_rep).stPR = stPR;
            Complexity(idx_rep).pop_coupling = pop_coupling;
            Complexity(idx_rep).max_lag= max_lag;
            Complexity(idx_rep).num_neurons = num_neurons;
            
            % continue looping if rep < number_reps and rec had to be shortened
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
end