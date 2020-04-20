
%% clean up, load meta-data and set global variables
clear

experiments = load_meta_data_lower_thr(1); % load meta data
traces_folder = 'Folder with 2P traces here';
folder2save = 'Folder in which to save results here';
number_reps = 10; % how many times to repeat if n° of neurons is > than min n° of neurons for that mouse
gamma_range = 0.01 : 0.01 : 3.0; % resolution parameter
G = length(gamma_range);
M = 100;  % number of repeats
es = 10000;
alpha = 0.05;

%% load dataset and downsample it if cells > min cells, resample and loop


for exp_idx = 1 : size(experiments, 1)
    
    experiment = experiments(exp_idx);
    to_shorten = 0;
    done_with_recording = 0;
    idx_rep = 1; % repetitions for this recording (if it hs to be shortened, rep = 5)
    
    while done_with_recording < 1 % keep looping if recording has to be shortened and rep < 5
        
        if exist(strcat(folder2save, num2str(experiment.Mouse), '_', ...
                experiment.Condition, num2str(experiment.Rec_idx_cum), '_Traces.mat'), 'file')
            disp('already calculated')
            done_with_recording = 1;
        else
            
            disp(strcat('analyzing__', num2str(experiment.Mouse), '__', ...
                experiment.Condition, num2str(experiment.Rec_idx_cum), '__', ...
                'repetition__', num2str(idx_rep)))
            
            % load traces/spikes
            load(strcat(traces_folder, num2str(experiment.Mouse), filesep, ...
                experiment.Condition, num2str(experiment.Rec_idx_cum), '.mat'))
            % if there are more neurons that in the lowest number of neurons
            % for this animal, reduce the number of neurons accodingly
            num_neurons = size(Traces, 1);
            if num_neurons > experiment.MinCellsAllCond
                idx2keep = randsample(1 : size(Traces, 1), experiment.MinCellsAllCond);
                Traces = Traces(idx2keep, :);
                centroid = centroid(idx2keep, :);
                to_shorten = 1;
            end
            
            %% prepare correlation matrices
            % compute correlations
            corr_coeffs = corrcoef(Traces');
            % remove main diagonal and prevent corr_coeffs=1/-1 from becoming Inf or 0
            corr_coeffs = corr_coeffs .*~ eye(experiment.MinCellsAllCond);  %
            % fisher-transform
            corr_coeffs = atanh(corr_coeffs);
            corr_coeffs(corr_coeffs == 1) = 0.999; %
            corr_coeffs(corr_coeffs == -1) = - 0.999;
            % conver to double for later
            corr_coeffs = double(corr_coeffs);
            
            %% compute Euclidean distance
            
            ED = zeros(experiment.MinCellsAllCond);
            for row_idx = 1 : experiment.MinCellsAllCond
                for col_idx = 1 : experiment.MinCellsAllCond
                    ED(row_idx, col_idx) = sqrt((centroid(row_idx, 1) - centroid(col_idx, 1)) ^2 ...
                        + (centroid(row_idx, 2) - centroid(col_idx, 2)) ^2);
                end
            end
            
            %% perform 'standard' Louvain community detection
            
            qmax = zeros(G);
            cimax = zeros(experiment.MinCellsAllCond, G);
            
            parfor g = 1 : G
                for m = 1 : M
                    [ci, q] = community_louvain(corr_coeffs, gamma_range(g), [], 'negative_asym');
                    if (q > qmax(g))
                        qmax(g) = q;
                        cimax(:, g) = ci;
                    end
                end
            end
            
            %% MRCC as in Jeub et al., 2018
            
            [S, gamma] = eventSamples(corr_coeffs, es, 'Modularity',@const_modularity);
            [Sc, Tree] = hierarchicalConsensus(S, alpha);
            C = coclassificationMatrix(S);
            [Tree] = dendrogramSimilarity(C, Sc, Tree, 'SimilarityFunction', 'min');
            [Sall, p] = allPartitions(Sc, Tree);
            s_int = treeSort(C, Sc, Tree);
            
            %% put stuff into a structure that will be saved
            
            Complexity(idx_rep).corr_coeffs = corr_coeffs;
            Complexity(idx_rep).ED = ED;
            Complexity(idx_rep).qmax = qmax;
            Complexity(idx_rep).cimax = cimax;
            Complexity(idx_rep).Sc = Sc;
            Complexity(idx_rep).Tree = Tree;
            Complexity(idx_rep).C = C;
            Complexity(idx_rep).Sall = Sall;
            Complexity(idx_rep).p = p;
            Complexity(idx_rep).s_int = s_int;
            
            % continue looping if rep < number_reps and rec had to be shortened
            if to_shorten > 0
                idx_rep = idx_rep + 1;
                if idx_rep < number_reps
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