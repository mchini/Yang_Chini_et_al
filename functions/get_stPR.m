function [stPR, pop_coupling, pop_coupling_1sthalf, pop_coupling_2ndhalf] ...
    = get_stPR(spike_matrix, max_lag, animal_name, repeat_calc, save_data, output_folder)

%% by Mattia 09.19
% compute stPR (spike-triggered population rate) and population coupling
% as described in Okun et al., 2015 Nature.

% inputs:
% - animal_name (string): to save/loda stuff
% - spike_matrix (2D matrix): as computed by getSpikeMatrixKlusta
% - max_lag (value): in ms (tipical values are 100-500-1000ms)
% - repeat_calc (0 or 1): 0=no, 1=yes
% - save_data (0 or 1): 0=no, 1=yes
% - output_folder (string): main folder to save results

% outputs
% - stPR (array, num_units*(max_lag*2+1)): spike triggered population rate
% - pop_coupling (array, num_units): population coupling 
% - pop_coupling_1sthalf array, num_units): population coupling first half (quality control, 
%                                           shouldn't differ from second half)
% - pop_coupling_2ndhalf (array, num_units): population coupling second half


if repeat_calc == 0 && ...
        exist(strcat(output_folder, animal_name, '.mat'), 'file')
    load(strcat(output_folder, animal_name))
    stPR = pop_coupling_stuff.stPR;
    pop_coupling = pop_coupling_stuff.pop_coupling;
    pop_coupling_1sthalf = pop_coupling_stuff.pop_coupling_1sthalf;
    pop_coupling_2ndhalf = pop_coupling_stuff.pop_coupling_2ndhalf;
else
    Gwindow = gausswin(101, 8.3); % gaussian window of 100ms with stdev of 12ms as in paper
    Gwindow = Gwindow / sum(Gwindow); % normalize the gaussian kernel
    num_units = size(spike_matrix, 1);
    half_rec = round(length(spike_matrix) / 2);
    
    % initialize variables
    stPR = zeros(num_units, 2 * max_lag + 1);
    pop_coupling = zeros(num_units, 1);
    pop_coupling_1sthalf = pop_coupling;
    pop_coupling_2ndhalf = pop_coupling;
    
    if ~ isnan(spike_matrix)
        % loop over units
        for unit_idx = 1 : num_units           
            % compute the population rate, excluding the unit for which you
            % are calculating the stPR
            if num_units > 2
                population_rate = sum(spike_matrix(~ ismember(1 : num_units, unit_idx), :));
            else
                population_rate = spike_matrix(~ ismember(1 : num_units, unit_idx), :);
            end
            % convolve with gaussian window
            population_rate = conv(full(population_rate), Gwindow, 'same');
            % subtract mean
            population_rate = population_rate - mean(population_rate);
            % convolve also firing of the single neuron
            firing_rate_neuron = conv(full(spike_matrix(unit_idx, :)), Gwindow, 'same');
            % compute xcorr between the two vectors (stPR)
            stPR(unit_idx, :) = xcorr(firing_rate_neuron, population_rate, max_lag);
            % compute population coupling as in the paper (stPR at 0 normalized by
            % number of spikes fired by the neuron).
            pop_coupling(unit_idx) = stPR(unit_idx, max_lag + 1) / sum(firing_rate_neuron);
            % to compute reliability of pop_coupling, compute it for first and
            % second half separately, to than check how they correlate to each
            % other
            stPR_1sthalf = XcorrNB(cat(1, firing_rate_neuron(1 : half_rec), ...
                population_rate(1 : half_rec)), 0, 0);
            pop_coupling_1sthalf(unit_idx) = stPR_1sthalf / sum(firing_rate_neuron(1 : half_rec));
            stPR_2ndhalf = XcorrNB(cat(1, firing_rate_neuron(half_rec + 1 : end), ...
                population_rate(half_rec + 1 : end)), 0, 0);
            pop_coupling_2ndhalf(unit_idx) = stPR_2ndhalf / sum(firing_rate_neuron(half_rec + 1 : end));
            
            % put stuff into a structure
            pop_coupling_stuff = struct;
            pop_coupling_stuff.stPR = stPR;
            pop_coupling_stuff.pop_coupling = pop_coupling;
            pop_coupling_stuff.pop_coupling_1sthalf = pop_coupling_1sthalf;
            pop_coupling_stuff.pop_coupling_2ndhalf = pop_coupling_2ndhalf;
        end
    else
        pop_coupling_stuff = struct;
        pop_coupling_stuff.stPR = NaN;
        pop_coupling_stuff.pop_coupling = NaN;
        pop_coupling_stuff.pop_coupling_1sthalf = NaN;
        pop_coupling_stuff.pop_coupling_2ndhalf = NaN;
    end
    if save_data == 1
        save(strcat(output_folder, animal_name), 'pop_coupling_stuff')
    else
        disp('Data not saved!')
    end
end
end