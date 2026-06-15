 %==========================================================================
% EMGRAMPpreproc.m
% 
% This script processes raw EMG signals together with segmented kinematic 
% (RAMP) data on a per-trial basis.
% 
% For each selected trial:
%   1. Load raw EMG and corresponding segmented kinematic file (.mat).
%   2. Parse subject and trial identifiers from the filename.
%   3. Combine raw and segmented data into a unified structure (`EMGKin`).
%   4. Preprocess EMG:
%        - High-pass filter (default 30 Hz).
%        - Notch filters at 50, 100, 150 Hz (and Cometa accelerometer band ~142 Hz).
%   5. Global high-peak detection & cleaning (whole trial).
%   6. Segment EMG into stride epochs and resample each stride to fixed length.
%   7. Segment EMG by treadmill speed-change indices into speed-based segments.
%   8. For each segment separately:
%        - Automatic spike detection & cleaning (interpolation of high
%        peaks).GGLMr
%        - Additional long-artifact cleaning (walls).
%        - GUI for channel quality check and exclusion.
%   9. Normalize EMG within each segment:
%        - Options: 'norm', 'range', 'max', or 'max_segment'
%        - In 'max_segment' mode, peaks are taken from the maximum-speed segment.
%  10. Plot normalized EMG profiles for each segment, arranged by reference order.
%  11. Save results per trial into a `.mat` file.
%
% LAST UPDATED: 1 December 2025, modified for segment-wise + global spike cleaning
%==========================================================================

clc; clear;

%% ================= Settings =================
fs                   = 2000;      % Hz
highpass_cutoff      = 30;        % Hz
lowpass_cutoff       = 10;        % Hz
target_samples       = 200;       % for time normalization
fs_kin               = 100;
normalization_method = 'max_segment';    % 'norm' or 'range' or 'max' or 'max_segment'

ref_order = {'TAr','SOLr','PERr','GMr','GLr','RFr','VLr','VMr','BFr','SEMr',...
             'SARTr','GMEDr','TFLr','GLMr','ESr','TAl','SOLl','PERl','GMl',...
             'GLl','RFl','VLl','VMl','BFl','SEMl','SARTl','GMEDl','TFLl','GLMl','ESl'};

% Giulia Folder
root_preproc = 'path';
root_segm    = 'path';
save_plot    = 'path';
results_dir  = 'path';

%% ================ Manual Trial Selection ================
[rawFile, rawPath] = uigetfile('*.mat', 'Select raw EMG file', root_preproc);
if isequal(rawFile,0)
    error('No EMG file selected');
end

[~, rawBase, ~] = fileparts(rawFile);
tokens = regexp(rawBase, '^([^_]+)_([^_]+)_([^_]+)$', 'tokens', 'once');

if ~isempty(tokens) && numel(tokens) == 3
    subject_name = tokens{1};
    trial_name   = tokens{2};
    ID_trial     = tokens{3};
else
    warning('Filename format does not match expected pattern (e.g., Subject_Trial_ID)');
    subject_name = rawBase;
    trial_name   = '';
    ID_trial     = '';
end

rawDataStruct = load(fullfile(rawPath, rawFile));
rawField      = fieldnames(rawDataStruct);
RAMP_DATA.rawData = rawDataStruct.(rawField{1});
EMGKin.rawData    = rawDataStruct.(rawField{1});

% Select segmented kinematic .mat file
seg_subfolder = fullfile(root_segm);
seg_fullfile  = fullfile(seg_subfolder, rawFile);

if ~isfile(seg_fullfile)
    error('Segmented file not found: %s', seg_fullfile);
end

segDataStruct          = load(seg_fullfile);
segField               = fieldnames(segDataStruct);
RAMP_DATA_segm.segData = segDataStruct.(segField{1});
EMGKin.segData         = segDataStruct.(segField{1});

%% --- Extract EMG ---
emg_matrix        = EMGKin.rawData.EMG.data_matrix;
muscle_names_orig = EMGKin.rawData.EMG.column_names;

% Remove AD muscles
keep_idx = ~startsWith(muscle_names_orig, 'AD', 'IgnoreCase', true);
emg_matrix        = emg_matrix(:, keep_idx);
muscle_names      = muscle_names_orig(keep_idx);
muscle_names_pre_gui = muscle_names;

% === Fix for P683: GLr and GMr swapped in acquisition ===
if exist('subject_name','var') && strcmpi(subject_name,'P683')
    idx_GLr = find(strcmp(muscle_names, 'GLr'));
    idx_GMr = find(strcmp(muscle_names, 'GMr'));

    if ~isempty(idx_GLr) && ~isempty(idx_GMr)
        fprintf('>> Applying GLr/GMr channel swap correction for %s\n', subject_name);
        tmp = emg_matrix(:, idx_GLr);
        emg_matrix(:, idx_GLr) = emg_matrix(:, idx_GMr);
        emg_matrix(:, idx_GMr) = tmp;
    else
        warning('P683 correction requested but GLr or GMr not found in muscle_names.');
    end
end

% --- Gait events & stride indexing ---
Heel_Strike_R = EMGKin.rawData.GaitEvents.RHS;
stride_time_R = round(Heel_Strike_R * (fs/100)); % in EMG frames

stride_length_matrix = EMGKin.segData.stride_length_matrix;
RHEX                 = EMGKin.rawData.Marker.RHE.x;

% --- Stride speed & segmentation metadata ---
speed_treadmill        = EMGKin.segData.speed_treadmill;
constant_stride_speed  = EMGKin.segData.constant_stride_speed;
seg_constant_speed     = EMGKin.segData.seg_constant_speed;
min_seg_constant_speed = EMGKin.segData.min_seg_constant_speed;
change_speed_stride    = EMGKin.segData.change_speed_stride;
change_speed_sec       = EMGKin.segData.change_speed_sec;

%% --- Filter & envelope (global, before global spike cleaning) ---
[b_high, a_high] = butter(4, highpass_cutoff / (fs/2), 'high');
EMG_high = filtfilt(b_high, a_high, emg_matrix);

% Notch filters at 50 Hz, 100 Hz, 150 Hz
notch_freqs = [50, 100, 150];
for nf = notch_freqs
    wo = [nf - 1, nf + 1] / (fs/2);
    [b_notch, a_notch] = butter(4, wo, 'stop');
    EMG_high = filtfilt(b_notch, a_notch, EMG_high);
end

% Filter around 142 Hz - Cometa accelerometer
[ab,aa] = butter(4,[(142-2)/(fs/2) (142+2)/(fs/2)],'stop');                
EMG_high = filtfilt(ab, aa, EMG_high(:,:));

%% Envelope (rectification + low-pass)
EMG_rect = abs(hilbert(EMG_high));
[b_low, a_low] = butter(4, lowpass_cutoff / (fs/2), 'low');
EMG_processed  = filtfilt(b_low, a_low, EMG_rect);

%% Simple Plot EMG Power (pre high-peak cleaning)
FigureEmgsPower(emg_matrix, fs, muscle_names, EMG_high, EMG_processed, 0, [1,1,100], 1);

%% === GLOBAL High-Peak Detection & Cleaning (whole trial) ===
%   - coarse artifact removal before stride segmentation
emg_raw_global = EMG_processed;           % [samples x channels], BEFORE global cleaning

opt0 = struct( ...
    'spikeSDthreshold', 10, ...                   % conservative whole-trial threshold
    'spikeABSthreshold', [], ...                  % no absolute threshold
    'spikeInterval',    round(2/1000*fs), ...     % 2 ms
    'spikeFilterOptions', {{ 10*fs, 10/(fs/2) }} ...
);

opt0.function = @(x) spikeFilter( ...
    x, ...
    opt0.spikeSDthreshold, ...
    opt0.spikeABSthreshold, ...
    opt0.spikeFilterOptions, ...
    opt0.spikeInterval, ...
    'Dimension', 2);

yG = struct();
yG.trial{1,1} = emg_raw_global';                 % [channels x time]

[yG_clean, idxG] = cellfun(opt0.function, yG.trial, 'UniformOutput', false);

EMG_processed_globalClean = yG_clean{1,1}';      % [time x channels]
emgQualityFlag_global     = full(idxG{1,1})';    % [time x channels] logical

% ---- PLOT: Global automatic spike cleaning (styled like your reference) ----
% index_spikes global is: idxG{1,1}  [channels x time]
% yG_clean is:            yG_clean{1,1}  [channels x time]

index_spikes_global = idxG{1,1};
emg_before_global   = yG.trial{1,1};      % [channels x time] BEFORE cleaning
emg_after_global    = yG_clean{1,1};      % [channels x time] AFTER cleaning

% find channels containing at least one detected spike
rowsToPlot = find( any(index_spikes_global==1, 2) );
n          = numel(rowsToPlot);

if n > 0
    % grid layout
    cols = ceil(sqrt(n));
    rows = ceil(n / cols);

    figure('Name','Global High-Peak Cleaning','Color','w');
    clf;
    
    for k = 1:n
        ee = rowsToPlot(k);                % channel index

        subplot(rows, cols, k);
        hold on;

        % raw (before cleaning) in blue
        plot(emg_before_global(ee,:), 'b');

        % cleaned in black
        plot(emg_after_global(ee,:), 'k');

        % spike markers (red circles)
        spike_idx = find(index_spikes_global(ee,:));
        if ~isempty(spike_idx)
            plot(spike_idx, emg_before_global(ee, spike_idx), 'or');
        end

        hold off;
        title(muscle_names{ee}, 'Interpreter','none');
        xlabel('Samples');
        ylabel('Amplitude');
        grid on;
    end

    sgtitle(sprintf('Global High-Peak Detection – %s %s', ...
        subject_name, trial_name), ...
        'FontSize',14, 'FontWeight','bold');
end

% Overwrite EMG_processed with globally cleaned version
EMG_processed = EMG_processed_globalClean;

%% === Pre-Segment by Strides (Before Speed-Based Segmentation) ===
num_strides    = numel(stride_time_R) - 1;
stride_epochs  = cell(1, num_strides);

for i = 1:num_strides
    idx1 = stride_time_R(i);
    idx2 = stride_time_R(i+1);
    if idx2 <= size(EMG_processed,1)
        stride_epochs{i} = EMG_processed(idx1:idx2, :);
    else
        stride_epochs{i} = [];
    end
end

stride_data_3D = [];
valid_mask     = true(1, num_strides);

for i = 1:num_strides
    stride_n = stride_epochs{i};
    if isempty(stride_n) || size(stride_n,1) < 2
        valid_mask(i) = false;
        continue;
    end
    try
        resampled = resample(stride_n, target_samples, size(stride_n,1));
        stride_data_3D(:,:,i) = resampled;  % [samples x muscles x strides]
    catch
        valid_mask(i) = false;
    end
end

%% === Segment EMG by speed-change strides ===
change_idx = change_speed_stride(:);

if exist('trial_name','var') && ~isempty(trial_name) && ...
   contains(upper(trial_name),'RAMPUP') && ~any(change_idx == 1)
    change_idx = [1; change_idx];
end

num_good_strides = size(stride_data_3D, 3);

change_idx(change_idx < 1)                = 1;
change_idx(change_idx > num_good_strides) = num_good_strides;

edges        = [change_idx; num_good_strides + 1];
nSeg         = numel(edges) - 1;
EMG_segments = cell(nSeg, 1);

for iSeg = 1:nSeg
    stride_start = edges(iSeg);
    stride_end   = edges(iSeg+1) - 1;
    if stride_end < stride_start
        EMG_segments{iSeg} = [];
    else
        segment_strides = stride_data_3D(:, :, stride_start:stride_end);
        
        valid_constant_idx = intersect(1:size(segment_strides,3), constant_stride_speed);
        if numel(valid_constant_idx) >= min_seg_constant_speed
            mid_start = floor((numel(valid_constant_idx) - min_seg_constant_speed) / 2) + 1;
            mid_end   = mid_start + min_seg_constant_speed - 1;
            idx_crop  = valid_constant_idx(mid_start:mid_end);
            EMG_segments{iSeg} = segment_strides(:,:,idx_crop);
            fprintf('Segment %d: cropped to middle %d constant-speed strides.\n', ...
                    iSeg, min_seg_constant_speed);
        else
            warning('Segment %d: fewer than %d constant-speed strides (%d), using all.', ...
                iSeg, min_seg_constant_speed, numel(valid_constant_idx));
            EMG_segments{iSeg} = segment_strides;
        end
    end
end

%% === Segment-wise High-Spike Detection (opt1) ===
opt1 = struct( ...
    'spikeSDthreshold', 10, ...             
    'spikeABSthreshold', [], ...            
    'spikeInterval',    round(10/1000*fs), ... 
    'spikeFilterOptions', {{ 10*fs, 10/(fs/2) }} ...
);

opt1.function = @(x) spikeFilter( ...
    x, ...
    opt1.spikeSDthreshold, ...
    opt1.spikeABSthreshold, ...
    opt1.spikeFilterOptions, ...
    opt1.spikeInterval, ...
    'Dimension', 2);

emgQualityFlag_seg  = cell(nSeg,1);  
excluded_by_gui_seg = cell(nSeg,1);  

fprintf('--- Segment-wise EMG spike cleaning and channel check ---\n');

for seg = 1:nSeg
    segData = EMG_segments{seg};   % [samples × channels × strides]
    if isempty(segData)
        fprintf('Segment %d: empty, skipping.\n', seg);
        continue;
    end

    [nSamp, nCh, nStr] = size(segData);

    % ===== 1) High-Spike Detection & Cleaning (segment-wise) =====
    seg_raw2D = reshape(segData, nSamp*nStr, nCh);      % BEFORE cleaning

    y = struct();
    y.trial{1,1} = seg_raw2D';                          % spikeFilter expects [channels × time]

    [y_clean, index_spikes] = cellfun(opt1.function, y.trial, 'UniformOutput', false);

    seg_clean2D  = y_clean{1,1}';                       % [time × channels]
    idx_spikes2D = full(index_spikes{1,1})';            % [time × channels], FULL logical

    % Extra cleaning of long "wall" artefacts
    seg_clean2D = clean_long_artifacts(seg_clean2D, fs);

    seg_clean3D = reshape(seg_clean2D, nSamp, nCh, nStr);
    EMG_segments{seg} = seg_clean3D;                    % overwrite with CLEANED data
    emgQualityFlag_seg{seg} = reshape(idx_spikes2D, nSamp, nCh, nStr);

    % optional spike-visualization
    rowsToPlot = find( any(idx_spikes2D == 1, 1) );
    if ~isempty(rowsToPlot)
        nPlot = numel(rowsToPlot);
        cols  = ceil(sqrt(nPlot));
        rows  = ceil(nPlot / cols);
        figure('Name', sprintf('Segment %d - spike cleaning', seg), 'Color','w');
        for k = 1:nPlot
            ch = rowsToPlot(k);
            subplot(rows, cols, k);
            plot(seg_raw2D(:,ch)); hold on;
            plot(seg_clean2D(:,ch), 'k');
            plot(find(idx_spikes2D(:,ch)), seg_raw2D(idx_spikes2D(:,ch),ch), 'or');
            title(muscle_names{ch}, 'Interpreter','none');
        end
    end

    % ===== 2) Channel GUI =====
    t_seg = (0:size(seg_clean2D,1)-1) / fs;

    fprintf('Checking EMG quality for subject %s, trial %s, SEGMENT %d...\n', ...
        subject_name, trial_name, seg);

    excludeMask_seg = run_segment_channel_gui( ...
        seg, ...
        t_seg, ...
        seg_raw2D, ...   % raw (pre-clean) envelope
        seg_clean2D, ... % cleaned envelope
        muscle_names, ...
        subject_name, ...
        trial_name ...
    );  % [nCh x 1] logical

    % store per-segment mask (row vector)
    excluded_by_gui_seg{seg} = excludeMask_seg(:)';

    % === APPLY EXCLUSION ONLY TO THIS SEGMENT ===
    if any(excludeMask_seg)
        segData_clean = EMG_segments{seg};        % [samples x channels x strides]
        if ~isempty(segData_clean)
            % mark excluded channels as NaN in this segment only
            segData_clean(:, excludeMask_seg, :) = NaN;
            EMG_segments{seg} = segData_clean;
        end

        % optionally also clear quality flags for those channels
        if ~isempty(emgQualityFlag_seg{seg})
            qFlag = emgQualityFlag_seg{seg};
            qFlag(:, excludeMask_seg, :) = false;
            emgQualityFlag_seg{seg} = qFlag;
        end

        fprintf('Segment %d: excluded %d channels via GUI.\n', ...
                seg, sum(excludeMask_seg));
    end

    close all;
end

%% === 3. Normalize Each Speed-Based Segment from Cleaned Strides ===
nSeg = numel(EMG_segments);
normalized_emg_seg_all = cell(nSeg,1);
normalization_vmax     = cell(nSeg,1);
mean_profiles_seg      = cell(nSeg,1);

channel_peaks_maxseg = [];
channel_peaks_all    = [];

%% --- 3.1 Global channel peaks across ALL segments (ignoring NaNs) ---
if nSeg > 0
    all_max = [];
    for s = 1:nSeg
        seg = EMG_segments{s};
        if isempty(seg), continue; end

        seg_abs = abs(seg);
        seg_abs(isnan(seg_abs)) = 0;                        % ignore NaNs
        seg_max_over_samples = squeeze(max(seg_abs, [], 1));% [ch x strides]

        if isempty(all_max)
            all_max = seg_max_over_samples;
        else
            all_max = cat(2, all_max, seg_max_over_samples);
        end
    end

    if ~isempty(all_max)
        channel_peaks_all = max(all_max, [], 2);    % [ch x 1]
        channel_peaks_all(channel_peaks_all == 0) = NaN;
    end
end

%% --- 3.2 Determine max-speed segment (same logic as before) ---
channel_peaks = [];        % not used anymore, kept only if some code expects it
maxSegIdx     = [];

if exist('segment_speeds','var') && numel(segment_speeds) == nSeg
    [~, maxSegIdx] = max(segment_speeds);
elseif exist('segment_speed_mean','var') && numel(segment_speed_mean) == nSeg
    [~, maxSegIdx] = max(segment_speed_mean);
else
    seg_means = nan(nSeg,1);
    for s = 1:nSeg
        segDataTmp = EMG_segments{s};
        if isempty(segDataTmp)
            seg_means(s) = NaN;
        else
            seg_means(s) = mean(abs(segDataTmp(:)));
        end
    end
    if all(isnan(seg_means))
        warning('Could not determine max-speed segment (all EMG_segments empty). Will not use segment-based peaks.');
        maxSegIdx = [];
    else
        [~, maxSegIdx] = max(seg_means);
        fprintf('Proxy: selected segment %d as max-speed segment (largest mean abs EMG).\n', maxSegIdx);
    end
end

%% --- 3.3 Channel peaks from MAX-SPEED segment (ignoring NaNs) ---
if ~isempty(maxSegIdx)
    seg_max = EMG_segments{maxSegIdx};
    if isempty(seg_max)
        warning('Selected max-speed segment %d is empty — skipping segment-peak normalization.', maxSegIdx);
        maxSegIdx = [];
    else
        abs_seg = abs(seg_max);
        abs_seg(isnan(abs_seg)) = 0;                        % ignore NaNs
        max_over_samples = squeeze(max(abs_seg, [], 1));    % [ch x strides]
        channel_peaks_maxseg = max(max_over_samples, [], 2);% [ch x 1]
        channel_peaks_maxseg(channel_peaks_maxseg == 0) = NaN;

        fprintf('Computed channel peaks from segment %d for segment-based normalization.\n', maxSegIdx);
    end
end

%% --- 3.4 Normalize each segment ---
for seg = 1:nSeg
    resampled_strides = EMG_segments{seg};  
    
    if isempty(resampled_strides)
        warning('Segment %d: empty segment, skipping.', seg);
        continue;
    end

    num_strides = size(resampled_strides, 3);
    num_ch      = size(resampled_strides, 2);
    normalized_emg = zeros(size(resampled_strides));
    vmax_used      = NaN(num_ch, num_strides);

    for ch = 1:num_ch
        for stride_n = 1:num_strides
            vec = resampled_strides(:,ch,stride_n);

            % skip all-NaN channels in this segment
            if all(isnan(vec))
                normalized_emg(:,ch,stride_n) = NaN;
                vmax_used(ch,stride_n) = NaN;
                continue;
            end

            switch lower(normalization_method)
                case 'norm'
                    nf = norm(vec);
                    if nf > 0
                        nd = vec / nf;
                        normalized_emg(:,ch,stride_n) = nd - min(nd);
                        vmax_used(ch,stride_n) = nf;
                    end

                case 'range'
                    vmin = min(vec); vmax = max(vec);
                    if vmax > vmin
                        normalized_emg(:,ch,stride_n) = (vec - vmin) / (vmax - vmin);
                        vmax_used(ch,stride_n) = vmax - vmin;
                    end

                case 'max'
                    vmax = max(abs(vec));
                    if vmax > 0
                        normalized_emg(:,ch,stride_n) = vec / vmax;
                        vmax_used(ch,stride_n) = vmax;
                    end

                case {'max_segment','maxseg','max_speed_segment'}
                    % priority: max-speed segment → global peak → local peak
                    vmax = NaN;
                    if ~isempty(channel_peaks_maxseg) && numel(channel_peaks_maxseg) >= ch && ...
                       ~isnan(channel_peaks_maxseg(ch))
                        vmax = channel_peaks_maxseg(ch);
                    elseif ~isempty(channel_peaks_all) && numel(channel_peaks_all) >= ch && ...
                           ~isnan(channel_peaks_all(ch))
                        vmax = channel_peaks_all(ch);
                    else
                        vmax = max(abs(vec));   % fallback
                    end

                    if vmax > 0 && ~isnan(vmax)
                        normalized_emg(:, ch, stride_n) = vec / vmax;
                        vmax_used(ch,stride_n) = vmax;
                    else
                        normalized_emg(:, ch, stride_n) = NaN;
                        vmax_used(ch,stride_n) = NaN;
                    end

                otherwise
                    error('Unknown normalization method "%s"', normalization_method);
            end
        end
    end

    normalized_emg_seg_all{seg} = normalized_emg;
    normalization_vmax{seg}     = vmax_used;
    mean_profiles_seg{seg}      = mean(normalized_emg, 3);
    mean_prof                   = mean_profiles_seg{seg};

    % plotting as you already had...
    nCh_total = numel(ref_order);
    tnorm     = linspace(0, 100, size(normalized_emg, 1));
    
    nCols = ceil(sqrt(nCh_total));
    nRows = ceil(nCh_total / nCols);
    fig = figure( ...
        'Name', sprintf('%s %s: Segment %d (normalized)', subject_name, trial_name, seg), ...
        'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.8], ...
        'Color','w' ...
    );
    
    tiledlayout(nRows, nCols, 'Padding', 'compact', 'TileSpacing', 'compact');
    sgtitle(sprintf('%s – %s – Segment %d (normalized)', subject_name, trial_name, seg), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    for idxC = 1:nCh_total
        ax     = nexttile;
        muscle = ref_order{idxC};
        chan_idx = find(strcmp(muscle_names, muscle));
        if ~isempty(chan_idx) && chan_idx <= size(normalized_emg,2)
            epData = squeeze(normalized_emg(:, chan_idx, :));
            plot(ax, tnorm, epData, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5); 
            hold(ax, 'on');
            plot(ax, tnorm, mean_prof(:, chan_idx), 'Color', [0 0.4470 0.7410], 'LineWidth', 2);
            hold(ax, 'off');
            xlabel(ax, 'Gait (%)', 'FontSize', 10);
            ylabel(ax, 'Norm Amp', 'FontSize', 10);
            title(ax, muscle, 'FontSize', 10);
            grid(ax, 'on');
            ylim(ax, [0 1.4]);
            set(ax, 'FontSize', 8);
            if idxC == 1
                legend(ax, {'Epochs', 'Mean'}, 'FontSize', 8, 'Location', 'best');
            end
        else
            axis(ax, 'off');
            title(ax, muscle, 'FontSize', 10, 'Color', [0.5 0.5 0.5]);
        end
    end
    
    drawnow;
    
    if ~exist(save_plot,'dir')
        mkdir(save_plot);
    end
    save_name = sprintf('%s_seg%d_normEMG.png', rawBase, seg);
    save_path = fullfile(save_plot, save_name);
    exportgraphics(fig, save_path, 'Resolution', 300);
    close(fig);
end

%% ---- Build GUI-exclusion metadata (union + per segment) ----
global_exclude = false(1, numel(muscle_names));   % union over segments
excluded_by_gui_per_segment = cell(nSeg, 1);      % cell of muscle-name lists

for seg = 1:nSeg
    m = excluded_by_gui_seg{seg};
    if isempty(m)
        excluded_by_gui_per_segment{seg} = {};
        continue;
    end

    m = logical(m(:)');   % ensure row logical
    global_exclude = global_exclude | m;

    % store muscle names excluded in THIS segment
    excluded_by_gui_per_segment{seg} = muscle_names(m);
end

if any(global_exclude)
    fprintf('Muscles excluded in at least one segment: %d\n', sum(global_exclude));
    excluded_by_gui_names = muscle_names(global_exclude);   % union list
else
    excluded_by_gui_names = {};
end

%% ======= Build Processing Metadata Struct =======
excluded_by_prefix = muscle_names_orig(~keep_idx);

emg_proc = struct();
emg_proc.pipeline_version = '1.0';
emg_proc.timestamp        = datestr(now,'yyyy-mm-dd HH:MM:SS');

emg_proc.fs_emg                  = fs;
emg_proc.fs_kin                  = fs_kin;
emg_proc.target_samples          = target_samples;
emg_proc.normalization.method    = normalization_method;
emg_proc.normalization.ref_order = ref_order;

emg_proc.filtering = struct();
emg_proc.filtering.highpass.design   = 'butter';
emg_proc.filtering.highpass.order    = 4;
emg_proc.filtering.highpass.cutoffHz = highpass_cutoff;
emg_proc.filtering.highpass.mode     = 'zero-phase filtfilt';

emg_proc.filtering.notch.design   = 'butter';
emg_proc.filtering.notch.order    = 4;
emg_proc.filtering.notch.freqsHz  = notch_freqs(:)';
emg_proc.filtering.notch.bandHz   = 1;
emg_proc.filtering.notch.mode     = 'zero-phase filtfilt';
emg_proc.filtering.extra_stopband_142Hz = struct( ...
    'design','butter', ...
    'order',4, ...
    'centerHz',142, ...
    'halfbandHz',2, ...
    'mode','zero-phase filtfilt' );

emg_proc.rectification.method = 'abs(hilbert(x))';

emg_proc.envelope_lowpass.design   = 'butter';
emg_proc.envelope_lowpass.order    = 4;
emg_proc.envelope_lowpass.cutoffHz = lowpass_cutoff;
emg_proc.envelope_lowpass.mode     = 'zero-phase filtfilt';

% Spike detection / cleaning metadata
if exist('opt0','var')
    emg_proc.spike_cleaning_global.method           = 'spikeFilter';
    emg_proc.spike_cleaning_global.SD_threshold     = opt0.spikeSDthreshold;
    emg_proc.spike_cleaning_global.ABS_threshold    = opt0.spikeABSthreshold;
    emg_proc.spike_cleaning_global.interval_samples = opt0.spikeInterval;
    emg_proc.spike_cleaning_global.interval_ms      = opt0.spikeInterval / fs * 1000;
    emg_proc.spike_cleaning_global.filter_opts      = opt0.spikeFilterOptions;
else
    emg_proc.spike_cleaning_global = [];
end

emg_proc.spike_cleaning_segment.method           = 'spikeFilter';
emg_proc.spike_cleaning_segment.SD_threshold     = opt1.spikeSDthreshold;
emg_proc.spike_cleaning_segment.ABS_threshold    = opt1.spikeABSthreshold;
emg_proc.spike_cleaning_segment.interval_samples = opt1.spikeInterval;
emg_proc.spike_cleaning_segment.interval_ms      = opt1.spikeInterval / fs * 1000;
emg_proc.spike_cleaning_segment.filter_opts      = opt1.spikeFilterOptions;

% Quality flags
if exist('emgQualityFlag_global','var')
    emg_proc.quality_flag.global_size   = size(emgQualityFlag_global);
else
    emg_proc.quality_flag.global_size   = [];
end
emg_proc.quality_flag.segments_sizes = cellfun(@size, emgQualityFlag_seg, 'UniformOutput', false);

% Channel handling 
emg_proc.channels.original_names        = muscle_names_orig(:)';      % before AD removal
emg_proc.channels.after_AD_removal      = muscle_names_pre_gui(:)';   % after AD removal
emg_proc.channels.final_names           = muscle_names(:)';           % same across segments
emg_proc.channels.excluded_by_prefix_AD = excluded_by_prefix(:)';     % AD muscles removed

% union of muscles excluded in at least one segment (GUI)
emg_proc.channels.excluded_by_gui_union = excluded_by_gui_names(:)';

% per-segment GUI exclusions: cell array, one cell per segment,
% each containing a cell array of muscle names excluded in THAT segment
emg_proc.channels.excluded_by_gui_per_segment = excluded_by_gui_per_segment;

%% ======= Save Final Results =======
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

S = struct();
S.subject_name       = subject_name;
S.trial_name         = trial_name;
S.muscle_names       = muscle_names;
S.norm_emg_seg       = normalized_emg_seg_all;
S.mean_emg_seg       = mean_profiles_seg;
S.target_samples     = target_samples;
S.normalization      = normalization_method;
S.segmented_data     = RAMP_DATA_segm.segData;
S.raw_data           = RAMP_DATA.rawData;
S.emgQualityFlag_seg = emgQualityFlag_seg;
if exist('emgQualityFlag_global','var')
    S.emgQualityFlag_global = emgQualityFlag_global;
end
S.EMG_processed_env  = EMG_processed;        % already globally cleaned
S.EMG_segments_clean = EMG_segments;
S.emg_processing     = emg_proc;

save_file = fullfile(results_dir, sprintf('%s.mat', rawBase));
save(save_file, '-struct', 'S');
fprintf('Saved results to: %s\n', save_file);

%% ======================== LOCAL FUNCTIONS ========================

function onTableSelect(src, event)
    if isempty(event.Indices), return; end
    row = event.Indices(1);
    updatePlotsByIndex(src, row);
end

function updatePlotsByIndex(src, chIdx)
    figParent = ancestor(src, 'figure');
    handles   = guidata(figParent);
    cla(handles.hAxRaw);
    plot(handles.hAxRaw, handles.t, handles.emg_clean(:, chIdx));
    title(handles.hAxRaw, sprintf('Raw EMG: %s', handles.muscle_names{chIdx}), 'Interpreter','none');
    xlabel(handles.hAxRaw, 'Time (s)');
    cla(handles.hAxProc);
    plot(handles.hAxProc, handles.EMG_processed(:, chIdx));
    title(handles.hAxProc, sprintf('Processed EMG: %s', handles.muscle_names{chIdx}), 'Interpreter','none');
    xlabel(handles.hAxProc, 'Time (s)');
end

function corrected_indices = speed_change_gui(speed_signal, initial_indices)
    f = figure('Name', 'Edit Speed Change Points', ...
               'NumberTitle', 'off', ...
               'CloseRequestFcn', @onClose, ...
               'Units', 'normalized', ...
               'Position', [0.1, 0.1, 0.8, 0.6]);

    hold on;
    plot(speed_signal, 'b-', 'DisplayName', 'Speed (km/h)', 'LineWidth', 1.5);
    hMarkers = scatter(initial_indices, speed_signal(initial_indices), ...
                       80, 'g', 'filled', 'DisplayName', 'Change Points');
    
    title('Left click = Add | Right click = Delete | Close to Save');
    xlabel('Index');
    ylabel('Speed (km/h)');
    legend();
    grid on;

    guiData.hMarkers = hMarkers;
    guiData.signal   = speed_signal;
    guidata(f, guiData);

    set(f, 'WindowButtonDownFcn', @mouse_click);
    uiwait(f);
    corrected_indices = evalin('base', 'CorrectedSpeedChangeIndices');
    evalin('base', 'clear CorrectedSpeedChangeIndices');

    function mouse_click(~, ~)
        d = guidata(f);
        clickType = get(f, 'SelectionType');
        coords    = get(gca, 'CurrentPoint');
        x         = round(coords(1,1));
        if x < 1 || x > length(d.signal), return; end
        if strcmp(clickType, 'normal')
            if ~ismember(x, d.hMarkers.XData)
                d.hMarkers.XData(end+1) = x;
                d.hMarkers.YData(end+1) = d.signal(x);
            end
        elseif strcmp(clickType, 'alt')
            if isempty(d.hMarkers.XData), return; end
            [~, idx] = min(abs(d.hMarkers.XData - x));
            d.hMarkers.XData(idx) = [];
            d.hMarkers.YData(idx) = [];
        end
        guidata(f, d);
    end

    function onClose(src, ~)
        d = guidata(src);
        corrected = sort(round(d.hMarkers.XData));
        assignin('base', 'CorrectedSpeedChangeIndices', corrected);
        uiresume(src);
        delete(src);
    end
end

function excludeMask = run_segment_channel_gui(segIdx, t, raw2D, clean2D, muscle_names, subject_name, trial_name)
    nCh = numel(muscle_names);

    fig = figure('Name', sprintf('%s - %s - SEGMENT %d - EMG Channel Check', ...
                    subject_name, trial_name, segIdx), ...
                 'NumberTitle','off','Units','normalized','Position',[0.1 0.1 0.8 0.8]);

    data = [num2cell(false(nCh,1)), muscle_names(:)];
    hTable = uitable(fig, ...
        'Data', data, ...
        'ColumnName', {'Exclude','Muscle'}, ...
        'ColumnEditable', [true false], ...
        'ColumnFormat', {'logical','char'}, ...
        'Units','normalized', ...
        'Position', [0.01 0.1 0.3 0.88], ...
        'CellSelectionCallback', @onTableSelect);

    hAxRaw  = axes('Parent', fig, 'Units','normalized', 'Position',[0.35 0.55 0.62 0.4]);
    hAxProc = axes('Parent', fig, 'Units','normalized', 'Position',[0.35 0.05 0.62 0.4]);
    title(hAxRaw,  'Raw EMG (segment-concat)');
    xlabel(hAxRaw,'Time (s)'); ylabel(hAxRaw,'Amplitude');
    title(hAxProc, 'Processed EMG (segment-concat)');
    xlabel(hAxProc,'Time (s)'); ylabel(hAxProc,'Amplitude');

    uicontrol(fig, 'Style','text', 'Units','normalized', ...
        'Position',[0.01 0.98 0.98 0.02], ...
        'String', sprintf('Segment %d – Check boxes to mark muscles for exclusion. Click a row to preview. Then press "Confirm Exclusion".', segIdx), ...
        'HorizontalAlignment','left', 'FontSize',10);

    uicontrol(fig, 'Style','pushbutton', 'String','Confirm Exclusion', ...
        'Units','normalized', 'Position',[0.01 0.02 0.2 0.05], ...
        'FontSize',12, 'Callback', @(~,~) uiresume(fig));

    handles.emg_clean     = raw2D;
    handles.EMG_processed = clean2D;
    handles.muscle_names  = muscle_names;
    handles.t             = t;
    handles.hAxRaw        = hAxRaw;
    handles.hAxProc       = hAxProc;
    handles.hTable        = hTable;
    guidata(fig, handles);

    updatePlotsByIndex(hTable, 1);
    uiwait(fig);

    tableData   = get(hTable, 'Data');
    excludeMask = cell2mat(tableData(:,1));   

    close(fig);
end

function x_clean2D = clean_long_artifacts(x2D, fs)
% CLEAN_LONG_ARTIFACTS removes long, very large artefact "walls"
% x2D: [time x channels] envelope (after spikeFilter)
% fs : sampling frequency (Hz)

    x_clean2D = x2D;
    [nSamples, nCh] = size(x2D);

    min_dur_ms = 10;               % minimum duration of an artefact (ms)
    min_len    = max(1, round(min_dur_ms/1000 * fs));
    kMAD       = 10;               % how far above median+MAD we call it artefact

    for ch = 1:nCh
        x  = x2D(:,ch);
        ax = abs(x);

        % robust threshold based on median + k * MAD
        medA = median(ax, 'omitnan');
        madA = mad(ax, 1);        % median abs deviation
        if ~isfinite(medA) || ~isfinite(madA) || madA == 0
            continue;
        end
        thr = medA + kMAD * madA;

        % samples above threshold
        mask = ax > thr;
        if ~any(mask), continue; end

        % find contiguous segments in mask
        d = diff([false; mask; false]);
        starts = find(d == 1);
        ends   = find(d == -1) - 1;

        for r = 1:numel(starts)
            i1 = starts(r);
            i2 = ends(r);

            % only treat as artefact if sufficiently long
            if (i2 - i1 + 1) < min_len
                continue;
            end

            % borders for interpolation
            L = i1 - 1;
            R = i2 + 1;

            if L < 1 && R > nSamples
                % whole signal is artefact – skip
                continue;
            elseif L < 1
                % artefact at start: replace with first valid point
                x(i1:i2) = x(R);
            elseif R > nSamples
                % artefact at end: replace with last valid point
                x(i1:i2) = x(L);
            else
                % interpolate linearly between borders
                x(i1:i2) = interp1([L R], x([L R]), i1:i2, 'linear');
            end
        end

        x_clean2D(:,ch) = x;
    end
end
