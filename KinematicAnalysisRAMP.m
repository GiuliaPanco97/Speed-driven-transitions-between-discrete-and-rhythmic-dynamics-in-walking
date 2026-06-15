% ========================================================================
% KinematicAnalysisRAMPGUI.m
% Script for kinematic gait analysis and extraction of discrete/rhythmic 
% movement parameters from RAMP trials.
% 
% This script:
% - Loads preprocessed subject data (.mat files) from a user-specified folder
% - Selects only RAMP and RAMPDOWN trials
% - Filters key lower-limb marker trajectories with a low-pass Butterworth filter
% - Computes spatiotemporal gait parameters:
%     * Step time, stride time, cadence
%     * Step and stride length (STEP LENGTH IN cm)
%     * Stance, swing, and double-support durations and percentages
% - Detects treadmill speed and automatically identifies constant-speed blocks
%     * Outlier correction using MAD-based method
%     * Interactive GUI for manual refinement of speed change points
% - Calculates discrete and rhythmic movement smoothness metrics:
%     * Logarithmic Dimensionless Jerk (LDJ)
%     * Spectral Arc Length (SAL)
% - Identifies movement zero-crossings to segment gait into primitives
% - Segments all gait and smoothness parameters according to treadmill 
%   speed-change indices
% - Stores results in structured variables:
%     * Global parameter matrices (per segmented speed interval)
%     * Subject- and trial-specific parameter arrays
%
% LAST UPDATED: 28 November 2025 
% ========================================================================
clc; clear;

subject_id   = 'P*'; % Change this
load_dir     = fullfile('path', subject_id); % Change this
filePath_save = 'path';      % Change this
saveDir       = 'path';                    % Change this

if ~isfolder(load_dir), error('The specified load_dir does not exist: %s', load_dir); end
if ~isfolder(filePath_save), mkdir(filePath_save); end
if ~isfolder(saveDir), mkdir(saveDir); end

mat_files = dir(fullfile(load_dir, '*.mat'));

RAMP_DATA   = struct();
ramp_count  = 0;

% ======================== LOAD ONLY RAMP TRIALS ==========================
for idx = 1:length(mat_files)
    fname = mat_files(idx).name;
    [~, baseName] = fileparts(fname);
    if contains(baseName, 'RAMP', 'IgnoreCase', true)
        ramp_count = ramp_count + 1;
        fullpath   = fullfile(load_dir, fname);
        tmp        = load(fullpath, baseName);
        if isfield(tmp, baseName)
            RAMP_DATA.(baseName) = tmp.(baseName);
        else
            warning('Variable %s not found in %s. Skipping.', baseName, fname);
        end
    end
end

if ramp_count == 0
    error('No RAMP/RAMPDOWN trials found in %s.', load_dir);
end

fieldNames = fieldnames(RAMP_DATA);

% ======================== MAIN LOOP OVER TRIALS ==========================
for j = 1:length(fieldNames)
    fullName      = fieldNames{j};
    trialData     = RAMP_DATA.(fullName);
    parameters_struct = struct();  % reset per trial
    % ---------- subject / trial labels (needed everywhere) ----------
    underscoreSplit = split(fullName, '_');
    subject = underscoreSplit{1};
    trial   = strjoin(underscoreSplit(2:end), '_');

    % ----------------- BASIC SETTINGS -----------------
    freq_rate_kin = trialData.Settings.Kinematic.VideoSamplingRate;
    dt_kin        = 1 / freq_rate_kin;

    Heel_Strike   = trialData.GaitEvents.HeelStrike.Locations; % combined HS (R+L)
    RHS           = trialData.GaitEvents.RHS;
    LHS           = trialData.GaitEvents.LHS;
    RTO           = trialData.GaitEvents.RTO;
    LTO           = trialData.GaitEvents.LTO;

    % ----------------- MARKERS (X-coordinates used below) -----------------
    RHEX = trialData.Marker.RHE.x;
    RVMX = trialData.Marker.RVM.x;
    RLMX = trialData.Marker.RLM.x;
    LHEX = trialData.Marker.LHE.x;
    LLMX = trialData.Marker.LLM.x;

    % -------- Gather XYZ markers, RAW + FILTERED --------
    marker_basenames = {'RGH','RELB','RWRI','RIL','RGT','RLE','RLM','RHE','RVM', ...
                        'LGH','LELB','LWRI','LIL','LGT','LLE','LLM','LHE','LVM'};
    
    raw_marker      = struct();
    filtered_marker = struct();
    col_names       = {};
    M               = [];

    for m = 1:numel(marker_basenames)
        base = marker_basenames{m};
        if ~isfield(trialData.Marker, base)
            warning('Marker %s not found in %s. Skipping.', base, fullName);
            continue
        end
        x = trialData.Marker.(base).x(:);
        y = trialData.Marker.(base).y(:);
        z = trialData.Marker.(base).z(:);

        raw_marker.(base).x = x; % AP
        raw_marker.(base).y = y; % ML
        raw_marker.(base).z = z; % Vertical

        M = [M, x, y, z]; 
        col_names = [col_names,{sprintf('%s_x',base), sprintf('%s_y',base), sprintf('%s_z',base)}]; 
    end

    % Low-pass filter 10 Hz
    if ~isempty(M)
        Wn = 10 / (freq_rate_kin/2);
        if Wn >= 1
            error('Cutoff frequency too high relative to sampling freq (Wn=%.3f).', Wn);
        end
        [b,a] = butter(2, Wn, 'low');
        Mf = filtfilt(b, a, M);

        col = 1;
        for m = 1:numel(marker_basenames)
            base = marker_basenames{m};
            if ~isfield(raw_marker, base)
                continue
            end
            filtered_marker.(base).x = Mf(:, col+0);
            filtered_marker.(base).y = Mf(:, col+1);
            filtered_marker.(base).z = Mf(:, col+2);
            col = col + 3;
        end
    end

    parameters_struct.raw_marker      = raw_marker;
    parameters_struct.filtered_marker = filtered_marker;
    parameters_struct.GaitEvents      = trialData.GaitEvents;

    RLMX = filtered_marker.RLM.x;
    LLMX = filtered_marker.LLM.x;
    
    RHEX = filtered_marker.RHE.x;
    LHEX = filtered_marker.LHE.x;
    
    % --- NaN check + pchip interpolation ---
    sigList  = {'RLMX','LLMX','RHEX','LHEX'};
    dataList = { RLMX , LLMX , RHEX , LHEX };
    
    for k = 1:numel(dataList)
        x = dataList{k};
        n = numel(x);
        idx = (1:n)';  
        nNaN = sum(isnan(x));
        if nNaN > 0
            good = ~isnan(x);
            if nnz(good) >= 2
                % fill leading/trailing NaNs with nearest valid sample
                firstGood = find(good,1,'first');
                lastGood  = find(good,1,'last');
                if firstGood > 1, x(1:firstGood-1)   = x(firstGood); end
                if lastGood  < n, x(lastGood+1:end)  = x(lastGood);  end
    
                % fill internal NaNs with pchip
                good = ~isnan(x);
                x(~good) = interp1(idx(good), x(good), idx(~good), 'pchip');
            else
                warning('%s: too few valid points (%d/%d). Leaving NaNs.', ...
                    sigList{k}, nnz(good), n);
            end
    
            fprintf('[NaN interp] %s: filled %d NaNs (of %d)\n', sigList{k}, nNaN, n);
        end
    
        dataList{k} = x;
    end    
    % put back
    RLMX = dataList{1};
    LLMX = dataList{2};
    RHEX = dataList{3};
    LHEX = dataList{4};

    % ======================== BASIC GAIT PARAMETERS ========================
    % STEP TIME (s) from combined heel strikes
    step_time = diff(Heel_Strike) * dt_kin;

    % STRIDE TIME (s) from leg-specific HS
    stride_time_R = diff(RHS) * dt_kin;
    stride_time_L = diff(LHS) * dt_kin;

    % CADENCE (steps/min) from step_time
    cadence = 60 ./ step_time;

    % STEP & STRIDE LENGTH
    % Heel_XRL: AP distance between heels at HS; markers assumed in mm
    Heel_XRL    = RHEX - LHEX;
    step_length = abs(Heel_XRL(Heel_Strike)) / 10;  % -> cm (explicit)
    % Stride length as sum of two consecutive step lengths (cm)
    stride_length = step_length(1:end-1) + step_length(2:end);

    % Map RHS/LHS to positions in Heel_Strike to get side-specific step lengths
    [idxR_in_all, locR] = ismember(RHS, Heel_Strike);
    [idxL_in_all, locL] = ismember(LHS, Heel_Strike);
    if any(~idxR_in_all) || any(~idxL_in_all)
        warning('Some RHS/LHS not found in Heel_Strike for %s. Check events.', fullName);
        locR = locR(idxR_in_all);
        locL = locL(idxL_in_all);
    end
    locR = sort(locR);
    locL = sort(locL);

    step_length_R = step_length(locR);
    step_length_L = step_length(locL);

    % STRIDE LENGTH per side
    nEvents = numel(Heel_Strike);
    stride_length_R = [];
    for k = 1:numel(locR)
        thisIdx = locR(k);
        if thisIdx+1 <= nEvents
            stride_length_R(end+1,1) = step_length(thisIdx) + step_length(thisIdx+1);
        else
            stride_length_R(end+1,1) = NaN;
        end
    end
    stride_length_L = [];
    for k = 1:numel(locL)
        thisIdx = locL(k);
        if thisIdx+1 <= nEvents
            stride_length_L(end+1,1) = step_length(thisIdx) + step_length(thisIdx+1);
        else
            stride_length_L(end+1,1) = NaN;
        end
    end
    stride_length_R = stride_length_R(~isnan(stride_length_R));
    stride_length_L = stride_length_L(~isnan(stride_length_L));

    % ======================== GAIT PHASES ================================
    stance_phase_right = zeros(1,length(RHS));
    stance_phase_left  = zeros(1,length(LHS));
    swing_phase_right  = zeros(1,length(RHS));
    swing_phase_left   = zeros(1,length(LHS));

    stance_idx_r = 1; swing_idx_r = 1;
    stance_idx_l = 1; swing_idx_l = 1;

    % Right foot
    for iR = 1:length(RHS)-1
        TO_idx = find(RTO > RHS(iR), 1, 'first');
        if ~isempty(TO_idx)
            stance_phase_right(stance_idx_r) = RTO(TO_idx) - RHS(iR);
            stance_idx_r = stance_idx_r + 1;

            HS_idx = find(RHS > RTO(TO_idx), 1, 'first');
            if ~isempty(HS_idx)
                swing_phase_right(swing_idx_r) = RHS(HS_idx) - RTO(TO_idx);
                swing_idx_r = swing_idx_r + 1;
            end
        end
    end

    % Left foot
    for iL = 1:length(LHS)-1
        TO_idx = find(LTO > LHS(iL), 1, 'first');
        if ~isempty(TO_idx)
            stance_phase_left(stance_idx_l) = LTO(TO_idx) - LHS(iL);
            stance_idx_l = stance_idx_l + 1;

            HS_idx = find(LHS > LTO(TO_idx), 1, 'first');
            if ~isempty(HS_idx)
                swing_phase_left(swing_idx_l) = LHS(HS_idx) - LTO(TO_idx);
                swing_idx_l = swing_idx_l + 1;
            end
        end
    end

    % Remove zeros
    stance_phase_right(stance_phase_right==0) = [];
    stance_phase_left(stance_phase_left==0)   = [];
    swing_phase_right(swing_phase_right==0)   = [];
    swing_phase_left(swing_phase_left==0)     = [];

    % Double support
    initial_double_support = [];
    for iR = 1:length(RHS)
        idx = find(LTO > RHS(iR), 1, 'first');
        if ~isempty(idx)
            dur = LTO(idx) - RHS(iR);
            if dur > 0
                initial_double_support(end+1) = dur;
            end
        end
    end

    terminal_double_support = [];
    for iL = 1:length(LHS)
        idx = find(RTO > LHS(iL), 1, 'first');
        if ~isempty(idx)
            dur = RTO(idx) - LHS(iL);
            if dur > 0
                terminal_double_support(end+1) = dur;
            end
        end
    end

    nDS = min(length(initial_double_support), length(terminal_double_support));
    double_stance = initial_double_support(1:nDS) + terminal_double_support(1:nDS);

    % Convert all to seconds
    stance_phase_right = stance_phase_right * dt_kin;
    stance_phase_left  = stance_phase_left  * dt_kin;
    swing_phase_right  = swing_phase_right * dt_kin;
    swing_phase_left   = swing_phase_left  * dt_kin;
    double_stance      = double_stance     * dt_kin;

    % Percentages RIGHT (0-100 % of stride)
    nR = min([numel(stance_phase_right), numel(swing_phase_right), ...
              numel(double_stance), numel(stride_time_R)]);
    perc_stance_R = 100 * stance_phase_right(1:nR) ./ stride_time_R(1:nR);
    perc_swing_R  = 100 * swing_phase_right(1:nR) ./ stride_time_R(1:nR);
    perc_double_R = 100 * double_stance(1:nR)      ./ stride_time_R(1:nR);

    perc_stance_R(perc_stance_R > 120) = NaN;
    perc_swing_R(perc_swing_R > 120)   = NaN;
    perc_double_R(perc_double_R > 120) = NaN;

    % Percentages LEFT (0-100 % of stride)
    nL = min([numel(stance_phase_left), numel(swing_phase_left), ...
              numel(double_stance), numel(stride_time_L)]);
    perc_stance_L = 100 * stance_phase_left(1:nL) ./ stride_time_L(1:nL);
    perc_swing_L  = 100 * swing_phase_left(1:nL)  ./ stride_time_L(1:nL);
    perc_double_L = 100 * double_stance(1:nL)     ./ stride_time_L(1:nL);

    perc_stance_L(perc_stance_L > 120) = NaN;
    perc_swing_L(perc_swing_L > 120)   = NaN;
    perc_double_L(perc_double_L > 120) = NaN;

    % ======================== CALCULATE TREADMILL SPEED ======================== 
    numSteps        = length(Heel_Strike);
    numVelocities   = numSteps - 1;
    
    dX  = diff(RHEX(Heel_Strike));                   % Δx between strikes
    dT  = diff(Heel_Strike);                          % Δt (in samples)
    speed_treadmill = (dX./dT);               % raw speeds
    speed_treadmill = speed_treadmill(speed_treadmill < 0);
    speed_treadmill = (speed_treadmill)*-0.36;
    
    % --- MAD-based outlier detection and correction ---
    med = median(speed_treadmill);
    mad_val = mad(speed_treadmill, 1);
    z = 0.6745 * (speed_treadmill - med) / mad_val;
    outliers = abs(z) > 2.5; % Threshold for MAD-based detection
    speed_treadmill(outliers) = interp1(find(~outliers), speed_treadmill(~outliers), find(outliers), 'linear', 'extrap');

    % --- Identify constant-speed stride indices ---
    window_size = 3; % 5
    std_thresh = 0.08; %0.08
    stride_idx = find(movstd(speed_treadmill, window_size) < std_thresh);
    
    % Group consecutive indices into blocks
    d = [Inf; diff(stride_idx)];
    g = find(d > 1);
    if isempty(g)
        g = [1; numel(stride_idx)+1];
    else
        g = [1; g; numel(stride_idx)+1];
    end
    
    % Create blocks of consecutive indices
    constant_blocks = arrayfun(@(i) stride_idx(g(i):g(i+1)-1), 1:(numel(g)-1), 'UniformOutput', false);
    constant_blocks = constant_blocks(cellfun(@numel, constant_blocks) >= 3);
    
    % === Check for short constant blocks (length < 8) ===
    short_blocks = constant_blocks(cellfun(@numel, constant_blocks) < 8);
    if ~isempty(short_blocks)
        disp('Some constant speed blocks have fewer than 8 points. Launching interactive correction GUI...');
    
        figName  = sprintf('Manual Speed Correction – %s', fullName);
        ttlName  = sprintf('%s | Click points to edit speed. Close figure or press Finish when done.', fullName);
    
        f = figure('Name', figName, 'NumberTitle', 'off');
    
        t = (1:length(speed_treadmill))';
        hPlot = plot(t, speed_treadmill, 'b.-', 'MarkerSize', 10); hold on
        hOutliers = plot(t(outliers), speed_treadmill(outliers), 'ro', 'MarkerSize', 10);
    
        xlabel('Sample Index');
        ylabel('Speed (m/s)');
        title(ttlName, 'Interpreter','none');
    
        % Add a button to finish editing
        btn = uicontrol('Style', 'pushbutton', 'String', 'Finish Editing', ...
                        'Position', [20 20 100 30], ...
                        'Callback', @(src, event) uiresume(f));
    
        uiwait(f);  % Pause until user presses button or closes figure
    
        % Interactive loop for editing points
        while ishandle(f)
            % Check if figure still exists before calling ginput
            if ~ishandle(f)
                break; % Figure was closed, exit gracefully
            end
    
            try
                [x_click, ~, button] = ginput(1);
            catch
                % ginput interrupted by figure close
                break;
            end
    
            if isempty(button) || button == 3  % Right-click or no click to exit
                break;
            end
    
            x_click = round(x_click);
            if x_click < 1 || x_click > length(speed_treadmill)
                disp('Clicked outside data range, try again');
                continue;
            end
    
            % Prompt user to enter new speed for selected point
            prompt = sprintf('Enter new speed value for sample %d (current %.3f):', ...
                x_click, speed_treadmill(x_click));
            answer = inputdlg(prompt, 'Edit Speed', [1 50]);
    
            if isempty(answer)
                disp('Edit cancelled');
                continue;
            end
    
            new_speed = str2double(answer{1});
            if isnan(new_speed)
                disp('Invalid input, please enter a numeric value');
                continue;
            end
    
            % Update data and plot
            speed_treadmill(x_click) = new_speed;
            set(hPlot, 'YData', speed_treadmill);
            drawnow;
        end
    
        if ishandle(f)
            close(f);
        end
    
        % --- Recompute constant-speed indices after manual correction ---
        stride_idx = find(movstd(speed_treadmill, window_size) < std_thresh);
        d = [Inf; diff(stride_idx)];
        g = find(d > 1);
        if isempty(g)
            g = [1; numel(stride_idx)+1];
        else
            g = [1; g; numel(stride_idx)+1];
        end
        constant_blocks = arrayfun(@(i) stride_idx(g(i):g(i+1)-1), 1:(numel(g)-1), 'UniformOutput', false);
        constant_blocks = constant_blocks(cellfun(@numel, constant_blocks) >= 3);

        % --- Save final corrected speed plot (robust) ---
        save_dir = 'path';
        
        % Fallbacks if subject/trial not in scope
        if ~exist('subject','var') || isempty(subject), subject = 'UnknownSubject'; end
        if ~exist('trial','var')   || isempty(trial),   trial   = 'UnknownTrial';   end
        
        % Sanitize for filenames
        subject_clean = regexprep(string(subject), '[^\w-]', '_');
        trial_clean   = regexprep(string(trial)  , '[^\w-]', '_');
        timestamp     = datestr(now, 'yyyymmdd_HHMMSS');
        base_name     = sprintf('SpeedCorrection_%s_%s_%s', subject_clean, trial_clean, timestamp);
        
        if ~exist(save_dir,'dir'); mkdir(save_dir); end
        png_path = fullfile(save_dir, base_name + ".png");
        fig_path = fullfile(save_dir, base_name + ".fig");
        
        hFig = figure('Name','Final Corrected Treadmill Speed','Color','w','Visible','on');
        t = (1:length(speed_treadmill))';
        plot(t, speed_treadmill, 'b.-', 'MarkerSize', 8, 'DisplayName','Corrected Speed'); hold on
        
        if exist('constant_stride_speed','var') && ~isempty(constant_stride_speed)
            idx = constant_stride_speed(:);
            idx = idx(idx >= 1 & idx <= numel(speed_treadmill));
            if ~isempty(idx)
                plot(idx, speed_treadmill(idx), 'go', 'DisplayName','Constant-Speed Strides');
            end
        end
        
        xlabel('Stride Index');
        ylabel('Speed (m/s)');
        title(sprintf('Final Corrected Treadmill Speed – %s | %s', subject, trial), 'Interpreter', 'none');
        legend('Location','best'); grid on; box on; drawnow;
    end
    
    % --- Concatenate final constant speed indices ---
    constant_stride_speed = unique(vertcat(constant_blocks{:}));
    
    v = constant_stride_speed(:);
    breaks = find(diff(v) ~= 1);
    starts = [1; breaks + 1];
    ends   = [breaks; numel(v)];
    seg_constant_speed    = ends - starts + 1;
    min_seg_constant_speed = min(seg_constant_speed);
    change_speed_stride    = v(starts);
    
    % Use GUI-corrected indices
    subjTrialName = fullName;
    
    % Call GUI with name
    corrected_indices = speed_change_gui(speed_treadmill, change_speed_stride, subjTrialName);
    change_speed_stride = corrected_indices(:)';
    change_speed_sec    = Heel_Strike(change_speed_stride) * dt_kin;

    parameters_struct.speed_treadmill        = speed_treadmill;
    parameters_struct.constant_stride_speed  = constant_stride_speed;
    parameters_struct.seg_constant_speed     = seg_constant_speed;
    parameters_struct.min_seg_constant_speed = min_seg_constant_speed;
    parameters_struct.change_speed_stride    = change_speed_stride;
    parameters_struct.change_speed_sec       = change_speed_sec;

    % ======================== DISCRETE & RHYTHMIC METRICS =================
    % ===================== SAL + LDJ on Ank_XRL segmented by zerocrossing =====================
    if any(isnan(Ank_XRL))
        Ank_XRL = Ank_XRL(:);
        t_all   = (0:numel(Ank_XRL)-1)' * dt_kin;
        good    = isfinite(Ank_XRL);
    
        if nnz(good) >= 2
            Ank_XRL(~good) = interp1(t_all(good), Ank_XRL(good), t_all(~good), 'pchip');
            % If any NaNs remain (typically at edges), fill with nearest valid value
            Ank_XRL = fillmissing(Ank_XRL, 'nearest');
        else
            Ank_XRL(:) = NaN;
        end
    
        Ank_XRL = Ank_XRL(:)';  % back to row, to match your later code style
    end

    Ank_XRL = Ank_XRL - mean(Ank_XRL, 'omitnan');
    
    [zeroCrossings, ~] = findZeroCrossings(Ank_XRL);
    
    SALs  = [];
    LDJs  = [];   
    for ii = 1:2:length(zeroCrossings)-2
        idx_start = zeroCrossings(ii);
        idx_end   = zeroCrossings(ii+2);
        if idx_end <= idx_start
            continue;
        end
    
        ankleMovement = Ank_XRL(idx_start:idx_end);
    
        Tseg = (numel(ankleMovement)-1) * dt_kin;
        if Tseg <= 0 || numel(ankleMovement) < 5
            continue;
        end
    
        % --- kinematics: vel, acc, jerk ---
        vel  = gradient(ankleMovement) / dt_kin;
        acc  = gradient(vel) / dt_kin;
        jerk = gradient(acc) / dt_kin;
    
        % --- SAL (your existing) ---
        try
            S = SpectralArcLength(abs(vel), dt_kin);
        catch
            S = NaN;
        end
        SALs(end+1,1) = S;
    
        % --- LDJ peak-speed LDLJ on primitive ---
        % Same formula as your gait-cycle LDLJ, but using Tseg.
        J_peak = trapz(jerk.^2) * dt_kin;     % ∫ jerk^2 dt
        v_peak = max(abs(vel));              % peak speed
    
        if J_peak > 0 && v_peak > 0
            DJ_peak   = (Tseg^3 / (v_peak^2)) * J_peak;
            LDLJ_peak = -log(DJ_peak);
        else
            LDLJ_peak = NaN;
        end
        LDJs(end+1,1) = LDLJ_peak;
    end

    % ==================== SEGMENTATION BY SPEED CHANGES ===================
    % Extract subject & trial from filename
    underscoreSplit = split(fullName, '_');
    subject = underscoreSplit{1};
    trial   = strjoin(underscoreSplit(2:end), '_');
    
    % --- SAL from primitives ---    
    try
        [SAL_matrix, SAL_array] = createSegmentedMatrix(SALs, change_speed_stride, true, trial, constant_stride_speed);
        parameters_struct.SAL_matrix              = SAL_matrix;
        parameters_struct.array_data.SAL_array    = SAL_array;
    catch ME
        warning('SALs not segmented for %s: %s', fullName, ME.message);
    end
    
    % --- LDJ from primitives ---
    try
        [LDJ_matrix, LDJ_array] = createSegmentedMatrix(LDJs, change_speed_stride, true, trial, constant_stride_speed);
        parameters_struct.LDJ_matrix              = LDJ_matrix;
        parameters_struct.array_data.LDJ_array    = LDJ_array;
    catch ME
        warning('LDJs not segmented for %s: %s', fullName, ME.message);
    end

    % --- stride_length (global) ---
    sl_change_idx   = change_speed_stride * 2;                                   % stride -> step
    sl_constant_idx = sort(unique([constant_stride_speed*2 - 1; constant_stride_speed*2]));
    [stride_length_matrix, stride_length_array] = createSegmentedMatrix( ...
        stride_length, sl_change_idx, true, trial, sl_constant_idx);
    parameters_struct.stride_length_matrix              = stride_length_matrix;
    parameters_struct.array_data.stride_length_array    = stride_length_array;
    
    % --- STRIDE-BASED fields ---
    fields_to_segm = { ...
        'stride_time_R','stride_time_L', ...
        'perc_stance_R','perc_swing_R','perc_double_R', ...
        'perc_stance_L','perc_swing_L','perc_double_L'};
    
    strideBased = struct( ...
        'stride_time_R', stride_time_R, ...
        'stride_time_L', stride_time_L, ...
        'perc_stance_R', perc_stance_R, ...
        'perc_swing_R',  perc_swing_R,  ...
        'perc_double_R', perc_double_R, ...
        'perc_stance_L', perc_stance_L, ...
        'perc_swing_L',  perc_swing_L,  ...
        'perc_double_L', perc_double_L);
    
    for k = 1:numel(fields_to_segm)
        fld = fields_to_segm{k};
        values = strideBased.(fld);
        try
            [seg_matrix, seg_array] = createSegmentedMatrix(values, change_speed_stride, true, trial, constant_stride_speed);
            parameters_struct.([fld '_matrix'])           = seg_matrix;
            parameters_struct.array_data.([fld '_array']) = seg_array;
        catch ME
            warning('Could not segment %s for %s: %s', fld, fullName, ME.message);
        end
    end
    
    % --- STEP-BASED fields ---
    fields_step_to_segm = {'cadence','step_length','step_time'};
    
    correct_step_indices = change_speed_stride * 2;            % stride->step
    constant_step_idx    = sort(unique([constant_stride_speed*2 - 1; ...
                                        constant_stride_speed*2]));
    
    stepBased = struct( ...
        'cadence',     cadence, ...
        'step_length', step_length, ...
        'step_time',   step_time);
    
    for k = 1:numel(fields_step_to_segm)
        fld = fields_step_to_segm{k};
        values = stepBased.(fld);
        try
            [seg_matrix, seg_array] = createSegmentedMatrix(values, correct_step_indices, true, trial, constant_step_idx);
            parameters_struct.([fld '_matrix'])           = seg_matrix;
            parameters_struct.array_data.([fld '_array']) = seg_array;
        catch ME
            warning('Could not segment %s for %s: %s', fld, fullName, ME.message);
        end
    end
    
    % ==================== SEGMENTED PLOTS (QC) ===========================
    segData      = parameters_struct.array_data;
    array_fields = fieldnames(segData);
    array_fields = array_fields(endsWith(array_fields,'_array'));
    
    lineColor    = [0 0.4470 0.7410];
    segLineColor = [0.85 0.33 0.10];
    lw_main      = 2;
    lw_seg       = 1.5;
    
    saveDirCheck = fullfile('path', 'SegPlots_new');
    if ~exist(saveDirCheck,'dir'), mkdir(saveDirCheck); end
    
    % ---------- (A) Standard QC plots for *_array fields ----------
    % NOTE:
    %  - *_array are "concatenated across segments", so segment boundaries MUST be drawn
    %    using cumulative segment lengths derived from the corresponding *_matrix.
    %  - We skip MSJR_new_array to avoid confusion (dedicated primitive-domain plot below).
    for ii = 1:numel(array_fields)
        fld = array_fields{ii};
    
        % Skip MSJR_new_array to avoid duplicate/confusing plot
        if strcmpi(fld, 'MSJR_new_array')
            continue;
        end
    
        array_data = segData.(fld);
        if isempty(array_data)
            continue;
        end
        if iscell(array_data)
            array_data = horzcat(array_data{:});
        end
        array_data = array_data(:)'; % row
    
        f = figure('Name',[fld ' - ' fullName], 'Color','w', ...
                   'Units','normalized', 'Position',[0.3 0.4 0.5 0.35]);
    
        plot(1:numel(array_data), array_data, '-o', ...
             'Color', lineColor, 'LineWidth', lw_main, ...
             'MarkerFaceColor', lineColor, 'MarkerSize',4);
        hold on;
    
        % Find corresponding matrix to draw segment boundaries
        base_name  = erase(fld,'_array');
        matrix_fld = [base_name '_matrix'];
    
        seg_matrix = [];
        if isfield(parameters_struct, matrix_fld)
            seg_matrix = parameters_struct.(matrix_fld);
        elseif isfield(segData, matrix_fld)
            seg_matrix = segData.(matrix_fld);
        end
    
        % Draw boundaries ONLY if we have the matching matrix
        if ~isempty(seg_matrix)
            if ~iscell(seg_matrix)
                seg_matrix = num2cell(seg_matrix, 2);
            end
    
            idx_cum = 0;
            for s = 1:numel(seg_matrix)
                seg_vals = seg_matrix{s};
                seg_len  = sum(isfinite(seg_vals));   % use finite, not ~isnan (robust)
                if seg_len > 0
                    start_idx = idx_cum + 1;
                    end_idx   = idx_cum + seg_len;
    
                    xline(start_idx, '--', 'Color', segLineColor, 'LineWidth', lw_seg);
                    xline(end_idx,   '--', 'Color', segLineColor, 'LineWidth', lw_seg);
    
                    idx_cum = end_idx;
                end
            end
        end
    
        grid on; box on;
        title(strrep([fld ' - ' fullName],'_','\_'), 'FontSize',14, 'FontWeight','bold');
        xlabel('Index (concatenated domain)', 'FontSize',12);
        ylabel('Value', 'FontSize',12);
        set(gca,'FontSize',11,'LineWidth',1);
    
        savefig(f, fullfile(saveDirCheck, sprintf('%s_%s_%s.fig', subject, trial, fld)));
        close(f);
    end
    
    % ======================= SAVE PER-TRIAL RESULTS ======================
    save(fullfile(filePath_save, [fullName '.mat']), 'parameters_struct');
    fprintf('%s saved.\n', fullName);

end

%% ====================== FUNCTIONS ======================
function [zeroCrossingIndices, segments] = findZeroCrossings(signal)
    % Find zero-crossings and return segments between zero-crossings
    zeroCrossingIndices = [];
    segments = {};

    for i = 1:length(signal)-1
        if (signal(i) > 0 && signal(i+1) < 0) || (signal(i) < 0 && signal(i+1) > 0)
            idx = i + (0 - signal(i)) / (signal(i+1) - signal(i));
            zeroCrossingIndices(end+1) = idx; 
        end
    end

    zeroCrossingIndices = round(zeroCrossingIndices);

    if ~isempty(zeroCrossingIndices)
        for j = 1:length(zeroCrossingIndices)-2
            segmentStart = zeroCrossingIndices(j);
            segmentEnd   = zeroCrossingIndices(j+2);
            if segmentEnd > segmentStart
                segments{end+1} = signal(segmentStart:segmentEnd); 
            end
        end
    end
end

function [segmented_matrix, concatenated_array] = createSegmentedMatrix(values, change_speed, do_winsorize, trial, constant_idx)
% Segment "values" based on change_speed indices.
% change_speed are START indices of segments (excluding 1 ideally).
% constant_idx: optional indices (same domain as values) to keep.
% do_winsorize: winsorize within each segment (10-90th).
%
% Returns:
%   segmented_matrix: [nSegments x maxLen] NaN-padded
%   concatenated_array: concatenated non-NaN values (row)

    if nargin < 3 || isempty(do_winsorize), do_winsorize = false; end
    if nargin < 4 || isempty(trial), trial = ''; end
    if nargin < 5, constant_idx = []; end

    values = values(:)';               % row
    N = numel(values);

    % ---- sanitize change_speed ----
    if isempty(change_speed)
        segments_idx = [1, N+1];
    else
        change_speed = unique(round(change_speed(:)'));
        change_speed(~isfinite(change_speed)) = [];
        change_speed(change_speed <= 1) = [];      % remove 1/0/neg
        change_speed(change_speed >= N) = [];      % cannot start at/after last sample

        segments_idx = unique([1, change_speed, N+1]);
        segments_idx = sort(segments_idx);
        if numel(segments_idx) < 2
            segments_idx = [1, N+1];
        end
    end

    num_segments = numel(segments_idx) - 1;

    % ---- sanitize constant_idx ----
    if ~isempty(constant_idx)
        constant_idx = unique(round(constant_idx(:)'));
        constant_idx = constant_idx(isfinite(constant_idx));
        constant_idx = constant_idx(constant_idx >= 1 & constant_idx <= N);
    end

    max_len = 0;
    tmp_segs = cell(1, num_segments);

    for s = 1:num_segments
        g_start = segments_idx(s);
        g_end   = segments_idx(s+1) - 1;
        if g_end < g_start
            tmp_segs{s} = [];
            continue;
        end

        seg_vals = values(g_start:g_end);

        % keep only constant indices if provided (must match value domain!)
        if ~isempty(constant_idx)
            idx_range = g_start:g_end;
            mask = ismember(idx_range, constant_idx);
            seg_vals = seg_vals(mask);
        end

        % winsorize (ignore NaN)
        if do_winsorize && ~isempty(seg_vals)
            good = isfinite(seg_vals);
            if nnz(good) >= 3
                p10 = prctile(seg_vals(good), 10);
                p90 = prctile(seg_vals(good), 90);
                seg_vals(good) = min(max(seg_vals(good), p10), p90);
            end
        end

        tmp_segs{s} = seg_vals;
        max_len = max(max_len, numel(seg_vals));
    end

    segmented_matrix = NaN(num_segments, max_len);
    for s = 1:num_segments
        seg_vals = tmp_segs{s};
        if ~isempty(seg_vals)
            segmented_matrix(s, 1:numel(seg_vals)) = seg_vals;
        end
    end

    % Remove first segment if RAMPDOWN (only if that is truly what you want)
    if contains(upper(trial), 'RAMPDOWN') && size(segmented_matrix,1) > 1
        segmented_matrix(1,:) = [];
    end

    % Concatenate non-NaNs
    concatenated_array = segmented_matrix';
    concatenated_array = concatenated_array(isfinite(concatenated_array))';
end


function corrected_indices = speed_change_gui(speed_signal, initial_indices,fullName)
    % GUI to inspect and correct treadmill speed change points

    f = figure('Name', sprintf('Edit Speed Change Points – %s', fullName), ...
               'NumberTitle', 'off', ...
               'CloseRequestFcn', @onClose, ...
               'Units', 'normalized', ...
               'Position', [0.1, 0.1, 0.8, 0.6]);
    hold on;
    plot(speed_signal, 'b-', 'DisplayName', 'Speed', 'LineWidth', 1.5);
    hMarkers = scatter(initial_indices, speed_signal(initial_indices), ...
                       80, 'g', 'filled', 'DisplayName', 'Change Points');
    title('Left click = Add | Right click = Delete | Close to Save');
    xlabel('Index'); ylabel('Speed'); legend(); grid on;

    guiData.hMarkers = hMarkers;
    guiData.signal   = speed_signal;
    guidata(f, guiData);

    set(f, 'WindowButtonDownFcn', @mouse_click);
    uiwait(f);

    if evalin('base','exist(''CorrectedSpeedChangeIndices'',''var'')')
        corrected_indices = evalin('base','CorrectedSpeedChangeIndices');
        evalin('base','clear CorrectedSpeedChangeIndices');
    else
        corrected_indices = initial_indices;
    end

    function mouse_click(~, ~)
        d = guidata(f);
        clickType = get(f, 'SelectionType');  % 'normal'=left, 'alt'=right
        coords = get(gca, 'CurrentPoint');
        x = round(coords(1,1));
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
        assignin('base','CorrectedSpeedChangeIndices', corrected);
        uiresume(src);
        delete(src);
    end
end

function [zc_t, zc_slope] = localZeroCrossings_timeSlope(xSm, vSm, t)
% Returns crossing time (seconds) + slope sign at the crossing.
% Slope sign comes from spline velocity at crossing (more stable than sign-change heuristic).

    xSm = xSm(:);
    vSm = vSm(:);
    t   = t(:);

    zc_t     = [];
    zc_slope = [];

    N = numel(xSm);

    % small epsilon to avoid "exact zero" sticking
    eps0 = 0;
    if any(isfinite(xSm))
        eps0 = 1e-12 * max(1, max(abs(xSm(isfinite(xSm)))));
    end

    for n = 1:(N-1)
        x1 = xSm(n);
        x2 = xSm(n+1);

        if ~isfinite(x1) || ~isfinite(x2)
            continue;
        end

        % treat exact zeros as tiny values so sign is defined
        if x1 == 0, x1 = eps0; end
        if x2 == 0, x2 = -eps0; end

        if (x1 < 0 && x2 > 0) || (x1 > 0 && x2 < 0)
            frac = -x1 / (x2 - x1);
            frac = min(max(frac, 0), 1);

            tz = t(n) + frac*(t(n+1)-t(n));

            % velocity at crossing -> slope sign
            vz = interp1(t, vSm, tz, 'linear', 'extrap');
            s  = sign(vz);

            % fallback if vz ~ 0
            if s == 0
                s = sign(x2 - x1);
            end

            zc_t(end+1,1)     = tz; 
            zc_slope(end+1,1) = s;  
        end
    end

    if ~isempty(zc_t)
        % remove duplicates, keep order
        [zc_t, ia] = unique(zc_t, 'stable');
        zc_slope = zc_slope(ia);
    end
end


% ---------------- local helpers (self-contained) ----------------
function [x,z] = getMarkerXZ(filtered_marker, name)
    x = []; z = [];
    if isfield(filtered_marker, name) && ...
       isfield(filtered_marker.(name),'x') && isfield(filtered_marker.(name),'z')
        x = filtered_marker.(name).x(:);
        z = filtered_marker.(name).z(:);
    end
end

function [x,z,used] = pickMarkerXZ(filtered_marker, primary, fallback, nanFracThresh)
    [x1,z1] = getMarkerXZ(filtered_marker, primary);
    if ~isempty(x1) && nanFraction_local(x1) <= nanFracThresh && nanFraction_local(z1) <= nanFracThresh
        x = x1; z = z1; used = char(primary); return;
    end
    [x2,z2] = getMarkerXZ(filtered_marker, fallback);
    if ~isempty(x2)
        x = x2; z = z2; used = char(fallback); return;
    end
    x = NaN(size(x1)); z = NaN(size(z1)); used = char(primary) + "->MISSING";
end

function [x,z,used] = pickFromListXZ(filtered_marker, nameList)
    used = "MISSING";
    x = []; z = [];
    for k = 1:numel(nameList)
        nm = char(nameList(k));
        [xt,zt] = getMarkerXZ(filtered_marker, nm);
        if ~isempty(xt)
            x = xt; z = zt; used = nm; return;
        end
    end
    x = NaN; z = NaN;
end

function f = nanFraction_local(x)
    x = x(:);
    if isempty(x), f = 1; return; end
    f = sum(~isfinite(x))/numel(x);
end

function x = padToN_local(x,N)
    x = x(:);
    if isempty(x), x = NaN(N,1); return; end
    if numel(x) < N, x(end+1:N,1) = NaN;
    elseif numel(x) > N, x = x(1:N);
    end
end

function x = fillNaNs_pchipNearest_local(x)
    x = x(:); n = numel(x);
    bad = ~isfinite(x);
    if ~any(bad), return; end
    good = ~bad;
    if nnz(good) < 2, return; end

    idx = (1:n)';
    fg = find(good,1,'first');
    lg = find(good,1,'last');
    if fg > 1, x(1:fg-1) = x(fg); end
    if lg < n, x(lg+1:end) = x(lg); end

    good = isfinite(x);
    x(~good) = interp1(idx(good), x(good), idx(~good), 'pchip');
end
