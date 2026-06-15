%==========================================================================
% EMG_RAMP_synergies.m
% 
% GROUP-LEVEL EMG SYNERGY ANALYSIS PIPELINE
% This script processes segmented EMG data across multiple subjects to extract
% and analyze group-level muscle synergies using Non-negative Matrix Factorization (NNMF).
% 
% Workflow:
%  1. Load subject-level .mat files containing segmented EMG (mean and stride-based).
%  2. Harmonize muscle channel names across all subjects to a common reference list.
%  3. Initialize a group-level data structure (`GroupData`) for each trial type (RAMPUP, RAMPDOWN).
%  4. For each segment within each trial type:
%     - Aggregate mean EMG matrices across subjects.
%     - Compute group-level mean and std EMG activity.
%     - Perform a synergy number sweep (1 → `max_synergies`) to compute VAF curves.
%     - Select the minimum number of synergies that exceeds 90% VAF (or fallback = 5).
%     - Extract final NNMF solutions (W = muscle weights, H = activations).
%     - Store raw and aligned synergies (before reordering).
%     - Compute reconstruction VAF per segment.
%  5. Align synergies across segments:
%     - Select a lead segment (first for RAMPDOWN, last for RAMPUP).
%     - Chronologically reorder lead synergies by H peak.
%     - Pad all segments to the same synergy count.
%     - Align synergies sequentially across segments using Hungarian matching
%       (cost = -similarity, combining spatial W and temporal H correlations).
%  6. Compute circular activation metrics on aligned H:
%     - Center of Activation (CoA, phase of peak activation).
%     - Full Width at Half Maximum (FWHM, activation duration).
%  7. Save results (`GroupData`) to file for further analysis.
%  8. Generate plots:
%     - VAF curves with chosen k per segment.
%     - Synergy weights (W) and activations (H) across segments.
%     - Circular metrics (CoA, FWHM) per synergy across segments.
% 
% Requirements:
%  - MATLAB with Statistics Toolbox
%  - munkres.m (Hungarian assignment algorithm implementation) on path
% 
% Outputs:
%  - `EMGSynergiesDATA.mat` (processed synergies and metrics)
%  - Figures (.fig) for VAF, W, H, CoA, and FWHM per condition
%
% LAST UPDATED: 9 September 2025 Giulia Panconi
%==========================================================================
clc; clear;
%clearvars -except AllSubjectData

% --- Specify input and output paths -------------------------------------
filepath_load   = 'path';   % Folder containing subject .mat files
filepath_save   = 'path'; % File to save group results
filepath_plot   = 'path';

%% 1) DEFINE PARAMETERS
trial_types             = {'RAMPUP','RAMPDOWN'};
max_synergies           = 10;
num_iter                = 100;
opts                    = statset('MaxIter',1000,'Display','off');
alpha                   = 0.8;  % α = 0 → only temporal activation similarity; α = 1 → only muscle weight similarity; 
ref_order = {'TAr','SOLr','PERr','GMr','GLr','RFr','VLr','VMr','BFr','SEMr', ...
             'SARTr','GMEDr','TFLr','GLMr','ESr','TAl','SOLl','PERl','GMl', ...
             'GLl','RFl','VLl','VMl','BFl','SEMl','SARTl','GMEDl','TFLl','GLMl','ESl'};

%% ====== CHOICE OF SYNERGY-SELECTION METHOD ======
% options: '90VAF', 'dVAF', 'KP', 'KP_or_dVAF'
method_mean   = '90VAF';

vaf_thr_mean   = 0.90;   % Torres-Oviedo 2006
dVAF_thr       = 0.05;   % 5% ΔVAF Clark 2010

%% 2) LOAD DATA
% all_files = dir(fullfile(filepath_load,'*.mat'));
% valid_files = all_files(arrayfun(@(f) f.name(1)~='.',all_files));
% AllSubjectData = struct();
% for f = valid_files'
%     data = load(fullfile(filepath_load,f.name));
%     fname = matlab.lang.makeValidName(f.name(1:end-4));
%     AllSubjectData.(fname) = data;
% end
% subjects = fieldnames(AllSubjectData);
% save(fullfile(filepath_save,'AllSubjectEMGData.mat'),'AllSubjectData','subjects', '-v7.3');

load('path');

%% 3) UNIFY MUSCLE LIST
tmp = cellfun(@(s) AllSubjectData.(s).muscle_names(:),subjects,'UniformOutput',false);
all_muscles = unique(vertcat(tmp{:}),'stable');
M = numel(all_muscles);

%% 4) INIT OUTPUT
for t = 1:numel(trial_types)
    tp = trial_types{t};
    GroupData.(tp) = struct( ...
        'segments',[], ...
        'mean_seg',{{}}, 'std_seg',{{}}, ...
        'VAFcurve',{{}}, 'num_synergies',[], ...
        'W',{{}}, 'H',{{}}, 'Word',{{}}, 'Hord',{{}}, 'VAF',[], ...
        'CoA',{{}}, 'FWHM',{{}}, ...
        'VAFcurve_stride',{{}}, ...
        'num_synergies_stride',[], ...
        'W_stride',{{}}, 'H_stride',{{}}, 'VAF_stride',[], ...
        'concat_analysis', struct( ...
            'VAFcurve',    {{}}, ...
            'num_synergies', [], ...
            'W',           {{}}, ...
            'H',           {{}}, ...
            'Word',        {{}}, ...
            'Hord',        {{}}, ...
            'VAF',          [], ...
            'concat_allruns', {{} } ...
        ) ...
    );
end

%% 5) MAIN LOOP
tic
for t = 1:numel(trial_types)
    tp = trial_types{t}; 
    lc = lower(tp);
    mean_list   = {}; 
    seg_ids     = [];
    stride_list = {}; 
    seg_ids_s   = [];
    
    % ---- collect subject matrices ----
    for i = 1:numel(subjects)
        S = AllSubjectData.(subjects{i});
        if ~isfield(S,'trial_name'), continue; end
        if ~startsWith(lower(S.trial_name), lc), continue; end
        mus = S.muscle_names(:);

        % mean_emg_seg
        if isfield(S,'mean_emg_seg')
            for seg = find(~cellfun(@isempty, S.mean_emg_seg))'
                X = S.mean_emg_seg{seg};                % time x subjMus
                Xref = nan(size(X,1), M);               % time x refMus
                for j = 1:M
                    idm = find(strcmp(all_muscles{j}, mus), 1);
                    if ~isempty(idm)
                        Xref(:,j) = X(:,idm);
                    end
                end
                mean_list{end+1,1} = Xref; 
                seg_ids(end+1,1)   = seg;
            end
        end
        % norm_emg_seg (stride-based)
        if isfield(S,'norm_emg_seg')
            for seg = find(~cellfun(@isempty, S.norm_emg_seg))'
                Y = S.norm_emg_seg{seg};     % time x muscles x strides
                Ycat = reshape(permute(Y,[1 3 2]), [], size(Y,2)); % (time*strides) x muscles
                Yref = nan(size(Ycat,1), M);
                for j = 1:M
                    idm = find(strcmp(all_muscles{j}, mus), 1);
                    if ~isempty(idm)
                        Yref(:,j) = Ycat(:,idm);
                    end
                end
                stride_list{end+1,1} = Yref; 
                seg_ids_s(end+1,1)   = seg;
            end
        end
    end

    % ---- segment list ----
    Sg   = unique(seg_ids);
    Sg_s = unique(seg_ids_s);
    if isempty(Sg), continue; end
    GroupData.(tp).segments = Sg;
    maxSeg = max(Sg);

    % ---- store NNMF input for mean-based ----
    GroupData.(tp).nnmf_input_matrix_mean = cell(maxSeg,1);

    % ---- extra fields for mean-based selection criteria ----
    GroupData.(tp).VAFcurve_mean_surrogate = cell(maxSeg,1);
    GroupData.(tp).dVAF_mean               = cell(maxSeg,1);
    GroupData.(tp).slope_real_mean         = cell(maxSeg,1);
    GroupData.(tp).slope_surrogate_mean    = cell(maxSeg,1);
    GroupData.(tp).num_synergies_90_mean   = nan(maxSeg,1);
    GroupData.(tp).num_synergies_dVAF_mean = nan(maxSeg,1);
    GroupData.(tp).num_synergies_KP_mean   = nan(maxSeg,1);

    % ---- pre-allocate stride-based fields ----
    GroupData.(tp).VAFcurve_stride       = cell(maxSeg,1);
    GroupData.(tp).num_synergies_stride  = zeros(maxSeg,1);
    GroupData.(tp).W_stride              = cell(maxSeg,1);
    GroupData.(tp).H_stride              = cell(maxSeg,1);
    GroupData.(tp).VAF_stride            = zeros(maxSeg,1);

    % ==========================================================
    % MEAN-BASED EXTRACTION PER SEGMENT
    % ==========================================================
    for us = Sg'
        idx_us = find(seg_ids == us);   % indices of subjects for this segment
        if isempty(idx_us), continue; end

        % ---- decide common length for this segment ----
        Ls = cellfun(@(C) size(C,1), mean_list(idx_us));
        L0 = mode(Ls);
        if isempty(L0) || L0==0, L0 = 200; end

        % ---- build 3D stack for mean-based ----
        mats = nan(L0, M, numel(idx_us));
        for k = 1:numel(idx_us)
            X = mean_list{idx_us(k)};
            t_old = linspace(0,1,size(X,1));
            t_new = linspace(0,1,L0);
            mats(:,:,k) = interp1(t_old, X, t_new);
        end

        % ---- group mean/std ----
        Gmean = mean(mats,3,'omitnan');  Gmean(isnan(Gmean)) = 0;
        Gstd  = std(mats,0,3,'omitnan'); Gstd(isnan(Gstd))   = 0;

        GroupData.(tp).mean_seg{us} = Gmean;
        GroupData.(tp).std_seg{us}  = Gstd;

        % save NNMF input (mean-based)
        GroupData.(tp).nnmf_input_matrix_mean{us} = Gmean;

        % ---- VAF sweep on mean + surrogate (MEAN: NaN -> 0, use nnmf) ----
        Vc_mean    = nan(1, max_synergies);
        Vsurr_mean = nan(1, max_synergies);
        
        Gtrue = Gmean;
        Gtrue(isnan(Gtrue)) = 0;                 % <-- RESTORE NaN -> 0 for MEAN
        Xtrue = Gtrue';                          % M x time (no NaNs)
        
        % save NNMF input (mean-based) as the actual matrix used by nnmf
        GroupData.(tp).nnmf_input_matrix_mean{us} = Gtrue;
        
        for ksyn = 1:max_synergies
        
            % ===== REAL: best-of-num_iter =====
            bestV = -inf;
            bestW = [];
            bestH = [];
        
            for r = 1:num_iter
                try
                    % nnmf expects nonnegative + finite
                    [Wm,Hm] = nnmf(Xtrue, ksyn, 'algorithm','mult', 'options',opts);
        
                    Ghat = (Wm*Hm)';            % time x M
                    v = vaf_fro(Gtrue, Ghat);
        
                    if ~isnan(v) && v > bestV
                        bestV = v;
                        bestW = Wm;
                        bestH = Hm;
                    end
                catch
                end
            end
            Vc_mean(ksyn) = bestV;
        
            % ===== SURROGATE: best-of-num_iter =====
            % (mean has NO NaNs now, so shuffle all entries)
            Gsh = Gtrue;
            Gsh(:) = Gsh(randperm(numel(Gsh)));
            Xsh = Gsh';
        
            bestVs = -inf;
            for r = 1:num_iter
                try
                    [Ws,Hs] = nnmf(Xsh, ksyn, 'algorithm','mult', 'options',opts);
                    Ghat_s = (Ws*Hs)';          % time x M
                    vs = vaf_fro(Gsh, Ghat_s);
                    if ~isnan(vs) && vs > bestVs
                        bestVs = vs;
                    end
                catch
                end
            end
            Vsurr_mean(ksyn) = bestVs;
        end
        
        GroupData.(tp).VAFcurve{us}                = Vc_mean;
        GroupData.(tp).VAFcurve_mean_surrogate{us} = Vsurr_mean;
        
        % --- selection criteria (mean-based)
        dV_mean   = diff(Vc_mean);
        slope_r_m = dV_mean;
        slope_s_m = diff(Vsurr_mean);
        
        GroupData.(tp).dVAF_mean{us}            = dV_mean;
        GroupData.(tp).slope_real_mean{us}      = slope_r_m;
        GroupData.(tp).slope_surrogate_mean{us} = slope_s_m;
        
        % 90% VAF
        k_90_m = find(Vc_mean >= vaf_thr_mean, 1, 'first');
        if isempty(k_90_m), k_90_m = NaN; end
        GroupData.(tp).num_synergies_90_mean(us) = k_90_m;
        
        % ΔVAF < threshold
        if ~isempty(dV_mean)
            k_dV_m = find(dV_mean < dVAF_thr, 1, 'first');
            if ~isempty(k_dV_m), k_dV_m = k_dV_m + 1; else, k_dV_m = NaN; end
        else
            k_dV_m = NaN;
        end
        GroupData.(tp).num_synergies_dVAF_mean(us) = k_dV_m;
        
             
        % ---- choose kopt for MEAN based on method_mean ----
        switch lower(method_mean)
            case '90vaf'
                kopt = k_90_m;
            case 'dvaf'
                kopt = k_dV_m;
        end
        if isnan(kopt) || kopt < 1
            kopt = max_synergies;
        end
        
        GroupData.(tp).num_synergies(us) = kopt;
        
        % ---- final MEAN factorization: rerun nnmf and keep best-of-num_iter ----
        bestV = -inf; bestW = []; bestH = [];
        for r = 1:num_iter
            try
                [Wm,Hm] = nnmf(Xtrue, kopt, 'algorithm','mult', 'options',opts);
                Ghat = (Wm*Hm)';                       % time x M
                v = vaf_fro(Gtrue, Ghat);
                if ~isnan(v) && v > bestV
                    bestV = v; bestW = Wm; bestH = Hm;
                end
            catch
            end
        end
        
        GroupData.(tp).W{us}    = bestW;
        GroupData.(tp).H{us}    = bestH;
        GroupData.(tp).Word{us} = bestW;
        GroupData.(tp).Hord{us} = bestH;
        
        Ghat = (bestW*bestH)';                       % time x M
        GroupData.(tp).VAF(us) = vaf_fro(Gtrue, Ghat);

    end % segment loop
end % trial type loop
toc
%% 6) PAD AND ALIGN SYNERGIES (lead-based, chain alignment)
% Requires munkres.m on the path
for t = 1:numel(trial_types)

    tp = trial_types{t};
    if ~isfield(GroupData, tp), continue; end
    seg_ids = GroupData.(tp).segments;
    if isempty(seg_ids), continue; end

    % --- Choose lead segment (same for mean) ---
    trialName = upper(tp);
    if contains(trialName,'RAMPDOWN')
        lead_idx = seg_ids(1);
    elseif contains(trialName,'RAMPUP')
        lead_idx = seg_ids(end);
    else
        lead_idx = seg_ids(1);
    end

    % Define the order: RAMPUP → from last to first, RAMPDOWN → first to last
    if contains(trialName,'RAMPUP')
        chain_order = flip(seg_ids); % start from lead (last) to first
    else
        chain_order = seg_ids;       % start from lead (first) to last
    end

    % ==========================================================
    % A) ALIGN MEAN-BASED SYNERGIES (Word / Hord)
    % ==========================================================
    leadW = GroupData.(tp).Word{lead_idx};
    leadH = GroupData.(tp).Hord{lead_idx};
    if ~isempty(leadW) && ~isempty(leadH)
        % --- Chronologically order leader by H peak ---
        [~,pks] = max(leadH,[],2);
        [~,ord] = sort(pks);
        leadW = leadW(:,ord);
        leadH = leadH(ord,:);

        GroupData.(tp).Word{lead_idx} = leadW;
        GroupData.(tp).Hord{lead_idx} = leadH;

        % >>> NEW: store permutation used for mean-based leader <<<
        GroupData.(tp).lead_perm_mean = ord;

        n_lead = size(leadW,2);

        % --- Pad all segments to n_lead ---
        Word_cells = GroupData.(tp).Word;
        Hord_cells = GroupData.(tp).Hord;
        for us = seg_ids(:)'
            W = Word_cells{us};
            H = Hord_cells{us};
            if isempty(W) || isempty(H), continue; end
            % pad W (muscles x n_lead)
            [nRows,nCols] = size(W);
            if nCols < n_lead
                Wpad = zeros(nRows,n_lead);
                Wpad(:,1:nCols) = W;
            else
                Wpad = W(:,1:n_lead);
            end

            % pad H (n_lead x time)
            [nHRows,nHCols] = size(H);
            if nHRows < n_lead
                Hpad = zeros(n_lead,nHCols);
                Hpad(1:nHRows,:) = H;
            else
                Hpad = H(1:n_lead,:);
            end

            Word_cells{us} = Wpad;
            Hord_cells{us} = Hpad;
        end

        alignedW = Word_cells;
        alignedH = Hord_cells;

        % --- Align segments in chain order ---
        alpha = 0.7; % weight between spatial (W) and temporal (H)
        prevW = leadW;
        prevH = leadH;
        for us = chain_order(:)'
            if us == lead_idx, continue; end
            W = Word_cells{us};
            H = Hord_cells{us};
            if isempty(W) || isempty(H), continue; end
            S = zeros(n_lead,n_lead);
            for k1 = 1:n_lead
                for k2 = 1:n_lead
                    vW1 = prevW(:,k1);
                    vW2 = W(:,k2);
                    vH1 = prevH(k1,:)';
                    vH2 = H(k2,:)';

                    % corr W
                    if all(vW1==0) && all(vW2==0)
                        cW = 1;
                    elseif all(vW1==0) || all(vW2==0)
                        cW = 0;
                    else
                        cW = corr(vW1,vW2,'rows','complete');
                        if isnan(cW), cW = 0; end
                    end

                    % corr H
                    if all(vH1==0) && all(vH2==0)
                        cH = 1;
                    elseif all(vH1==0) || all(vH2==0)
                        cH = 0;
                    else
                        cH = corr(vH1,vH2,'rows','complete');
                        if isnan(cH), cH = 0; end
                    end

                    S(k1,k2) = alpha*cW + (1-alpha)*cH;
                end
            end

            [assign,~] = munkres(-S);

            Wnew = zeros(size(W));
            Hnew = zeros(size(H));
            for k1 = 1:n_lead
                k2 = assign(k1);
                if k2 > 0
                    Wnew(:,k1) = W(:,k2);
                    Hnew(k1,:) = H(k2,:);
                end
            end

            alignedW{us} = Wnew;
            alignedH{us} = Hnew;

            prevW = Wnew;
            prevH = Hnew;
        end

        GroupData.(tp).Word = alignedW;
        GroupData.(tp).Hord = alignedH;
    end
end

%% Manual reorder (optional global view + per-segment editing)
%GroupData = manual_reorder_segments(GroupData, 'RAMPDOWN');
%GroupData = manual_reorder_segments(GroupData, 'RAMPUP');

%% Circular metrics (CoA, FWHM) after alignment
for type = {'RAMPDOWN','RAMPUP'}
    tp = type{1};

    % A) Mean-based synergies (GroupData.(tp).Hord)
    if isfield(GroupData,tp) && isfield(GroupData.(tp),'Hord') && ~isempty(GroupData.(tp).Hord)
        N = numel(GroupData.(tp).Hord);
        % ensure CoA/FWHM exist as cell
        if ~isfield(GroupData.(tp),'CoA')   || isempty(GroupData.(tp).CoA),   GroupData.(tp).CoA   = cell(N,1); end
        if ~isfield(GroupData.(tp),'FWHM') || isempty(GroupData.(tp).FWHM), GroupData.(tp).FWHM = cell(N,1); end

        for us = 1:N
            Hm = GroupData.(tp).Hord{us};

            if isempty(Hm)
                GroupData.(tp).CoA{us}  = [];
                GroupData.(tp).FWHM{us} = [];
                continue;
            end

            theta = deg2rad(linspace(360/size(Hm,2),360,size(Hm,2)));
            coa  = zeros(1,size(Hm,1));
            fwhm = zeros(1,size(Hm,1));

            for s = 1:size(Hm,1)
                Acol = Hm(s,:);
                if all(Acol == 0)
                    coa(s)  = NaN;
                    fwhm(s) = NaN;
                    continue;
                end

                ang = atan2(sum(sin(theta).*Acol), sum(cos(theta).*Acol));
                if ang < 0, ang = ang + 2*pi; end
                coa(s) = rad2deg(ang);

                halfmax = max(Acol)/2;
                if halfmax == 0
                    fwhm(s) = NaN;
                else
                    ix = Acol >= halfmax;
                    props = regionprops(ix,'Area');
                    if isempty(props)
                        fwhm(s) = NaN;
                    else
                        fwhm(s) = max([props.Area]);
                    end
                end
            end

            GroupData.(tp).CoA{us}  = coa;
            GroupData.(tp).FWHM{us} = fwhm;
        end

        disp(['Circular metrics (mean-based) computed for ', tp]);
    end
end


%% save EMG synergies results data
save(fullfile(filepath_save,'EMGSynergiesDATA_WNNMF.mat'), 'GroupData','-v7.3');
disp('saved');

%% ====================================
%  PLOTTING ROUTINES (Mean)
% =====================================
% Define trial types to loop over both
trial_types = {'RAMPUP', 'RAMPDOWN'};
for tp_idx = 1:numel(trial_types)
    tp = trial_types{tp_idx};
    if ~isfield(GroupData, tp)
        warning('GroupData does not have field %s. Skipping.', tp);
        continue;
    end
    segList = GroupData.(tp).segments;

    % Ensure the plot directory exists
    if ~exist(filepath_plot, 'dir')
        mkdir(filepath_plot);
    end

    %% 7a) VAF Curves for MEAN-EMG only (with flipped colors for RAMPUP)
    figure('Name', sprintf('%s - VAF Curves (Mean)', tp), ...
           'NumberTitle','off','Color','w');
    ax = gca; hold(ax,'on');
    
    % --- Fixed 10-color palette ---
    speed_hex = {'#471365','#463480','#3B528B','#2F6C8E','#25848E', ...
                 '#1E9C89','#2FB47C','#5EC962','#9BD93C','#DFE318'};
    
    tmp    = cellfun(@(hh) sscanf(hh(2:end),'%2x%2x%2x',[1 3]) / 255, ...
                     speed_hex, 'UniformOutput', false);
    colors = vertcat(tmp{:});  % 10x3
    
    % --- Flip palette ONLY for incremental condition (RAMPUP) ---
    isRampUp = strcmpi(tp,'RAMPUP') || contains(upper(tp),'RAMPUP');
    if isRampUp
        colors = flipud(colors);
    end
    
    % Safety: ensure enough colors for segments
    if size(colors,1) < numel(segList)
        error('Palette has %d colors but segList has %d segments.', size(colors,1), numel(segList));
    end
    
    plotted = false;
    
    for i = 1:numel(segList)
        us = segList(i);
    
        v_mean = GroupData.(tp).VAFcurve{us};
        if isempty(v_mean)
            continue;
        end
        plotted = true;
    
        % segment index -> color index (after optional flip)
        if us < 1 || us > size(colors,1)
            warning('Segment index us=%d is out of palette range. Using fallback color.', us);
            c = [0 0 0];
        else
            c = colors(us,:);
        end
    
        % VAF curve
        plot(1:numel(v_mean), v_mean, '-o', ...
            'Color', c, ...
            'LineWidth', 3, ...
            'MarkerSize', 7, ...
            'MarkerFaceColor', c, ...
            'MarkerEdgeColor', c, ...
            'DisplayName', sprintf('Seg %d', us));
    
        % Highlight selected/optimal synergy number
        k_m = GroupData.(tp).num_synergies(us);
        if ~isempty(k_m) && k_m >= 1 && k_m <= numel(v_mean)
            plot(k_m, v_mean(k_m), 'o', ...
                'MarkerSize', 8, ...
                'LineWidth', 1.5, ...
                'MarkerFaceColor', [1 1 1], ...
                'MarkerEdgeColor', [0.85 0 0], ...
                'HandleVisibility', 'off');
        end
    end
    
    if ~plotted
        text(0.5, 0.5, 'No mean-EMG VAF data available', ...
            'HorizontalAlignment','center', 'Parent', ax);
    end
    
    xlabel('Number of Synergies','FontWeight','bold');
    ylabel('VAF','FontWeight','bold');
    title(sprintf('%s: Mean-EMG VAF Curves', tp),'FontWeight','bold');
    
    legend('Location','eastoutside');
    grid on; box off; hold(ax,'off');
    
    savefig(gcf, fullfile(filepath_plot, sprintf('%s_VAFcurve_Mean_WNNMF.fig', tp)));


    % 7c) SYNERGY WEIGHTS & ACTIVATIONS - MEAN-EMG (Robust to missing Hord / NaNs)

    if isfield(GroupData, tp) && isfield(GroupData.(tp), 'Word')
    
        % -------- Define muscle names (update these to your dataset)
        %muscleNames = ref_order;
        % (Add/remove names to match size(W,1))
    
        % -------- Build segment list robustly (Word & Hord present and non-empty)
        nWords = numel(GroupData.(tp).Word);
        segList = [];
        hasHordField = isfield(GroupData.(tp), 'Hord');
        for ii = 1:nWords
            Wok = ~isempty(GroupData.(tp).Word{ii});
            Hok = hasHordField && numel(GroupData.(tp).Hord) >= ii && ~isempty(GroupData.(tp).Hord{ii});
            if Wok && Hok
                segList(end+1) = ii; 
            end
        end
        if isempty(segList)
            warning('No segments with both Word and Hord found for %s.', tp);
            continue
        end
    
        % -------- Determine max # of synergies across segments (aligned sizes)
        perSegK = zeros(1, numel(segList));
        for sIdx = 1:numel(segList)
            s = segList(sIdx);
            W = GroupData.(tp).Word{s};
            H = GroupData.(tp).Hord{s};
            kW = 0; kH = 0;
            if ~isempty(W), kW = size(W,2); end
            if ~isempty(H), kH = size(H,1); end
            perSegK(sIdx) = max(kW, kH);
        end
        max_synergies = max(perSegK);
        if max_synergies == 0
            warning('Found segments but no synergies (max_synergies == 0).');
            return
        end
    
        % -------- Global plotting extents (robust to NaNs)
        globalWmax = 0; maxHlen = 0;
        for sIdx = 1:numel(segList)
            s = segList(sIdx);
            W = GroupData.(tp).Word{s};
            if ~isempty(W)
                wmax = max(W(:), [], 'omitnan');
                if ~isempty(wmax) && ~isnan(wmax) && wmax > globalWmax
                    globalWmax = wmax;
                end
            end
            H = GroupData.(tp).Hord{s};
            if ~isempty(H)
                maxHlen = max(maxHlen, size(H,2));
            end
        end
        if globalWmax == 0, globalWmax = 1; end
        if maxHlen   == 0, maxHlen   = 1; end
    
        colors    = lines(max_synergies); % keep your palette
        nSegments = numel(segList);
        baseFont  = 10;
    
        % =========================
        % PLOT W (weights) — bars
        % =========================
        globalWmax = 0;
        for sIdx = 1:numel(segList)
            W = GroupData.(tp).Word{segList(sIdx)};
            if ~isempty(W)
                wm = max(W(:), [], 'omitnan');
                if wm > globalWmax, globalWmax = wm; end
            end
        end
        if globalWmax == 0, globalWmax = 1; end
        W_ylim = [0, globalWmax * 1.1];
    
        figW = figure('Name', sprintf('%s Synergies (Mean) - W', tp), ...
                      'Color','w','Units','inches','Position',[0 0 9 6]);
        tlW = tiledlayout(nSegments, max_synergies, 'Padding','compact','TileSpacing','compact');
        sgtitle(tlW, sprintf('%s Synergies – W (Mean)', tp), 'FontWeight','bold','FontSize',16);
    
        for j = 1:nSegments
            seg = segList(j);
            W = GroupData.(tp).Word{seg};
            Kw = 0; if ~isempty(W), Kw = size(W,2); end
            nBars = size(W,1);
            midX  = nBars/2 + 0.5;
    
            for syn = 1:max_synergies
                ax = nexttile(tlW, (j-1)*max_synergies + syn);
                hold(ax,'on');
    
                if syn <= Kw && ~isempty(W) && ~all(isnan(W(:,syn)))
                    thisFace = colors(syn,:);
                    edgeCol  = thisFace * 0.35; % subtle darker edge
    
                    bar(ax, W(:,syn), 'FaceColor', thisFace, ...
                                       'EdgeColor', edgeCol, 'LineWidth', 0.5);
                    if nBars >= 2
                        xline(ax, midX, '--k', 'LineWidth', 0.6);
                    end
    
                    ylim(ax, W_ylim);
                    xlim(ax, [0.5, nBars + 0.5]);
                    ax.Box = 'on';
                    ax.LineWidth = 0.75;
                    ax.FontSize = baseFont;
                    ax.XTick = [];
                    if syn == 1
                        ax.YTickMode = 'auto';     % show ticks automatically
                    else
                        ax.YTick = [];             % hide ticks for other columns
                    end
    
                    % Show X axis ONLY on bottom-left subplot
                    if j == nSegments && syn == 1
                        ax.XTick = 1:nBars;
                    
                        % Safely slice or pad labels from ref_order
                        nRef = numel(ref_order);
                        if nBars <= nRef
                            labels = ref_order(1:nBars);
                        else
                            extra = arrayfun(@(k) sprintf('M%d', k), 1:(nBars-nRef), 'UniformOutput', false);
                            labels = [ref_order(:).', extra];  % pad if needed
                        end
                    
                        % If too many labels, thin them for readability (optional)
                        if nBars > 30
                            showIdx = unique([1:2:nBars, nBars]);  % every 2 + last
                            ax.XTick = showIdx;
                            labels = labels(showIdx);
                        end
                    
                        ax.XTickLabel = labels;
                        ax.XTickLabelRotation = 90;
                        ax.TickDir = 'out';
                        xlabel(ax, 'Muscles', 'FontWeight','normal');
                    
                        % keep Y ticks hidden
                        if syn == 1
                            ax.YTickMode = 'auto';     % show ticks automatically
                        else
                            ax.YTick = [];             % hide ticks for other columns
                        end
                    end

                else
                    axis(ax,'off');
                    text(0.5,0.5,'missing','Units','normalized', ...
                        'HorizontalAlignment','center','VerticalAlignment','middle', ...
                        'FontAngle','italic','FontSize',baseFont,'Color',[0.4 0.4 0.4]);
                end
    
                if j == 1
                    title(ax, sprintf('Syn %d', syn), 'FontWeight','bold','FontSize',baseFont+1);
                end
                hold(ax,'off');
            end
        end
    
        % =========================
        % PLOT H (activations) — lines
        % =========================
        globalHmax = 0; maxHlen = 0;
        for sIdx = 1:numel(segList)
            Htmp = GroupData.(tp).Hord{segList(sIdx)};
            if ~isempty(Htmp)
                hm = max(Htmp(:), [], 'omitnan');
                if hm > globalHmax, globalHmax = hm; end
                maxHlen = max(maxHlen, size(Htmp,2));
            end
        end
        if globalHmax == 0, globalHmax = 1; end
        if maxHlen   == 0, maxHlen   = 1; end
        H_ylim = [0, globalHmax * 1.1];
    
        figH = figure('Name', sprintf('%s Synergies (Mean) - H', tp), ...
                      'Color','w','Units','inches','Position',[0 0 9 6]);
        tlH = tiledlayout(nSegments, max_synergies, 'Padding','compact','TileSpacing','compact');
        sgtitle(tlH, sprintf('%s Synergies – H (Mean)', tp), 'FontWeight','bold','FontSize',16);
    
        for j = 1:nSegments
            seg = segList(j);
            H = GroupData.(tp).Hord{seg};
            Kh = 0; if ~isempty(H), Kh = size(H,1); end
    
            for syn = 1:max_synergies
                ax = nexttile(tlH, (j-1)*max_synergies + syn);
                hold(ax,'on');
    
                if syn <= Kh && ~isempty(H) && ~all(isnan(H(syn,:)))
                    hrow = nan(1, maxHlen);
                    len  = size(H,2);
                    hrow(1:len) = H(syn,:);
                    %plot(ax, linspace(0,100,maxHlen), hrow, '-', 'Color', colors(syn,:), 'LineWidth', 1.0);

                    xgc = linspace(0,100,maxHlen);
                    % --- Smooth H (robust to NaNs)
                    h_s = hrow;
                    % Fill NaNs only for smoothing (keeps your original hrow intact)
                    ok = ~isnan(h_s);
                    if nnz(ok) >= 3
                        h_s(~ok) = interp1(xgc(ok), h_s(ok), xgc(~ok), 'linear', 'extrap');
                    
                        % Moving-average window (in samples); tune as you like
                        % (e.g., 5–11 is usually a good range for 100–200 points)
                        win = max(3, round(0.05 * maxHlen));   % ~5% of the cycle
                        if mod(win,2) == 0, win = win + 1; end % odd window
                    
                        h_s = movmean(h_s, win, 'omitnan');    % smooth
                    else
                        h_s = hrow; % not enough points to smooth reliably
                    end
                    % --- Plot (thicker line)
                    plot(ax, xgc, h_s, '-', 'Color', colors(syn,:), 'LineWidth', 1.6);
    
                    ylim(ax, H_ylim);
                    xlim(ax, [0, 100]);
                    ax.Box = 'on';
                    ax.LineWidth = 0.75;
                    ax.FontSize = baseFont;
                    ax.XTick = [];
                    if syn == 1
                        ax.YTickMode = 'auto';     % show ticks automatically
                    else
                        ax.YTick = [];             % hide ticks for other columns
                    end
    
                    % Show X axis ONLY on bottom-left subplot
                    if j == nSegments && syn == 1
                        ax.XTick = 0:20:100;
                        ax.TickDir = 'out';
                        xlabel(ax, 'Gait cycle (%)', 'FontWeight','normal');
                    end
                else
                    axis(ax,'off');
                    text(0.5,0.5,'missing','Units','normalized', ...
                        'HorizontalAlignment','center','VerticalAlignment','middle', ...
                        'FontAngle','italic','FontSize',baseFont,'Color',[0.4 0.4 0.4]);
                end
    
                if j == 1
                    title(ax, sprintf('Syn %d', syn), 'FontWeight','bold','FontSize',baseFont+1);
                end
                hold(ax,'off');
            end
        end
    
        % -------- High-DPI export (+ optional .fig)
        if ~exist(filepath_plot, 'dir'), mkdir(filepath_plot); end
        %outW = fullfile(filepath_plot, sprintf('%s_Synergies_W_mean.png', tp));
        %outH = fullfile(filepath_plot, sprintf('%s_Synergies_H_mean.png', tp));
        %exportgraphics(figW, outW, 'Resolution', 600);
        %exportgraphics(figH, outH, 'Resolution', 600);
        saveas(figW, fullfile(filepath_plot, sprintf('%s_Synergies_W_mean_WNNMF.fig', tp)));
        saveas(figH, fullfile(filepath_plot, sprintf('%s_Synergies_H_mean_WNNMF.fig', tp)));
    
    else
        warning('GroupData.%s.Word not found.', tp);
    end
end

%%  -------------------- FUNCTION --------------------

function v = vaf_fro(G, Ghat)
% VAF = 1 - SSE/SST where SST is energy of original signal
% Assumes NO NaNs in G and Ghat.
    num = sum((G(:) - Ghat(:)).^2);
    den = sum((G(:)).^2);
    if den <= eps
        v = NaN;
    else
        v = 1 - (num/den);
    end
end

function GroupData = manual_reorder_segments(GroupData, tp)
    % Open a global view of all segments, allow editing one segment at a time.
    % Edits are committed to GroupData.(tp).Word / .Hord.
    %
    % Inputs:
    %   GroupData - structure with fields .(tp).Word and .(tp).Hord
    %   tp        - string, trial type ('RAMPDOWN','RAMPUP', etc.)
    %
    % Output:
    %   GroupData - updated with manual reorderings
    
    % ---------------------- GLOBAL VIEW + PER-SEGMENT EDITING ----------------------
    % Place AFTER you've built GroupData.(tp).Word / GroupData.(tp).Hord for the trial type tp
    if usejava('desktop') && feature('ShowFigureWindows')
        % Build segList robustly
        if isfield(GroupData, tp) && isfield(GroupData.(tp), 'Word')
            nWords = numel(GroupData.(tp).Word);
            segList = [];
            hasHordField = isfield(GroupData.(tp), 'Hord');
            for ii = 1:nWords
                Wok = ~isempty(GroupData.(tp).Word{ii});
                Hok = hasHordField && numel(GroupData.(tp).Hord) >= ii && ~isempty(GroupData.(tp).Hord{ii});
                if Wok && Hok
                    segList(end+1) = ii; 
                end
            end
        else
            segList = [];
        end
    
        if isempty(segList)
            warning('No segments with both Word and Hord found for %s.', tp);
        else
            % prepare global sizes
            perSegK = zeros(1, numel(segList));
            muscleCount = 0;
            maxHlen = 0;
            for sIdx = 1:numel(segList)
                s = segList(sIdx);
                W = GroupData.(tp).Word{s};
                H = GroupData.(tp).Hord{s};
                kW = 0; kH = 0;
                if ~isempty(W), kW = size(W,2); muscleCount = max(muscleCount, size(W,1)); end
                if ~isempty(H), kH = size(H,1); maxHlen = max(maxHlen, size(H,2)); end
                perSegK(sIdx) = max(kW, kH);
            end
            finalK = max(perSegK);
            if finalK == 0
                warning('Found segments but no synergies (finalK == 0).');
                return;
            end
            if muscleCount == 0, muscleCount = 1; end
            if maxHlen == 0, maxHlen = 1; end
    
            colors = lines(finalK);
    
            % create padded cell arrays for global plotting and manipulation
            Wcells = cell(1, numel(segList));
            Hcells = cell(1, numel(segList));
            for j = 1:numel(segList)
                seg = segList(j);
                W = GroupData.(tp).Word{seg};
                H = GroupData.(tp).Hord{seg};
                Wpad = nan(muscleCount, finalK);
                Hpad = nan(finalK, maxHlen);
                if ~isempty(W), Wpad(1:size(W,1), 1:size(W,2)) = W; end
                if ~isempty(H), Hpad(1:size(H,1), 1:size(H,2)) = H; end
                Wcells{j} = Wpad;
                Hcells{j} = Hpad;
            end
    
            % global plotting extents
            globalWmax = 0;
            for j = 1:numel(Wcells)
                wm = max(Wcells{j}(:));
                if ~isempty(wm) && ~isnan(wm) && wm > globalWmax, globalWmax = wm; end
            end
            if globalWmax == 0, globalWmax = 1; end
            W_ylim = [0 globalWmax*1.1];
    
            globalHmax = 0;
            for j = 1:numel(Hcells)
                hm = max(Hcells{j}(:));
                if ~isempty(hm) && ~isnan(hm) && hm > globalHmax, globalHmax = hm; end
            end
            if globalHmax == 0, globalHmax = 1; end
            H_ylim = [0 globalHmax*1.1];
    
            % originalWcells = Wcells;
            % originalHcells = Hcells;
            keepGlobal = true;
    
            while keepGlobal
                % show global W
                figW = figure('Name', sprintf('%s Synergies (Mean) - W (global)', tp), 'Color', 'w');
                tlW = tiledlayout(numel(segList), finalK, 'Padding', 'compact', 'TileSpacing', 'compact');
                sgtitle(tlW, sprintf('%s Synergies – W (Mean) (global view)', tp), 'FontWeight', 'bold', 'FontSize', 14);
    
                for j = 1:numel(segList)
                    Wpad = Wcells{j};
                    for syn = 1:finalK
                        ax = nexttile(tlW, (j-1)*finalK + syn);
                        if syn <= size(Wpad,2) && ~all(isnan(Wpad(:,syn)))
                            bar(ax, Wpad(:,syn), 'FaceColor', colors(syn,:), 'EdgeColor', 'none');
                            ylim(ax, W_ylim);
                            ax.XTick = []; ax.YTick = [];
                        else
                            axis(ax, 'off');
                            text(0.5, 0.5, 'missing', 'Units','normalized', ...
                                 'HorizontalAlignment','center','VerticalAlignment','middle', ...
                                 'FontAngle','italic','FontSize',9,'Color',[0.4 0.4 0.4]);
                        end
                        if j == 1, title(ax, sprintf('Syn %d', syn)); end
                    end
                    annotation(figW, 'textbox', [0.005, 1 - (j-0.5)/numel(segList) - 1/(2*numel(segList)), 0.03, 1/numel(segList)], ...
                               'String', sprintf('Seg %d', segList(j)), 'FitBoxToText','on', 'EdgeColor','none', ...
                               'FontSize', 9, 'FontWeight','bold', ...
                               'HorizontalAlignment','left', 'VerticalAlignment','middle');
                end
    
                % show global H
                figH = figure('Name', sprintf('%s Synergies (Mean) - H (global)', tp), 'Color', 'w');
                tlH = tiledlayout(numel(segList), finalK, 'Padding', 'compact', 'TileSpacing', 'compact');
                sgtitle(tlH, sprintf('%s Synergies – H (Mean) (global view)', tp), 'FontWeight', 'bold', 'FontSize', 14);
                for j = 1:numel(segList)
                    Hpad = Hcells{j};
                    for syn = 1:finalK
                        ax = nexttile(tlH, (j-1)*finalK + syn);
                        if syn <= size(Hpad,1) && ~all(isnan(Hpad(syn,:)))
                            hrow = nan(1, maxHlen);
                            len = size(Hpad,2);
                            hrow(1:len) = Hpad(syn,:);
                            plot(ax, 1:maxHlen, hrow, '-', 'Color', colors(syn,:), 'LineWidth', 1.2);
                            ylim(ax, H_ylim);
                            xlim(ax, [1 maxHlen]);
                            ax.XTick = []; ax.YTickMode = 'auto';
                        else
                            axis(ax, 'off');
                            text(0.5, 0.5, 'missing', 'Units','normalized', ...
                                 'HorizontalAlignment','center','VerticalAlignment','middle', ...
                                 'FontAngle','italic','FontSize',9,'Color',[0.4 0.4 0.4]);
                        end
                        if j == 1, title(ax, sprintf('Syn %d', syn)); end
                    end
                    annotation(figH, 'textbox', [0.005, 1 - (j-0.5)/numel(segList) - 1/(2*numel(segList)), 0.03, 1/numel(segList)], ...
                               'String', sprintf('Seg %d', segList(j)), 'FitBoxToText','on', 'EdgeColor','none', ...
                               'FontSize', 9, 'FontWeight','bold', ...
                               'HorizontalAlignment','left', 'VerticalAlignment','middle');
                end
    
                % Ask user which segment to edit (single selection)
                segLabels = arrayfun(@(s) sprintf('Seg %d', s), segList, 'UniformOutput', false);
                [sel, ok] = listdlg('ListString', segLabels, 'SelectionMode', 'single', ...
                                    'PromptString', 'Select a segment to EDIT (or Cancel to finish):');
                % close global figs to avoid clutter (they will be reopened next iteration)
                if ishandle(figW), close(figW); end
                if ishandle(figH), close(figH); end
    
                if isempty(sel) || ~ok
                    % finish or cancel -> confirm
                    q = questdlg('Finish editing all segments?', 'Finish', 'Yes','No','Yes');
                    if strcmp(q,'Yes')
                        keepGlobal = false;
                        break;
                    else
                        continue; % redisplay global view
                    end
                end
    
                % get selected segment index and data
                jsel = sel;
                segToEdit = segList(jsel);
                Wpad = Wcells{jsel};
                Hpad = Hcells{jsel};
                % determine this segment's "actual" original sizes to preserve when saving
                Worig = GroupData.(tp).Word{segToEdit};
                Horig = GroupData.(tp).Hord{segToEdit};
                origWrows = 0; origWcols = 0; origHrows = 0; origHcols = 0;
                if ~isempty(Worig), [origWrows, ~] = size(Worig); end
                if ~isempty(Horig), [~, origHcols] = size(Horig); end
    
                % Single-segment edit loop
                keepEditingSeg = true;
                while keepEditingSeg
                    % show W/H for this single segment (in the same tiled single-row style)
                    figSegW = figure('Name', sprintf('%s - Seg %d — W (edit)', tp, segToEdit), 'Color', 'w');
                    tlSegW = tiledlayout(1, finalK, 'Padding', 'compact', 'TileSpacing', 'compact');
                    sgtitle(tlSegW, sprintf('%s — W (Seg %d)', tp, segToEdit), 'FontWeight','bold');
    
                    for syn = 1:finalK
                        ax = nexttile(tlSegW, syn);
                        if syn <= size(Wpad,2) && ~all(isnan(Wpad(:,syn)))
                            bar(ax, Wpad(:,syn), 'FaceColor', colors(syn,:), 'EdgeColor', 'none');
                            ylim(ax, W_ylim);
                            ax.XTick = []; ax.YTick = [];
                        else
                            axis(ax, 'off');
                            text(0.5, 0.5, 'missing', 'Units','normalized', ...
                                 'HorizontalAlignment','center','VerticalAlignment','middle', ...
                                 'FontAngle','italic','FontSize',9,'Color',[0.4 0.4 0.4]);
                        end
                        if syn == 1, title(ax, sprintf('Syn %d', syn)); end
                    end
    
                    figSegH = figure('Name', sprintf('%s - Seg %d — H (edit)', tp, segToEdit), 'Color', 'w');
                    tlSegH = tiledlayout(1, finalK, 'Padding', 'compact', 'TileSpacing', 'compact');
                    sgtitle(tlSegH, sprintf('%s — H (Seg %d)', tp, segToEdit), 'FontWeight','bold');
    
                    for syn = 1:finalK
                        ax = nexttile(tlSegH, syn);
                        if syn <= size(Hpad,1) && ~all(isnan(Hpad(syn,:)))
                            hrow = nan(1, maxHlen);
                            len = size(Hpad,2);
                            hrow(1:len) = Hpad(syn,:);
                            plot(ax, 1:maxHlen, hrow, '-', 'Color', colors(syn,:), 'LineWidth', 1.2);
                            ylim(ax, H_ylim);
                            xlim(ax, [1 maxHlen]);
                            ax.XTick = []; ax.YTickMode = 'auto';
                        else
                            axis(ax, 'off');
                            text(0.5, 0.5, 'missing', 'Units','normalized', ...
                                 'HorizontalAlignment','center','VerticalAlignment','middle', ...
                                 'FontAngle','italic','FontSize',9,'Color',[0.4 0.4 0.4]);
                        end
                        if syn == 1, title(ax, sprintf('Syn %d', syn)); end
                    end
    
                    % prompt permutation for this single segment
                    prompt = {sprintf('Enter NEW order for components for Seg %d (1..%d) as comma/space separated indices\n(e.g. "2 1 3") or leave empty to keep current order:', segToEdit, finalK)};
                    dlgtitle = sprintf('Manual reorder — Seg %d', segToEdit);
                    dims = [1 100];
                    definput = {''};
                    answer = inputdlg(prompt, dlgtitle, dims, definput);
    
                    if isempty(answer)
                        % user cancelled -> ask whether to go back to global or retry
                        choice = questdlg('Cancel editing this segment? Return to global view?', 'Cancel', 'Yes','Edit again','Yes');
                        if strcmp(choice,'Yes')
                            keepEditingSeg = false;
                            % close figs and continue to global view (no change)
                            if ishandle(figSegW), close(figSegW); end
                            if ishandle(figSegH), close(figSegH); end
                            break;
                        else
                            close(figSegW); close(figSegH);
                            continue;
                        end
                    end
    
                    s = strtrim(answer{1});
                    if isempty(s)
                        % treat as keep current
                        keepEditingSeg = false;
                        if ishandle(figSegW), close(figSegW); end
                        if ishandle(figSegH), close(figSegH); end
                        break;
                    end
    
                    ord = str2num(s); 
                    if isempty(ord) || ~all(ismember(ord, 1:finalK)) || numel(ord) ~= finalK || numel(unique(ord)) ~= finalK
                        uiwait(errordlg(sprintf('Invalid order. You must supply a permutation of 1:%d (all indices exactly once).', finalK),'Invalid input','modal'));
                        close(figSegW); close(figSegH);
                        continue;
                    end
    
                    % Build preview for this segment only
                    Wp = nan(size(Wpad));
                    Hp = nan(size(Hpad));
                    Wp(:,1:finalK) = Wpad(:, ord);
                    Hp(1:finalK,:) = Hpad(ord, :);
    
                    % close old and show previews
                    close(figSegW); close(figSegH);
    
                    % show preview W/H for the segment
                    figPreviewW = figure('Name', sprintf('%s - Seg %d — W (preview)', tp, segToEdit), 'Color', 'w');
                    tlPW = tiledlayout(1, finalK, 'Padding','compact','TileSpacing','compact');
                    sgtitle(tlPW, sprintf('%s — W (Seg %d) (preview)', tp, segToEdit), 'FontWeight','bold');
                    for syn = 1:finalK
                        ax = nexttile(tlPW, syn);
                        if ~all(isnan(Wp(:,syn)))
                            bar(ax, Wp(:,syn), 'FaceColor', colors(syn,:), 'EdgeColor', 'none');
                            ylim(ax, W_ylim);
                            ax.XTick = []; ax.YTick = [];
                        else
                            axis(ax, 'off');
                            text(0.5,0.5,'missing','Units','normalized','HorizontalAlignment','center');
                        end
                    end
    
                    figPreviewH = figure('Name', sprintf('%s - Seg %d — H (preview)', tp, segToEdit), 'Color', 'w');
                    tlPH = tiledlayout(1, finalK, 'Padding','compact','TileSpacing','compact');
                    sgtitle(tlPH, sprintf('%s — H (Seg %d) (preview)', tp, segToEdit), 'FontWeight','bold');
                    for syn = 1:finalK
                        ax = nexttile(tlPH, syn);
                        if ~all(isnan(Hp(syn,:)))
                            hrow = nan(1, maxHlen);
                            len = size(Hp,2);
                            hrow(1:len) = Hp(syn,:);
                            plot(ax, 1:maxHlen, hrow, '-', 'Color', colors(syn,:), 'LineWidth', 1.2);
                            ylim(ax, H_ylim); xlim(ax, [1 maxHlen]); ax.XTick=[];
                        else
                            axis(ax, 'off');
                            text(0.5,0.5,'missing','Units','normalized','HorizontalAlignment','center');
                        end
                    end
    
                    % Accept / Retry / Revert
                    choice2 = questdlg(sprintf('Accept permutation for Seg %d?', segToEdit), 'Preview', 'Yes','Edit again','Cancel (revert)','Yes');
                    if strcmp(choice2,'Yes')
                        % commit permutation only for this segment to GroupData
                        % build padded real arrays then permute and trim trailing NaNs
                        Wreal = nan(muscleCount, finalK);
                        Hreal = nan(finalK, maxHlen);
                        if ~isempty(Worig), Wreal(1:size(Worig,1), 1:size(Worig,2)) = Worig; end
                        if ~isempty(Horig), Hreal(1:size(Horig,1), 1:size(Horig,2)) = Horig; end
                        Wreal = Wreal(:, ord);
                        Hreal = Hreal(ord, :);
                        lastWcol = find(~all(isnan(Wreal),1), 1, 'last');
                        if isempty(lastWcol), lastWcol = 0; end
                        lastHrow = find(~all(isnan(Hreal),2), 1, 'last');
                        if isempty(lastHrow), lastHrow = 0; end
                        % write back trimmed to original sensible sizes
                        if lastWcol > 0
                            GroupData.(tp).Word{segToEdit} = Wreal(1:origWrows + (max(0, size(Wreal,1)-origWrows)), 1:lastWcol);
                            % above keeps original muscle rows if present, otherwise keeps muscleCount rows
                        else
                            GroupData.(tp).Word{segToEdit} = [];
                        end
                        if lastHrow > 0
                            GroupData.(tp).Hord{segToEdit} = Hreal(1:lastHrow, 1:origHcols + (max(0,size(Hreal,2)-origHcols)));
                        else
                            GroupData.(tp).Hord{segToEdit} = [];
                        end
    
                        % update Wcells/Hcells for global view from updated GroupData
                        Wu = GroupData.(tp).Word{segToEdit};
                        Hu = GroupData.(tp).Hord{segToEdit};
                        WpadNew = nan(muscleCount, finalK);
                        HpadNew = nan(finalK, maxHlen);
                        if ~isempty(Wu), WpadNew(1:size(Wu,1),1:size(Wu,2)) = Wu; end
                        if ~isempty(Hu), HpadNew(1:size(Hu,1),1:size(Hu,2)) = Hu; end
                        Wcells{jsel} = WpadNew;
                        Hcells{jsel} = HpadNew;
    
                        % close previews and go back to global view
                        if ishandle(figPreviewW), close(figPreviewW); end
                        if ishandle(figPreviewH), close(figPreviewH); end
                        keepEditingSeg = false;
                    elseif strcmp(choice2,'Cancel (revert)')
                        % revert: do nothing, close previews
                        if ishandle(figPreviewW), close(figPreviewW); end
                        if ishandle(figPreviewH), close(figPreviewH); end
                        keepEditingSeg = false;
                    else
                        % edit again -> close previews and re-open edit loop
                        if ishandle(figPreviewW), close(figPreviewW); end
                        if ishandle(figPreviewH), close(figPreviewH); end
                        continue;
                    end
                end % end single-seg edit
                % loop back to global view (Wcells/Hcells updated)
            end % while keepGlobal
        end % else segList empty
    end % if desktop
end

