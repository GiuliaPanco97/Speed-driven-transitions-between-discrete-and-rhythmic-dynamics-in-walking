clc; clear;

% --- Specify input and output paths -------------------------------------
filepath_load   = 'path';   % Folder containing subject .mat files
filepath_save   = 'path'; % File to save group results
filepath_plot   = 'path';
filepath_plot_synxsubj = 'path';

%% 1) DEFINE PARAMETERS
trial_types             = {'RAMPUP','RAMPDOWN'};
max_synergies           = 15;
num_iter                = 100;
opts                    = statset('MaxIter',1000,'Display','off');
alpha                   = 0.8;  % α = 0 → only temporal activation similarity; α = 1 → only muscle weight similarity; 
ref_order = {'TAr','SOLr','PERr','GMr','GLr','RFr','VLr','VMr','BFr','SEMr', ...
             'SARTr','GMEDr','TFLr','GLMr','ESr','TAl','SOLl','PERl','GMl', ...
             'GLl','RFl','VLl','VMl','BFl','SEMl','SARTl','GMEDl','TFLl','GLMl','ESl'};

%% 2) LOAD DATA
all_files = dir(fullfile(filepath_load,'*.mat'));
valid_files = all_files(arrayfun(@(f) f.name(1)~='.',all_files));
AllSubjectData = struct();
for f = valid_files'
    data = load(fullfile(filepath_load,f.name));
    fname = matlab.lang.makeValidName(f.name(1:end-4));
    AllSubjectData.(fname) = data;
end
subjects = fieldnames(AllSubjectData);
save(fullfile(filepath_save,'AllSubjectEMGData.mat'),'AllSubjectData','subjects', '-v7.3');

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
        'VAFcurve',{{}}, 'num_synergies',[], 'W',{{}}, 'H',{{}}, 'Word',{{}}, 'Hord',{{}}, 'VAF',[], 'CoA',{{}}, 'FWHM',{{}}, ...
        'VAFcurve_stride',{{}}, 'num_synergies_stride',[], 'W_stride',{{}}, 'H_stride',{{}}, 'VAF_stride',[] ...
    );
end

%% === SUBJECT-LEVEL NNMF (average all strides per speed) ===
SubjectSynergies = struct();

for i = 1:numel(subjects)
    sname = subjects{i};
    S = AllSubjectData.(sname);
    if ~isfield(S,'trial_name'), continue; end

    mus = S.muscle_names(:);
    % map muscles to global muscle list
    mapIdx = nan(1,M);
    for j = 1:M
        idx = find(strcmp(all_muscles{j}, mus));
        if ~isempty(idx), mapIdx(j) = idx; end
    end

    tp = upper(S.trial_name); tp = matlab.lang.makeValidName(tp);
    SubjectSynergies.(sname).trial_name = S.trial_name;
    SubjectSynergies.(sname).segments = [];
    SubjectSynergies.(sname).mean_seg = {};
    SubjectSynergies.(sname).std_seg  = {};
    SubjectSynergies.(sname).VAFcurve = {};
    SubjectSynergies.(sname).num_synergies = [];
    SubjectSynergies.(sname).W = {}; SubjectSynergies.(sname).H = {};
    SubjectSynergies.(sname).Word = {}; SubjectSynergies.(sname).Hord = {};
    SubjectSynergies.(sname).VAF = [];

    refW_sub = []; refH_sub = [];

    % collect segment ids
    seg_ids = [];
    if isfield(S,'mean_emg_seg')
        seg_ids = [seg_ids, find(~cellfun(@isempty,S.mean_emg_seg))'];
    end
    if isfield(S,'norm_emg_seg')
        seg_ids = [seg_ids, find(~cellfun(@isempty,S.norm_emg_seg))'];
    end
    seg_ids = unique(seg_ids);

    if isempty(seg_ids), continue; end
    SubjectSynergies.(sname).segments = seg_ids;
    maxSeg = max(seg_ids);

    % pre-alloc
    SubjectSynergies.(sname).mean_seg = cell(maxSeg,1);
    SubjectSynergies.(sname).std_seg  = cell(maxSeg,1);
    SubjectSynergies.(sname).VAFcurve= cell(maxSeg,1);
    SubjectSynergies.(sname).num_synergies=zeros(maxSeg,1);
    SubjectSynergies.(sname).W=cell(maxSeg,1); SubjectSynergies.(sname).H=cell(maxSeg,1);
    SubjectSynergies.(sname).Word=cell(maxSeg,1); SubjectSynergies.(sname).Hord=cell(maxSeg,1);
    SubjectSynergies.(sname).VAF=zeros(maxSeg,1);

    for us = seg_ids
        % --- Extract EMG segment ---
        if isfield(S,'norm_emg_seg') && ~isempty(S.norm_emg_seg{us})
            Y = S.norm_emg_seg{us}; % time x muscles x strides
    
            if ndims(Y)==3
                if use_concat
                    % ===== Concatenate all strides =====
                    [Tseg, Mseg, Nstrides] = size(Y);
                    G = reshape(Y, Tseg*Nstrides, Mseg);   % (time*strides) x muscles
                    Gstd = zeros(size(G));                 % no std in concat mode
                else
                    % ===== Mean across strides =====
                    G = mean(Y,3,'omitnan'); % time x muscles
                    Gstd = std(Y,0,3,'omitnan');
                end
            else
                G = Y;                     % already time x muscles
                Gstd = zeros(size(G));
            end
    
        elseif isfield(S,'mean_emg_seg') && ~isempty(S.mean_emg_seg{us})
            % fallback if only mean provided
            G = S.mean_emg_seg{us};
            Gstd = zeros(size(G));
        else
            continue;
        end
    
        % --- Map to full muscle list (unchanged) ---
        Gref = nan(size(G,1), M);
        Gstd_ref = nan(size(Gstd,1), M);
        for j=1:M
            if ~isnan(mapIdx(j))
                Gref(:,j) = G(:, mapIdx(j));
                if ~isempty(Gstd), Gstd_ref(:,j) = Gstd(:, mapIdx(j)); end
            else
                Gref(:,j) = 0; Gstd_ref(:,j) = 0;
            end
        end
    
        % --- Reorder muscles to ref_order (unchanged) ---
        [~, ref_idx] = ismember(ref_order, all_muscles);
        Gref_reordered = zeros(size(Gref,1), numel(ref_order));
        Gstd_ref_reordered = zeros(size(Gstd_ref,1), numel(ref_order));
        for rr = 1:numel(ref_order)
            if ref_idx(rr) > 0
                Gref_reordered(:,rr) = Gref(:, ref_idx(rr));
                Gstd_ref_reordered(:,rr) = Gstd_ref(:, ref_idx(rr));
            else
                Gref_reordered(:,rr) = 0;
                Gstd_ref_reordered(:,rr) = 0;
            end
        end
        Gref = Gref_reordered;
        Gstd_ref = Gstd_ref_reordered;
        M = numel(ref_order);
    
        % replace NaNs with 0 for NNMF
        Gref(isnan(Gref)) = 0;
        Gstd_ref(isnan(Gstd_ref)) = 0;
    
        % --- Save mean & std (if concat, std will just be zeros) ---
        SubjectSynergies.(sname).mean_seg{us} = Gref;
        SubjectSynergies.(sname).std_seg{us}  = Gstd_ref;
        
        % --- VAF sweep over k ---
        Vc = nan(1,max_synergies);
        for k = 1:max_synergies
            bestV = -inf; bestW = []; bestH = [];
            for r = 1:num_iter
                try
                    [Wm,Hm] = nnmf(Gref',k,'Options',opts);
                    R = (Wm*Hm)'; 
                    v = 1 - sum((Gref-R).^2,'all')/sum(Gref.^2,'all');
                    if v > bestV
                        bestV = v; bestW = Wm; bestH = Hm;
                    end
                catch
                    % ignore failed factorization
                end
            end
            Vc(k) = bestV;
        end
        SubjectSynergies.(sname).VAFcurve{us} = Vc;
        
        % --- pick kopt ---
        idx_k = find(Vc>0.9,1);
        if ~isempty(idx_k)
            kopt = idx_k;
            if kopt > max_synergies, kopt = max_synergies; end
        else
            [~,kopt] = max(Vc); % choose k with maximum VAF
        end
        SubjectSynergies.(sname).num_synergies(us) = kopt;
        
        % --- final best NNMF for chosen k ---
        bestV = -inf; bestW = []; bestH = [];
        for r = 1:num_iter
            try
                [Wm,Hm] = nnmf(Gref',kopt,'Options',opts);
                R = (Wm*Hm)'; 
                v = 1 - sum((Gref-R).^2,'all')/sum(Gref.^2,'all');
                if v > bestV
                    bestV = v; bestW = Wm; bestH = Hm;
                end
            catch
            end
        end
        
        % --- store results directly (NO reordering) ---
        SubjectSynergies.(sname).W{us}    = bestW;
        SubjectSynergies.(sname).H{us}    = bestH;
        SubjectSynergies.(sname).Word{us} = bestW;  % same as W
        SubjectSynergies.(sname).Hord{us} = bestH;  % same as H
        
        % --- Compute reconstruction and VAF ---
        Rf = (bestW * bestH)';   % T x M
        
        % Ensure Rf matches size of Gref (T x M)
        if ~isequal(size(Rf), size(Gref))
            Rf2 = zeros(size(Gref));
            rows = min(size(Rf,1), size(Gref,1));
            cols = min(size(Rf,2), size(Gref,2));
            Rf2(1:rows,1:cols) = Rf(1:rows,1:cols);
            Rf = Rf2;
        end
        
        denom = sum(Gref(:).^2);
        if denom == 0
            SubjectSynergies.(sname).VAF(us) = NaN;
        else
            SubjectSynergies.(sname).VAF(us) = 1 - sum((Gref - Rf).^2,'all') / denom;
        end

    end    
end

% optional: save results
save(fullfile(filepath_save,'SubjectSynergies.mat'),'SubjectSynergies','-v7.3');
toc

%%
max_synergies = 10;
SubjectSynergies = align_synergies_chain(SubjectSynergies, 0.7);

% Optional: save again
save(fullfile(filepath_save,'SubjectSynergies.mat'),'SubjectSynergies','-v7.3');

%% === Plot VAF, W and H for ALL subjects in SubjectSynergies and SAVE ===
% Load SubjectSynergies if it's not in workspace
if ~exist('SubjectSynergies','var')
    if exist(fullfile(filepath_save,'SubjectSynergies.mat'),'file')
        tmp = load(fullfile(filepath_save,'SubjectSynergies.mat'),'SubjectSynergies');
        SubjectSynergies = tmp.SubjectSynergies;
    else
        error('SubjectSynergies not found in workspace and SubjectSynergies.mat not found in filepath_save.');
    end
end

% Create output folder if not exist
if ~exist('filepath_plot_synxsubj','var') || isempty(filepath_plot_synxsubj)
    filepath_plot_synxsubj = fullfile(pwd,'plots_synxsubj');
end
if ~exist(filepath_plot_synxsubj,'dir'), mkdir(filepath_plot_synxsubj); end

subject_ids = fieldnames(SubjectSynergies);

for si = 1:numel(subject_ids)
    subject_id = subject_ids{si};
    fprintf('Processing %s...\n', subject_id);

    Ssub = SubjectSynergies.(subject_id);

    % obtain segments
    if ~isfield(Ssub,'segments') || isempty(Ssub.segments)
        warning('No segments found for subject %s', subject_id);
        continue;
    end
    segList = Ssub.segments(:)';        % row vector

    % gather k per segment, VAFcurves, Word/Hord
    K_per_seg = zeros(1,numel(segList));
    hasVAF = false(1,numel(segList));
    for ii = 1:numel(segList)
        us = segList(ii);
        if numel(Ssub.VAFcurve) >= us && ~isempty(Ssub.VAFcurve{us})
            hasVAF(ii) = true;
        end
        if numel(Ssub.num_synergies) >= us && ~isempty(Ssub.num_synergies)
            K_per_seg(ii) = Ssub.num_synergies(us);
        elseif numel(Ssub.W) >= us && ~isempty(Ssub.W{us})
            K_per_seg(ii) = size(Ssub.W{us},2);
        elseif numel(Ssub.Word) >= us && ~isempty(Ssub.Word{us})
            K_per_seg(ii) = size(Ssub.Word{us},2);
        else
            K_per_seg(ii) = 0;
        end
    end

    if all(K_per_seg==0) && ~any(hasVAF)
        warning('Subject %s has no synergies or VAF data to plot.', subject_id);
        continue;
    end

    max_k = max(K_per_seg);
    max_k = max(max_k,1);

    % ---- VAF curves ----
    figV = figure('Name',sprintf('%s - VAF Curves',subject_id),'Color','w','NumberTitle','off');
    axV = gca; hold(axV,'on');
    cols = parula(numel(segList));
    plotted = false;
    for ii = 1:numel(segList)
        us = segList(ii);
        if numel(Ssub.VAFcurve) >= us && ~isempty(Ssub.VAFcurve{us})
            v = Ssub.VAFcurve{us};
            plot(1:numel(v), v, '-o', 'LineWidth',1.5, 'Color', cols(ii,:), 'DisplayName',sprintf('Seg %d',us));
            km = NaN;
            if numel(Ssub.num_synergies) >= us, km = Ssub.num_synergies(us); end
            if ~isnan(km) && km>0 && km<=numel(v)
                plot(km, v(km), 'ro', 'MarkerSize',8, 'LineWidth',1.5, 'HandleVisibility','off');
            end
            plotted = true;
        end
    end
    if ~plotted
        text(0.5,0.5,'No VAF data available','HorizontalAlignment','center','Parent',axV);
    end
    xlabel('Number of Synergies','FontWeight','bold');
    ylabel('VAF','FontWeight','bold');
    title(sprintf('%s: VAF Curves', subject_id),'Interpreter','none','FontWeight','bold');
    legend('Location','eastoutside'); grid on; box off; hold(axV,'off');
    savefig(figV, fullfile(filepath_plot_synxsubj, sprintf('%s_VAFcurve.fig', subject_id)));
    close(figV);

    % ---- W (weights) ----
    figW = figure('Name',sprintf('%s - W',subject_id),'Color','w','NumberTitle','off');
    nSeg = numel(segList);
    tlW = tiledlayout(max(1,nSeg), max_k, 'Padding','compact','TileSpacing','compact');
    sgtitle(tlW, sprintf('%s — W', subject_id), 'Interpreter','none', 'FontWeight','bold');

    % global W ylim
    globalWmax = 0;
    for ii=1:nSeg
        seg = segList(ii);
        if numel(Ssub.Word) >= seg && ~isempty(Ssub.Word{seg})
            globalWmax = max(globalWmax, nanmax(Ssub.Word{seg}(:)));
        elseif numel(Ssub.W) >= seg && ~isempty(Ssub.W{seg})
            globalWmax = max(globalWmax, nanmax(Ssub.W{seg}(:)));
        end
    end
    if globalWmax==0, globalWmax = 1; end
    W_ylim = [0, globalWmax*1.1];
    colors = lines(max_k);

    for row = 1:nSeg
        seg = segList(row);
        Wmat = [];
        if numel(Ssub.Word) >= seg && ~isempty(Ssub.Word{seg})
            Wmat = Ssub.Word{seg};          
        elseif numel(Ssub.W) >= seg && ~isempty(Ssub.W{seg})
            Wmat = Ssub.W{seg};
        end
        Kw = 0; if ~isempty(Wmat), Kw = size(Wmat,2); end
        for syn = 1:max_k
            ax = nexttile(tlW, (row-1)*max_k + syn);
            if syn <= Kw && ~all(isnan(Wmat(:,syn)))
                bar(ax, Wmat(:,syn), 'FaceColor', colors(syn,:), 'EdgeColor','none');
                ylim(ax, W_ylim);
                ax.XTick = [];
                ax.YTick = [];
            else
                axis(ax,'off');
                text(0.5,0.5,'missing','Units','normalized','HorizontalAlignment','center','VerticalAlignment','middle',...
                    'FontAngle','italic','FontSize',9,'Color',[0.4 0.4 0.4]);
            end
            if row==1, title(ax, sprintf('Syn %d', syn)); end
        end
        annotation(figW,'textbox',[0.005, 1 - (row-0.5)/nSeg, 0.03, 1/nSeg], ...
            'String',sprintf('Seg %d',seg),'FitBoxToText','on','EdgeColor','none','FontSize',10,'FontWeight','bold',...
            'HorizontalAlignment','left','VerticalAlignment','middle');
    end
    savefig(figW, fullfile(filepath_plot_synxsubj, sprintf('%s_W.fig', subject_id)));
    close(figW);

    % ---- H (activations) ----
    figH = figure('Name',sprintf('%s - H',subject_id),'Color','w','NumberTitle','off');
    tlH = tiledlayout(max(1,nSeg), max_k, 'Padding','compact','TileSpacing','compact');
    sgtitle(tlH, sprintf('%s — H', subject_id), 'Interpreter','none', 'FontWeight','bold');

    globalHmax = 0; maxHlen = 0;
    for ii=1:nSeg
        seg = segList(ii);
        if numel(Ssub.Hord) >= seg && ~isempty(Ssub.Hord{seg})
            Hc = Ssub.Hord{seg};
            globalHmax = max(globalHmax, nanmax(Hc(:)));
            maxHlen = max(maxHlen, size(Hc,2));
        elseif numel(Ssub.H) >= seg && ~isempty(Ssub.H{seg})
            Hc = Ssub.H{seg};
            globalHmax = max(globalHmax, nanmax(Hc(:)));
            maxHlen = max(maxHlen, size(Hc,2));
        end
    end
    if globalHmax==0, globalHmax = 1; end
    if maxHlen==0, maxHlen = 1; end
    H_ylim = [0, globalHmax*1.1];

    for row = 1:nSeg
        seg = segList(row);
        Hmat = [];
        if numel(Ssub.Hord) >= seg && ~isempty(Ssub.Hord{seg})
            Hmat = Ssub.Hord{seg};   
        elseif numel(Ssub.H) >= seg && ~isempty(Ssub.H{seg})
            Hmat = Ssub.H{seg};
        end
        Kh = 0; if ~isempty(Hmat), Kh = size(Hmat,1); end
        for syn = 1:max_k
            ax = nexttile(tlH, (row-1)*max_k + syn);
            if syn <= Kh && ~all(isnan(Hmat(syn,:)))
                hrow = nan(1, maxHlen);
                len = size(Hmat,2);
                hrow(1:len) = Hmat(syn,:);
                plot(ax, 1:maxHlen, hrow, '-', 'LineWidth',1.5, 'Color', colors(syn,:));
                ylim(ax, H_ylim);
                xlim(ax, [1 maxHlen]);
                ax.XTick = [];
            else
                axis(ax,'off');
                text(0.5,0.5,'missing','Units','normalized','HorizontalAlignment','center','VerticalAlignment','middle',...
                    'FontAngle','italic','FontSize',9,'Color',[0.4 0.4 0.4]);
            end
            if row==1, title(ax, sprintf('Syn %d', syn)); end
        end
        annotation(figH,'textbox',[0.005, 1 - (row-0.5)/nSeg, 0.03, 1/nSeg], ...
            'String',sprintf('Seg %d',seg),'FitBoxToText','on','EdgeColor','none','FontSize',10,'FontWeight','bold',...
            'HorizontalAlignment','left','VerticalAlignment','middle');
    end
    savefig(figH, fullfile(filepath_plot_synxsubj, sprintf('%s_H.fig', subject_id)));
    close(figH);
end 

%%  -------------------- FUNCTION --------------------
function SubjectSynergies = align_synergies_chain(SubjectSynergies, alpha)
    % Align synergies across segments using chain alignment (Hungarian).
    % alpha = weight between W (spatial) and H (temporal) similarity (0..1).
    % If alpha is missing or outside [0 1], default alpha=0.7 is used.
    if nargin < 2 || isempty(alpha) || ~isnumeric(alpha) || alpha < 0 || alpha > 1
        alpha = 0.7;
    end

    subjects = fieldnames(SubjectSynergies);
    for i = 1:numel(subjects)
        sname = subjects{i};
        if ~isfield(SubjectSynergies.(sname), 'Word'), continue; end

        Word_cells = SubjectSynergies.(sname).Word;
        Hord_cells = SubjectSynergies.(sname).Hord;
        seg_ids    = SubjectSynergies.(sname).segments;
        if isempty(seg_ids), continue; end
        seg_ids = seg_ids(:)';

        trialName = upper(SubjectSynergies.(sname).trial_name);

        % --- Choose lead segment (first for RAMPDOWN/other, last for RAMPUP) ---
        if contains(trialName, 'RAMPDOWN')
            lead_seg = seg_ids(1);
        elseif contains(trialName, 'RAMPUP')
            lead_seg = seg_ids(end);
        else
            lead_seg = seg_ids(1);
        end

        leadW = Word_cells{lead_seg};
        leadH = Hord_cells{lead_seg};
        if isempty(leadW) || isempty(leadH), continue; end

        % --- Order leader by H peak (temporal) ---
        [~,pks] = max(leadH, [], 2);
        [~,ord] = sort(pks);
        leadW = leadW(:, ord);
        leadH = leadH(ord, :);
        Word_cells{lead_seg} = leadW;
        Hord_cells{lead_seg} = leadH;

        n_lead = size(leadW, 2);
        if n_lead == 0, continue; end

        % canonical sizes
        T_lead = size(leadH, 2);   % target time length for all H
        M = size(leadW, 1);        % number of muscles (rows in W)

        % --- Normalize shapes: pad/truncate W columns, pad/truncate H rows,
        %     and resample H time-dimension to T_lead ---
        for us = seg_ids
            W = Word_cells{us};
            H = Hord_cells{us};
            if isempty(W) || isempty(H), continue; end

            % ensure W has M rows
            if size(W,1) ~= M
                W2 = zeros(M, size(W,2));
                rows = min(M, size(W,1));
                W2(1:rows, :) = W(1:rows, :);
                W = W2;
            end

            % pad / truncate columns of W to n_lead
            nCols = size(W,2);
            if nCols < n_lead
                Wpad = zeros(M, n_lead);
                Wpad(:, 1:nCols) = W;
            else
                Wpad = W(:, 1:n_lead);
            end

            % pad / truncate rows of H to n_lead
            H_rows = size(H,1);
            if H_rows < n_lead
                Hpad = zeros(n_lead, size(H,2));
                Hpad(1:H_rows, :) = H;
                H = Hpad;
            elseif H_rows > n_lead
                H = H(1:n_lead, :);
            end

            % resample time dimension of H to T_lead if needed
            Tseg = size(H,2);
            if Tseg ~= T_lead && Tseg > 1 && T_lead > 1
                x_old = linspace(1, Tseg, Tseg);
                x_new = linspace(1, Tseg, T_lead);
                Hres = zeros(size(H,1), T_lead);
                for r = 1:size(H,1)
                    Hres(r, :) = interp1(x_old, H(r, :), x_new, 'linear', 0);
                end
                H = Hres;
            elseif Tseg == 1 && T_lead > 1
                % replicate single-column pattern
                H = repmat(H, 1, T_lead);
            end

            Word_cells{us} = Wpad;
            Hord_cells{us} = H;
        end

        alignedW = Word_cells;
        alignedH = Hord_cells;

        % --- Choose chain order: propagate from lead outwards ---
        if contains(trialName, 'RAMPUP')
            chain_order = flip(seg_ids); % start from lead (last) to first
        else
            chain_order = seg_ids;       % start from lead (first) to last
        end

        prevW = leadW;
        prevH = leadH;

        % --- Align each segment to previous (chain propagation) ---
        for us = chain_order
            if us == lead_seg, continue; end
            W = Word_cells{us};
            H = Hord_cells{us};
            if isempty(W) || isempty(H), continue; end

            nC = n_lead;
            S = zeros(nC, nC);
            for k1 = 1:nC
                vW1 = prevW(:, k1);
                vH1 = prevH(k1, :)';
                for k2 = 1:nC
                    vW2 = W(:, k2);
                    vH2 = H(k2, :)';

                    % --- spatial similarity (W) ---
                    if all(vW1==0) && all(vW2==0)
                        cW = 1;
                    elseif all(vW1==0) || all(vW2==0)
                        cW = 0;
                    else
                        if std(vW1) == 0 || std(vW2) == 0
                            denom = norm(vW1) * norm(vW2);
                            cW = 0;
                            if denom > 0
                                cW = (vW1' * vW2) / denom; % cosine
                            end
                        else
                            cW = corr(vW1, vW2, 'rows', 'complete');
                            if isnan(cW), cW = 0; end
                        end
                    end

                    % --- temporal similarity (H) ---
                    if all(vH1==0) && all(vH2==0)
                        cH = 1;
                    elseif all(vH1==0) || all(vH2==0)
                        cH = 0;
                    else
                        if std(vH1) == 0 || std(vH2) == 0
                            denom = norm(vH1) * norm(vH2);
                            cH = 0;
                            if denom > 0
                                cH = (vH1' * vH2) / denom;
                            end
                        else
                            cH = corr(vH1, vH2, 'rows', 'complete');
                            if isnan(cH), cH = 0; end
                        end
                    end

                    S(k1, k2) = alpha * cW + (1 - alpha) * cH;
                end
            end

            % handle non-finite entries
            S(~isfinite(S)) = -inf;

            % Hungarian assignment (minimize cost = -similarity) (Zhao et al., 2019)
            cost = -S;
            assign = munkres(cost);

            % fallback greedy assignment if munkres didn't return expected shape
            if isempty(assign) || numel(assign) ~= nC
                assign = zeros(1, nC);
                Scopy = S;
                for r = 1:nC
                    [val, idx] = max(Scopy(:));
                    if val == -inf, break; end
                    [r0, c0] = ind2sub(size(Scopy), idx);
                    assign(r0) = c0;
                    Scopy(r0, :) = -inf;
                    Scopy(:, c0) = -inf;
                end
            end

            % reorder W and H according to assignment
            Wnew = zeros(size(W));
            Hnew = zeros(size(H));
            for k1 = 1:nC
                k2 = assign(k1);
                if k2 > 0 && k2 <= nC
                    Wnew(:, k1) = W(:, k2);
                    Hnew(k1, :)  = H(k2, :);
                else
                    % leave zeros if no assignment
                end
            end

            alignedW{us} = Wnew;
            alignedH{us} = Hnew;

            % propagate
            prevW = Wnew;
            prevH = Hnew;
        end

        % Save aligned results back
        SubjectSynergies.(sname).Word = alignedW;
        SubjectSynergies.(sname).Hord = alignedH;
    end
end