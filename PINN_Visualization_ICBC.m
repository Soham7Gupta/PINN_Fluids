
clear; clc; close all;

%% --- Paths (edit if needed) ---
rawPath     = '/Users/sam/Desktop/DataPinns/cylinder_nektar_wake.mat';
trainedPath = 'trained_PINN_LBFGS_withICBC_NEW.mat';

fprintf('Loading data and trained model...\n');
raw = load(rawPath);
trained = load(trainedPath);

if ~isfield(trained,'net'), error('trained.net not found in trained file.'); end
if ~isfield(trained,'testIdx'), error('trained.testIdx not found. Visualization requires saved test indices.'); end

net = trained.net;
testIdx = double(trained.testIdx(:));   % flattened indices (points across snapshots)

% Accept either (mu,sigma) or (mean,std) naming
if isfield(trained,'inputNorm')
    inN = trained.inputNorm;
    if isfield(inN,'mu') && isfield(inN,'sigma'), mu_in = inN.mu; s_in = inN.sigma;
    elseif isfield(inN,'mean') && isfield(inN,'std'), mu_in = inN.mean; s_in = inN.std;
    else error('Unrecognized trained.inputNorm structure'); end
else
    mu_in = zeros(1,3); s_in = ones(1,3);
end

if isfield(trained,'outputNorm')
    outN = trained.outputNorm;
    if isfield(outN,'mu') && isfield(outN,'sigma'), mu_out = outN.mu; s_out = outN.sigma;
    elseif isfield(outN,'mean') && isfield(outN,'std'), mu_out = outN.mean; s_out = outN.std;
    else error('Unrecognized trained.outputNorm structure'); end
else
    mu_out = zeros(1,3); s_out = ones(1,3);
end

%% --- Spatial & temporal layout ---
X_star = double(raw.X_star);   % [N x 2]
t_star = raw.t(:);            % [T x 1]
N = size(X_star,1);
T = numel(t_star);

%% --- Snapshot selection mode ---
% Set selectionMode:
%   'A' -> random test snapshot (recommended)
%   'D' -> densest snapshot (original behavior)
selectionMode = 'A';   % change to 'D' to use the densest snapshot instead

% If you want reproducible random picks (useful for debugging), replace rng('shuffle')
% with rng(0) below.

% Determine which snapshots contain test points
test_snap_ids = unique( ceil(testIdx ./ N) );
if isempty(test_snap_ids)
    error('No test snapshots found in testIdx.');
end

switch upper(selectionMode)
    case 'A'  % Random test snapshot
        rng('shuffle');                    % for non-reproducible random picks
        % rng(0);                         % uncomment for reproducible picks
        snap = test_snap_ids( randi(numel(test_snap_ids)) );
        snapTime = t_star(snap);
        fprintf('Randomly picked snapshot %d (t = %.6f) for visualization.\n', snap, snapTime);
    case 'D'  % Densest snapshot (original behavior)
        % count test points per snapshot and pick the snapshot with most test points
        count_per_snap = zeros(T,1);
        snap_of_each_test = ceil(testIdx ./ N);
        for s = 1:T
            count_per_snap(s) = sum(snap_of_each_test == s);
        end
        [~, snap] = max(count_per_snap);
        snapTime = t_star(snap);
        fprintf('Choosing densest snapshot %d (t = %.6f) with %d test points for visualization.\n', snap, snapTime, count_per_snap(snap));
    otherwise
        error('Unknown selectionMode "%s". Use ''A'' or ''D''.', selectionMode);
end

%% --- Build the list of test-local point indices (1..N) for that snapshot ---
snap_inds_flat = (snap-1)*N + (1:N);        % flattened indices belonging to this snapshot
test_in_snap_flat = intersect(testIdx, snap_inds_flat);
if isempty(test_in_snap_flat)
    error('Selected snapshot contains no test points (unexpected).');
end
local_idxs = mod(test_in_snap_flat-1, N) + 1;   % convert to local 1..N

% Coordinates and true fields for those test points
x_all = X_star(:,1); y_all = X_star(:,2);
x_t = x_all(local_idxs);
y_t = y_all(local_idxs);
u_snap = double(squeeze(raw.U_star(:,1,snap)));
v_snap = double(squeeze(raw.U_star(:,2,snap)));
p_snap = double(raw.p_star(:,snap));
u_t = u_snap(local_idxs);
v_t = v_snap(local_idxs);
p_t = p_snap(local_idxs);

%% --- Normalize test inputs using training stats (no leakage) ---
xN = (x_t - mu_in(1)) ./ s_in(1);
yN = (y_t - mu_in(2)) ./ s_in(2);
tN = (snapTime - mu_in(3)) ./ s_in(3);

% Candidate input matrix (3 x B) and variations will be tried
data_candidate = [xN'; yN'; repmat(tN,1,numel(xN))];

%% --- Try to move network to GPU if appropriate ---
useGPU = canUseGPU && (gpuDeviceCount > 0);
if useGPU
    try
        net = dlupdate(@gpuArray, net);
        fprintf('Moved network parameters to GPU.\n');
    catch ME
        warning('Could not move network to GPU: %s. Using CPU instead.', ME.message);
        useGPU = false;
    end
end

%% --- Robust forward pass: try a few label/layout combos ---
attempts = {
    struct('mat', data_candidate, 'labels','CB'),    % common: C x B
    struct('mat', data_candidate, 'labels','BC'),    % swapped labels
    struct('mat', data_candidate.', 'labels','CB'),  % transposed inputs
    struct('mat', data_candidate.', 'labels','BC')
};
success = false; tried = {};
for k = 1:numel(attempts)
    choice = attempts{k};
    try
        dlX = dlarray(choice.mat, choice.labels);
        if useGPU, dlX = gpuArray(dlX); end
        try
            Yp_try = predict(net, dlX);
        catch
            Yp_try = forward(net, dlX);
        end
        Yp_raw = gather(extractdata(Yp_try));
        sY = size(Yp_raw);
        tried{end+1} = sprintf('Attempt %d: labels=%s mat=%s -> out=%s', k, choice.labels, mat2str(size(choice.mat)), mat2str(sY));
        % accept outputs that contain 3 as a dimension and divisible by 3
        if any(sY==3) && mod(prod(sY),3)==0
            success = true;
            chosen = choice;
            Yp_selected = Yp_raw;
            break;
        end
    catch ME
        tried{end+1} = sprintf('Attempt %d failed: %s', k, ME.message);
        continue;
    end
end

if ~success
    fprintf('Tried these layouts:\n'); for i=1:numel(tried), fprintf('  %s\n', tried{i}); end
    error('Network forward failed to produce expected 3-output layout. Check network input labels used during training.');
end

%% --- Reshape network output to [B x 3] reliably and denormalize ---
if isvector(Yp_selected)
    error('Network returned a vector; likely collapsed batch dimension. Check layout.');
end

if size(Yp_selected,1) == 3 && size(Yp_selected,2) ~= 3
    Yp_mat = Yp_selected.';      % [B x 3]
elseif size(Yp_selected,2) == 3 && size(Yp_selected,1) ~= 3
    Yp_mat = Yp_selected;        % [B x 3]
else
    if mod(numel(Yp_selected),3)==0
        Yp_mat = reshape(Yp_selected, 3, []).';
    else
        error('Unexpected network output size: %s. Cannot reshape to [B x 3].', mat2str(size(Yp_selected)));
    end
end

B = size(Yp_mat,1);
if B ~= numel(xN)
    error('Number of predictions (%d) != number of test-sample points (%d). Check input layout.', B, numel(xN));
end

% normalized outputs
uN_pred = Yp_mat(:,1); vN_pred = Yp_mat(:,2); pN_pred = Yp_mat(:,3);

% denormalize (use training output stats)
u_pred = uN_pred .* s_out(1) + mu_out(1);
v_pred = vN_pred .* s_out(2) + mu_out(2);
p_pred = pN_pred .* s_out(3) + mu_out(3);

%% --- Clean NaNs (shouldn't happen often) ---
validIdx = ~isnan(u_pred) & ~isnan(v_pred) & ~isnan(p_pred) & ~isnan(u_t) & ~isnan(v_t) & ~isnan(p_t);
if ~all(validIdx)
    warning('Removing %d test points that contain NaNs before plotting.', sum(~validIdx));
    x_t = x_t(validIdx); y_t = y_t(validIdx);
    u_t = u_t(validIdx); v_t = v_t(validIdx); p_t = p_t(validIdx);
    u_pred = u_pred(validIdx); v_pred = v_pred(validIdx); p_pred = p_pred(validIdx);
    B = numel(u_t);
end

%% --- Interpolate test-sampled fields to a regular grid for images and vorticity ---
nn = 200;  % grid resolution for images
lb = min(X_star,[],1); ub = max(X_star,[],1);
xg = linspace(lb(1), ub(1), nn);
yg = linspace(lb(2), ub(2), nn);
[Xg, Yg] = meshgrid(xg, yg);

% If too few test points for interpolation, scatteredInterpolant can still work,
% but results may be coarse. We catch errors and fall back to scatter-only plotting.
try
    Fu_t = scatteredInterpolant(x_t, y_t, u_t, 'natural', 'nearest');
    Fv_t = scatteredInterpolant(x_t, y_t, v_t, 'natural', 'nearest');
    Fp_t = scatteredInterpolant(x_t, y_t, p_t, 'natural', 'nearest');
    Fu_p = scatteredInterpolant(x_t, y_t, u_pred, 'natural', 'nearest');
    Fv_p = scatteredInterpolant(x_t, y_t, v_pred, 'natural', 'nearest');
    Fp_p = scatteredInterpolant(x_t, y_t, p_pred, 'natural', 'nearest');

    U_t = Fu_t(Xg,Yg);  V_t = Fv_t(Xg,Yg);  P_t = Fp_t(Xg,Yg);
    U_p = Fu_p(Xg,Yg);  V_p = Fv_p(Xg,Yg);  P_p = Fp_p(Xg,Yg);

    Vmag_t = sqrt(U_t.^2 + V_t.^2);
    Vmag_p = sqrt(U_p.^2 + V_p.^2);
    interp_ok = true;
catch ME
    warning('Could not interpolate test-sampled fields to grid: %s. Images/vorticity will be skipped.', ME.message);
    interp_ok = false;
end

%% --- Plots: match previous layout but using only test-sampling points ---
% 1) Velocity magnitude images (true / pred) + quiver (true / pred) in a 2x2 figure
figure('Name','Velocity Magnitude (test points only)','Units','normalized','Position',[0.05 0.05 0.9 0.85]);

subplot(2,2,1);
if interp_ok
    imagesc(xg,yg,Vmag_t); axis equal tight; set(gca,'YDir','normal'); colorbar;
    title(sprintf('True speed (interp from test samples) t=%.6f', snapTime));
else
    scatter(x_t, y_t, 15, sqrt(u_t.^2+v_t.^2), 'filled'); axis equal tight; colorbar;
    title('True speed (scatter from test samples)');
end
xlabel('x'); ylabel('y');

subplot(2,2,2);
if interp_ok
    imagesc(xg,yg,Vmag_p); axis equal tight; set(gca,'YDir','normal'); colorbar;
    title('Predicted speed (interp from test samples)');
else
    scatter(x_t, y_t, 15, sqrt(u_pred.^2+v_pred.^2), 'filled'); axis equal tight; colorbar;
    title('Pred speed (scatter from test samples)');
end
xlabel('x'); ylabel('y');

% quiver spacing (plot a subset for clarity)
skip = max(1,round(nn/30));
subplot(2,2,3);
if interp_ok
    quiver(Xg(1:skip:end,1:skip:end), Yg(1:skip:end,1:skip:end), U_t(1:skip:end,1:skip:end), V_t(1:skip:end,1:skip:end), 1.5, 'k');
else
    quiver(x_t, y_t, u_t, v_t, 1.5, 'k');
end
axis equal tight; title('True velocity vectors'); xlabel('x'); ylabel('y');

subplot(2,2,4);
if interp_ok
    quiver(Xg(1:skip:end,1:skip:end), Yg(1:skip:end,1:skip:end), U_p(1:skip:end,1:skip:end), V_p(1:skip:end,1:skip:end), 1.5);
else
    quiver(x_t, y_t, u_pred, v_pred, 1.5);
end
axis equal tight; title('Predicted velocity vectors'); xlabel('x'); ylabel('y');

% 2) Pressure field figure (true / pred)
figure('Name','Pressure Field (test points only)','Units','normalized','Position',[0.2 0.2 0.6 0.5]);
subplot(1,2,1);
if interp_ok
    imagesc(xg,yg,P_t); axis equal tight; set(gca,'YDir','normal'); colorbar; title('True p (from test samples)');
else
    scatter(x_t, y_t, 15, p_t, 'filled'); axis equal tight; colorbar; title('True p (scatter test samples)');
end
xlabel('x'); ylabel('y');

subplot(1,2,2);
if interp_ok
    imagesc(xg,yg,P_p); axis equal tight; set(gca,'YDir','normal'); colorbar; title('Predicted p (from test samples)');
else
    scatter(x_t, y_t, 15, p_pred, 'filled'); axis equal tight; colorbar; title('Pred p (scatter test samples)');
end
xlabel('x'); ylabel('y');

%% --- Vorticity error (if we have interpolated fields) ---
if interp_ok
    % gradient uses (rows -> y, cols -> x): gradient(Z, y_vector, x_vector)
    [dU_dy, dU_dx] = gradient(U_t, yg, xg);
    [dV_dy, dV_dx] = gradient(V_t, yg, xg);
    omega_t = dV_dx - dU_dy;

    [dU_dy_p, dU_dx_p] = gradient(U_p, yg, xg);
    [dV_dy_p, dV_dx_p] = gradient(V_p, yg, xg);
    omega_p = dV_dx_p - dU_dy_p;

    vortErr = omega_t - omega_p;

    figure('Name','Vorticity Error (test samples interp)');
    contourf(Xg, Yg, vortErr, 30, 'LineColor','none'); axis equal tight; colorbar;
    title('Vorticity error (true - pred) from test-sample interpolation');
else
    fprintf('Skipping vorticity image: insufficient test points to interpolate.\n');
end

%% --- R² (computed on test sampling points used here) ---
eps_val = 1e-12;
SSu = sum((u_t - mean(u_t)).^2);
SSv = sum((v_t - mean(v_t)).^2);
SSp = sum((p_t - mean(p_t)).^2);
if SSu <= eps_val || SSv <= eps_val || SSp <= eps_val
    warning('Near-zero variance in true test field; R² may be meaningless.');
end

R2_u = 1 - sum((u_t - u_pred).^2) / max(SSu, eps_val);
R2_v = 1 - sum((v_t - v_pred).^2) / max(SSv, eps_val);
R2_p = 1 - sum((p_t - p_pred).^2) / max(SSp, eps_val);

fprintf('\nR2 Scores at time = %.6f (using B = %d test samples)\n', snapTime, B);
fprintf('R2 for u = %.6f\n', R2_u);
fprintf('R2 for v = %.6f\n', R2_v);
fprintf('R2 for p = %.6f\n', R2_p);

fprintf('Visualization complete. Only test sampling points were used for predictions and plots.\n');

