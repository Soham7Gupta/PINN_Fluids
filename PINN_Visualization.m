clear; clc; close all;

% Load raw simulation data and trained model
raw     = load('/Users/sam/Desktop/DataPinns/cylinder_nektar_wake.mat');
trained = load('trained_PINN_LBFGS.mat');  

fprintf("Loaded L-BFGS trained model\n");

X_star  = double(raw.X_star);    % Space coordinates
t_star  = raw.t;                 % Time values

net       = trained.net;         % Trained network
viscosity = trained.viscosity;   % Learned fluid viscosity

% Show the learned viscosity value after training
fprintf("Learned viscosity nu after L-BFGS: %.5f\n", extractdata(viscosity));

% Store normalization values for input and output
mu_in   = trained.inputNorm.mean;    
s_in    = trained.inputNorm.std;
mu_out  = trained.outputNorm.mean;   
s_out   = trained.outputNorm.std;

% Select one time snapshot to test
snap = 150;                   % Choose frame index
tval = t_star(snap);          % Get time at that frame

% Get x y coordinates and true u v p from data
x = X_star(:,1);
y = X_star(:,2);
u_true = squeeze(raw.U_star(:,1,snap));
v_true = squeeze(raw.U_star(:,2,snap));
p_true = raw.p_star(:,snap);

% Normalize x y t for prediction
xN = (x - mu_in(1)) / s_in(1);
yN = (y - mu_in(2)) / s_in(2);
tN = (tval - mu_in(3)) / s_in(3);

% Build network input
dlX = dlarray([xN'; yN'; repmat(tN,1,numel(xN))], 'CB');
if canUseGPU, dlX = gpuArray(dlX); end

% Predict u v p using the trained PINN
Yp = forward(net, dlX);
u_pred = extractdata(Yp(1,:))' * s_out(1) + mu_out(1);
v_pred = extractdata(Yp(2,:))' * s_out(2) + mu_out(2);
p_pred = extractdata(Yp(3,:))' * s_out(3) + mu_out(3);

% Create grid for interpolation
nn = 200;
lb = min(X_star); ub = max(X_star);
xg = linspace(lb(1), ub(1), nn);
yg = linspace(lb(2), ub(2), nn);
[Xg, Yg] = meshgrid(xg, yg);

% Interpolate both true and predicted fields
Fu_t = scatteredInterpolant(x,y,double(u_true),'natural','none');
Fv_t = scatteredInterpolant(x,y,double(v_true),'natural','none');
Fp_t = scatteredInterpolant(x,y,double(p_true),'natural','none');
Fu_p = scatteredInterpolant(x,y,double(u_pred),'natural','none');
Fv_p = scatteredInterpolant(x,y,double(v_pred),'natural','none');
Fp_p = scatteredInterpolant(x,y,double(p_pred),'natural','none');

U_t = Fu_t(Xg,Yg);  V_t = Fv_t(Xg,Yg);  P_t = Fp_t(Xg,Yg);
U_p = Fu_p(Xg,Yg);  V_p = Fv_p(Xg,Yg);  P_p = Fp_p(Xg,Yg);

Vmag_t = sqrt(U_t.^2 + V_t.^2);   % True velocity magnitude
Vmag_p = sqrt(U_p.^2 + V_p.^2);   % Predicted velocity magnitude

% Show true and predicted velocity magnitudes
figure('Name','L-BFGS: Velocity Magnitude');
subplot(1,2,1)
imagesc(xg,yg,Vmag_t); axis equal tight; set(gca,'YDir','normal')
title(sprintf('True velocity at t = %.3f',tval)); colorbar
xlabel('x'); ylabel('y');
subplot(1,2,2)
imagesc(xg,yg,Vmag_p); axis equal tight; set(gca,'YDir','normal')
title('Predicted velocity'); colorbar
xlabel('x'); ylabel('y');

% Show true and predicted velocity vectors
figure('Name','L-BFGS: Velocity Vectors');
subplot(1,2,1)
quiver(Xg,Yg,U_t,V_t,1.5,'k'); axis equal tight
title(sprintf('True velocity vectors at t = %.3f',tval))
xlabel('x'); ylabel('y');
subplot(1,2,2)
quiver(Xg,Yg,U_p,V_p,1.5); axis equal tight
title('Predicted velocity vectors')
xlabel('x'); ylabel('y');

% Show pressure field from true and predicted
figure('Name','L-BFGS: Pressure Field');
subplot(1,2,1)
imagesc(xg,yg,P_t); axis equal tight; set(gca,'YDir','normal')
title(sprintf('True pressure at t = %.3f',tval)); colorbar
xlabel('x'); ylabel('y');
subplot(1,2,2)
imagesc(xg,yg,P_p); axis equal tight; set(gca,'YDir','normal')
title('Predicted pressure'); colorbar
xlabel('x'); ylabel('y');

% Vorticity error using gradients
[Ux_t,Uy_t] = gradient(U_t,xg,yg);
[Vx_t,Vy_t] = gradient(V_t,xg,yg);
[Ux_p,Uy_p] = gradient(U_p,xg,yg);
[Vx_p,Vy_p] = gradient(V_p,xg,yg);

omega_t = Vx_t - Uy_t;  % True vorticity
omega_p = Vx_p - Uy_p;  % Predicted vorticity
vortErr = omega_t - omega_p;

figure('Name','L-BFGS: Vorticity Error');
contourf(Xg,Yg,vortErr,30,'LineColor','none'); axis equal tight
title('Difference in vorticity true minus predicted'); colorbar
xlabel('x'); ylabel('y');

% Compute R2 score for u v p
SSu = sum((u_true - mean(u_true)).^2);   % Total variance for u
SSv = sum((v_true - mean(v_true)).^2);   % Total variance for v
SSp = sum((p_true - mean(p_true)).^2);   % Total variance for p

R2_u = 1 - sum((u_true - u_pred).^2) / SSu;
R2_v = 1 - sum((v_true - v_pred).^2) / SSv;
R2_p = 1 - sum((p_true - p_pred).^2) / SSp;

% Print final R2 scores
fprintf('\nR2 Scores at time = %.3f\n', tval);
fprintf('R2 for u = %.4f\n', R2_u);
fprintf('R2 for v = %.4f\n', R2_v);
fprintf('R2 for p = %.4f\n', R2_p);
