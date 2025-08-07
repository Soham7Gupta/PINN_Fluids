clearvars; clc;% clearing all existing variables and command window

%% Load & Preprocess Data
raw = load('cylinder_nektar_wake.mat');
% Assigning fluid simulation data (space, time, velocity, pressure) to variables for ease of use.
X_star = raw.X_star;
t_star = raw.t;
U_star = raw.U_star;
P_star = raw.p_star;

N = size(X_star,1);% Number of grid points per snapshot
Tt = numel(t_star);% Number of time snapshots in sample
% Repeating the x and y coordinates for each time step to match the full time evolution.
x  = repmat(X_star(:,1), Tt, 1);
y  = repmat(X_star(:,2), Tt, 1);
tt = reshape(repmat(t_star', N, 1), N*Tt, 1);% Repeat and reshape time steps to match each (x,y) coordinate
% Extract and reshape horizontal (u) and vertical (v) velocity components into column vectors.
u  = reshape(U_star(:,1,:), N*Tt, 1);
v  = reshape(U_star(:,2,:), N*Tt, 1);
%Flatten pressure values into a single column vector
p  = reshape(P_star, N*Tt, 1);
% Calculating mean and standard deviations
mu = mean([x,y,tt,u,v,p],1);
s  = std( [x,y,tt,u,v,p],1);
% Normalizing the data
xN = (x  - mu(1)) / s(1);
yN = (y  - mu(2)) / s(2);
tN = (tt - mu(3)) / s(3);
uN = (u  - mu(4)) / s(4);
vN = (v  - mu(5)) / s(5);
pN = (p  - mu(6)) / s(6);
% Save mean and std of input variables (x, y, t) for denormalizing inputs later.
inputNorm.mean = mu(1:3); inputNorm.std = s(1:3);
outputNorm.mean = mu(4:6); outputNorm.std = s(4:6);

%% Train-Test Split (70% train, 30% test)
totalPoints = numel(xN);
splitIdx = round(0.7 * totalPoints);
idxAll = randperm(totalPoints);

trainIdx = idxAll(1:splitIdx);
testIdx  = idxAll(splitIdx+1:end);

% Training Data
xN_train = xN(trainIdx);
yN_train = yN(trainIdx);
tN_train = tN(trainIdx);
uN_train = uN(trainIdx);
vN_train = vN(trainIdx);
pN_train = pN(trainIdx);

% (Optional) Test Data – use later for validation
xN_test = xN(testIdx);
yN_test = yN(testIdx);
tN_test = tN(testIdx);
uN_test = uN(testIdx);
vN_test = vN(testIdx);
pN_test = pN(testIdx);

%% Collocation Points
Nf = 20000;% For computing physical residual loss
% Defining lower and upper bounds
lb = [min(xN), min(yN), min(tN)];
ub = [max(xN), max(yN), max(tN)];
% Generating collocation points using sampling
Xf_pool = lb + (ub-lb).*rand(Nf,3);

%% Build Network
layers = [3,80,80,80,80,80,3];
lgraph = layerGraph(featureInputLayer(3,'Normalization','none','Name','in'));
for i = 2:length(layers)-1
    fcName = sprintf('fc%d',i-1);
    tanhName = sprintf('tanh%d',i-1);
    lgraph = addLayers(lgraph, [
        fullyConnectedLayer(layers(i),'Name',fcName)
        tanhLayer('Name',tanhName)]);%Add a fully connected layer and a tanh nonlinearity for this layer index.
    if i==2
        lgraph = connectLayers(lgraph,'in',fcName);%  Connect the input layer to the first FC layer
    else
        prevTanh = sprintf('tanh%d',i-2);
        lgraph = connectLayers(lgraph,prevTanh,fcName);% From the second hidden layer onward, connect each block to the tanh from the previous layer.
    end
end
lgraph = addLayers(lgraph, fullyConnectedLayer(3,'Name','out'));% Add the final fully connected output layer which returns 3 values: u, v, and p.
%Connect the last tanh activation to the output layer.
lastTanh = sprintf('tanh%d',length(layers)-2);
lgraph = connectLayers(lgraph,lastTanh,'out');
net = dlnetwork(lgraph);

%% Initialize viscosity as learnable parameter
viscosity = dlarray(0.01, 'CB');

%% TRAINING
% Adam Phase
maxEpochs = 15;
miniBatchSize = 64;
learnRate = 1e-3;
trAvg = []; trAvgSq = []; iteration = 0;

for epoch = 1:maxEpochs
    idx = randperm(numel(xN_train));
    xN_train = xN_train(idx); yN_train = yN_train(idx); tN_train = tN_train(idx);
    uN_train = uN_train(idx); vN_train = vN_train(idx); pN_train = pN_train(idx);
    % randomly selecting 1000 colloaction points from large pool
    sub = randperm(Nf,1000);
    Xf = Xf_pool(sub,:);

    for i = 1:miniBatchSize:numel(xN_train)
        iteration = iteration + 1;
        ib = i:min(i+miniBatchSize-1,numel(xN_train));
        % Converting mini-batch input and output to dlarray for auto diff
        dlX = dlarray([xN_train(ib)'; yN_train(ib)'; tN_train(ib)'], 'CB');
        dlU = dlarray(uN_train(ib)', 'CB');
        dlV = dlarray(vN_train(ib)', 'CB');
        dlP = dlarray(pN_train(ib)', 'CB');

        cf = randperm(1000, numel(ib));
        dlXf = dlarray(Xf(cf,:)', 'CB');
        % Moving data to gpu for faster training
        if canUseGPU
            dlX = gpuArray(dlX); dlU = gpuArray(dlU);
            dlV = gpuArray(dlV); dlP = gpuArray(dlP); dlXf = gpuArray(dlXf);
        end
        % Evaluating gradient and looses using auto-differentiation
        [lossData, lossPhys, ~, grads, uPred, vPred, pPred] = ...
            dlfeval(@modelGradients, net, viscosity, dlX, dlU, dlV, dlP, dlXf, inputNorm, outputNorm);

        [net, trAvg, trAvgSq] = adamupdate(net, grads.net, trAvg, trAvgSq, iteration, learnRate);
        viscosity = viscosity - learnRate * grads.viscosity;

        if mod(iteration,100)==0
            nu_val = extractdata(viscosity);
            fprintf("Epoch %d, Iter %d - LossD=%.3e, LossP=%.3e, nu=%.4f\n", ...
                epoch, iteration, lossData, lossPhys, nu_val);
        end
    end
end
%L-BFGS Phase
fprintf("Adam training complete. Starting L-BFGS refinement. \n");
% Storing current trained network for lbfgs updates.
params0.net  = net;
params0.viscosity = viscosity;
% Conerting dlnetwork weights and viscosity into a single vector
theta0 = double(gather(packNetworkAndViscosity(net, viscosity)));

% Set Lbfgs options
opts = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'SpecifyObjectiveGradient',true, ...
    'MaxIterations', 15, ...
    'Display','iter');
% Running the optimizer
theta_opt = fminunc(@(theta)objectiveLBFGS(theta, ...
    params0, inputNorm, outputNorm, xN, yN, tN, uN, vN, pN, Xf_pool), ...
    theta0, opts);
% Unpack the optimized flat parameter vector back into a structured dlnetwork and scalar dlarray for viscosity.
[net, viscosity] = unpackTheta(theta_opt, params0);

save('trained_PINN_LBFGS.mat', 'net', 'viscosity', 'inputNorm', 'outputNorm');% Saving the results
% Prepare test inputs
xTest = xN_test;
yTest = yN_test;
tTest = tN_test;

% Get predictions
dlTestInput = dlarray([xTest, yTest, tTest]', 'CB');
dlOutput = forward(net, dlTestInput);
uPred = extractdata(dlOutput(1, :))' * outputNorm.std(1) + outputNorm.mean(1);
vPred = extractdata(dlOutput(2, :))' * outputNorm.std(2) + outputNorm.mean(2);
pPred = extractdata(dlOutput(3, :))' * outputNorm.std(3) + outputNorm.mean(3);


% Compare with ground truth %1
uTrue = u(testIdx);
vTrue = v(testIdx);
pTrue = p(testIdx);


% R² scores
R2u = 1 - sum((uTrue - uPred).^2) / sum((uTrue - mean(uTrue)).^2);
R2v = 1 - sum((vTrue - vPred).^2) / sum((vTrue - mean(vTrue)).^2);
R2p = 1 - sum((pTrue - pPred).^2) / sum((pTrue - mean(pTrue)).^2);

fprintf("R² (u): %.4f | R² (v): %.4f | R² (p): %.4f\n", R2u, R2v, R2p);
fprintf("Final learned viscosity (nu): %.5f\n", nu_val);


function [lossData, lossPhys, resNorms, grads, uN, vN, pN] = ...
    modelGradients(net, viscosity, dlX, dlU, dlV, dlP, dlXf, inN, outN)
    % Forward pass
    % Predict u,v,p for supervised data pts
    Y    = forward(net, dlX);
    uN   = Y(1,:); 
    vN   = Y(2,:); 
    pN   = Y(3,:);

    % Supervised loss
    lossData = mse(uN, dlU) + mse(vN, dlV) + 0.01 * mse(pN, dlP);

    % Forward pass for colloaction points to calculate phy residual
    Yf   = forward(net, dlXf);
    uNf  = Yf(1,:); 
    vNf  = Yf(2,:); 
    pNf  = Yf(3,:);

    % De-normalize velocities
    muO  = outN.mean; sO = outN.std;
    muI  = inN.mean;  sI = inN.std;
    uf   = uNf * sO(1) + muO(1);
    vf   = vNf * sO(2) + muO(2);

    % Gradients for u
    gradU = dlgradient(sum(uNf, "all"), dlXf, 'EnableHigherDerivatives', true);
    u_x   = gradU(1,:) * (sO(1) / sI(1));
    u_y   = gradU(2,:) * (sO(1) / sI(2));
    u_t   = gradU(3,:) * (sO(1) / sI(3));

    % Gradients for v
    gradV = dlgradient(sum(vNf, "all"), dlXf, 'EnableHigherDerivatives', true);
    v_x   = gradV(1,:) * (sO(2) / sI(1));
    v_y   = gradV(2,:) * (sO(2) / sI(2));
    v_t   = gradV(3,:) * (sO(2) / sI(3));

    % Gradients for p
    gradP = dlgradient(sum(pNf, "all"), dlXf, 'EnableHigherDerivatives', true);
    p_x   = gradP(1,:) * (sO(3) / sI(1));
    p_y   = gradP(2,:) * (sO(3) / sI(2));

    % Second derivatives (Laplacian terms)
    u_xx  = dlgradient(sum(u_x, "all"), dlXf, 'EnableHigherDerivatives', true);
    u_yy  = dlgradient(sum(u_y, "all"), dlXf, 'EnableHigherDerivatives', true);
    v_xx  = dlgradient(sum(v_x, "all"), dlXf, 'EnableHigherDerivatives', true);
    v_yy  = dlgradient(sum(v_y, "all"), dlXf, 'EnableHigherDerivatives', true);

    % Residuals of Navier Stokes and continuity
    nu    = viscosity;
    fu    = u_t + uf .* u_x + vf .* u_y + p_x - nu .* (u_xx(1,:) + u_yy(2,:));
    fv    = v_t + uf .* v_x + vf .* v_y + p_y - nu .* (v_xx(1,:) + v_yy(2,:));
    fc    = u_x + v_y;

    % Physics loss 
    lossPhys = mean(fu.^2 + fv.^2 + fc.^2, 'all');
    resNorms = [sqrt(mean(fu.^2)), sqrt(mean(fv.^2)), sqrt(mean(fc.^2))];

    % Total loss and gradients
    lossTotal = lossData + 0.1 * lossPhys;
    [gradsNet, gradsVisc] = dlgradient(lossTotal, net.Learnables, viscosity);

    % Output gradients
    grads.net       = gradsNet;
    grads.viscosity = gradsVisc;
end
% Flattens all network weights and viscosity into a single vector theta.
function theta = packNetworkAndViscosity(net, viscosity)
    values = net.Learnables.Value;
    flatParams = cellfun(@(x) extractdata(x(:)), values, 'UniformOutput', false);%Extract all learnable weights
    flatParams = vertcat(flatParams{:});% Flatten each parameter
    theta = [flatParams; extractdata(viscosity)];%  Vertically concatenate all the weight vectors into one long column vector
end
function [net, viscosity] = unpackTheta(theta, params0)
    % Rebuild network architecture
    layers = [3,80,80,80,80,80,3];
    lgraph = layerGraph(featureInputLayer(3,'Normalization','none','Name','in'));

    for i = 2:length(layers)-1
        fcName = sprintf('fc%d',i-1);
        tanhName = sprintf('tanh%d',i-1);
        lgraph = addLayers(lgraph, [
            fullyConnectedLayer(layers(i), 'Name', fcName)
            tanhLayer('Name', tanhName)]);
        if i==2
            lgraph = connectLayers(lgraph,'in',fcName);
        else
            prevTanh = sprintf('tanh%d',i-2);
            lgraph = connectLayers(lgraph,prevTanh,fcName);
        end
    end
    lgraph = addLayers(lgraph, fullyConnectedLayer(3,'Name','out'));
    lastTanh = sprintf('tanh%d',length(layers)-2);
    lgraph = connectLayers(lgraph,lastTanh,'out');

    net = dlnetwork(lgraph);

    % Unpack theta into learnables
    learnables = net.Learnables;
    idx = 1;
    % load weights from theta
    for i = 1:height(learnables)
        sz = size(learnables.Value{i});
        n = prod(sz);
        newValue = reshape(theta(idx:idx+n-1), sz);
        learnables.Value{i} = dlarray(newValue);
        idx = idx + n;
    end

    % Update net with new learnables
    net = dlnetwork(net.Layers);
    net.Learnables = learnables;

    % Remaining theta is viscosity
    viscosity = dlarray(theta(end), 'CB');
end

function [loss, grad] = objectiveLBFGS(theta, params0, inN, outN, x, y, t, u, v, p, Xf)
    %Track Number of Calls
    persistent callNum
    if isempty(callNum), callNum = 1; else, callNum = callNum + 1; end

    % Unpack parameters
    [net, viscosity] = unpackTheta(theta, params0);

    % Format inputs
    dlX  = dlarray([x'; y'; t'], 'CB');
    dlU  = dlarray(u', 'CB');
    dlV  = dlarray(v', 'CB');
    dlP  = dlarray(p', 'CB');
    dlXf = dlarray(Xf', 'CB');

    % Forward pass + gradients
    [lossData, lossPhys, resNorms, grads, ~, ~, ~] = ...
        dlfeval(@modelGradients, net, viscosity, dlX, dlU, dlV, dlP, dlXf, inN, outN);

    % Total loss
    lossTotal = lossData + 0.1 * lossPhys;

    % Flatten gradients
    values = grads.net.Value;
    flatGrads = cellfun(@(x) extractdata(x(:)), values, 'UniformOutput', false);
    flatGrads = vertcat(flatGrads{:});
    grad = [flatGrads; extractdata(grads.viscosity)];

    % Convert to double
    loss = double(gather(extractdata(lossTotal)));
    grad = double(gather(grad));

    % Display current state
    fprintf("L-BFGS Call %d - Loss=%.4e, LossD=%.4e, LossP=%.4e, nu=%.5f, |fu|=%.3e, |fv|=%.3e, |fc|=%.3e\n", ...
        callNum, loss, double(lossData), double(lossPhys), extractdata(viscosity), ...
        resNorms(1), resNorms(2), resNorms(3));
end
