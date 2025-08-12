clearvars; clc;

% load simulation data
raw = load('cylinder_nektar_wake.mat');
X_star = raw.X_star;
t_star = raw.t;
U_star = raw.U_star;
P_star = raw.p_star;

N = size(X_star,1);            
Tt = numel(t_star);           

% make full coordinate and field vectors
x_full  = repmat(X_star(:,1), Tt, 1);
y_full  = repmat(X_star(:,2), Tt, 1);
t_full  = reshape(repmat(t_star', N, 1), N*Tt, 1);
u_full  = reshape(U_star(:,1,:), N*Tt, 1);
v_full  = reshape(U_star(:,2,:), N*Tt, 1);
p_full  = reshape(P_star,       N*Tt, 1);

% split into train and test before normalization
totalPoints = numel(x_full);
splitIdx = round(0.7 * totalPoints);
rng(0);
idxAll = randperm(totalPoints);

trainIdx = idxAll(1:splitIdx);
testIdx  = idxAll(splitIdx+1:end);

% training data
x_train = x_full(trainIdx);
y_train = y_full(trainIdx);
t_train = t_full(trainIdx);
u_train = u_full(trainIdx);
v_train = v_full(trainIdx);
p_train = p_full(trainIdx);

% test data
x_test = x_full(testIdx);
y_test = y_full(testIdx);
t_test = t_full(testIdx);
u_test = u_full(testIdx);
v_test = v_full(testIdx);
p_test = p_full(testIdx);

% find normalization values from training data
mu_in  = mean([x_train,y_train,t_train], 1);
s_in   = std( [x_train,y_train,t_train], 1);
mu_out = mean([u_train,v_train,p_train], 1);
s_out  = std( [u_train,v_train,p_train], 1);

inputNorm.mean = mu_in;
inputNorm.std  = s_in;
outputNorm.mean = mu_out;
outputNorm.std  = s_out;

% normalize training data
xN_train = (x_train - mu_in(1)) / s_in(1);
yN_train = (y_train - mu_in(2)) / s_in(2);
tN_train = (t_train - mu_in(3)) / s_in(3);
uN_train = (u_train - mu_out(1)) / s_out(1);
vN_train = (v_train - mu_out(2)) / s_out(2);
pN_train = (p_train - mu_out(3)) / s_out(3);

% keep copy of normalized training inputs for collocation range
trainInputsNorm = [xN_train, yN_train, tN_train];

% normalize test data with training values
xN_test = (x_test - mu_in(1)) / s_in(1);
yN_test = (y_test - mu_in(2)) / s_in(2);
tN_test = (t_test - mu_in(3)) / s_in(3);
uN_test = (u_test - mu_out(1)) / s_out(1);
vN_test = (v_test - mu_out(2)) / s_out(2);
pN_test = (p_test - mu_out(3)) / s_out(3);

% choose collocation points inside training range
Nf = 20000;
minTrain = min(trainInputsNorm, [], 1);
maxTrain = max(trainInputsNorm, [], 1);
Xf_pool = minTrain + (maxTrain - minTrain) .* rand(Nf, 3);

% make the neural network
layers = [3, 80, 80, 80, 80, 80, 3];
lgraph = layerGraph(featureInputLayer(3,'Normalization','none','Name','in'));
for i = 2:length(layers)-1
    fcName = sprintf('fc%d', i-1);
    tanhName = sprintf('tanh%d', i-1);
    lgraph = addLayers(lgraph, [
        fullyConnectedLayer(layers(i),'Name',fcName)
        tanhLayer('Name',tanhName)]);
    if i == 2
        lgraph = connectLayers(lgraph,'in',fcName);
    else
        prevTanh = sprintf('tanh%d', i-2);
        lgraph = connectLayers(lgraph, prevTanh, fcName);
    end
end
lgraph = addLayers(lgraph, fullyConnectedLayer(3,'Name','out'));
lastTanh = sprintf('tanh%d', length(layers)-2);
lgraph = connectLayers(lgraph, lastTanh, 'out');
net = dlnetwork(lgraph);

% starting viscosity
viscosity = dlarray(0.01, 'CB');

% training with Adam
maxEpochs = 30;
miniBatchSize = 64;
learnRate = 1e-3;
trAvg = []; trAvgSq = [];
iteration = 0;

for epoch = 1:maxEpochs
    idx = randperm(splitIdx);
    xN_train = xN_train(idx); yN_train = yN_train(idx); tN_train = tN_train(idx);
    uN_train = uN_train(idx); vN_train = vN_train(idx); pN_train = pN_train(idx);
    
    sub = randperm(Nf, 1000);
    Xf_epoch = Xf_pool(sub,:);
    
    for i = 1:miniBatchSize:splitIdx
        iteration = iteration + 1;
        ib = i:min(i+miniBatchSize-1, splitIdx);
        
        dlX = dlarray([xN_train(ib)'; yN_train(ib)'; tN_train(ib)'], 'CB');
        dlU = dlarray(uN_train(ib)', 'CB');
        dlV = dlarray(vN_train(ib)', 'CB');
        dlP = dlarray(pN_train(ib)', 'CB');
        
        cf = randperm(1000, numel(ib));
        dlXf = dlarray(Xf_epoch(cf,:)', 'CB');
        
        if canUseGPU
            dlX = gpuArray(dlX); dlU = gpuArray(dlU);
            dlV = gpuArray(dlV); dlP = gpuArray(dlP);
            dlXf = gpuArray(dlXf);
        end
        
        [lossData, lossPhys, ~, grads, ~, ~, ~] = ...
            dlfeval(@modelGradients, net, viscosity, dlX, dlU, dlV, dlP, dlXf, inputNorm, outputNorm);
        
        [net, trAvg, trAvgSq] = adamupdate(net, grads.net, trAvg, trAvgSq, iteration, learnRate);
        viscosity = viscosity - learnRate * grads.viscosity;
        
        if mod(iteration, 100) == 0
            nu_val = extractdata(viscosity);
            fprintf("Epoch %d, Iter %d - Loss_data=%.3e, Loss_phys=%.3e, nu=%.4f\n", ...
                epoch, iteration, lossData, lossPhys, nu_val);
        end
    end
end
fprintf("Adam training complete. Starting L-BFGS refinement.\n");

% L-BFGS using training data
params0.net = net;
params0.viscosity = viscosity;
theta0 = double(gather(packNetworkAndViscosity(net, viscosity)));

opts = optimoptions('fminunc',...
    'Algorithm','quasi-newton',...
    'SpecifyObjectiveGradient',true,...
    'MaxIterations',30,...
    'Display','iter');

theta_opt = fminunc(@(theta)objectiveLBFGS(theta, params0, inputNorm, outputNorm, ...
                   xN_train, yN_train, tN_train, uN_train, vN_train, pN_train, Xf_pool), ...
                   theta0, opts);

[net, viscosity] = unpackTheta(theta_opt, params0);
save('trained_PINN_LBFGS.mat', ...
     'net', 'viscosity', 'inputNorm', 'outputNorm', ...
     'trainIdx', 'testIdx', 'X_star', 't_star');

% test evaluation
dlTestInput = dlarray([xN_test'; yN_test'; tN_test'], 'CB');
if canUseGPU
    dlTestInput = gpuArray(dlTestInput);
end
dlOutput = forward(net, dlTestInput);

uPred = extractdata(dlOutput(1,:))' * s_out(1) + mu_out(1);
vPred = extractdata(dlOutput(2,:))' * s_out(2) + mu_out(2);
pPred = extractdata(dlOutput(3,:))' * s_out(3) + mu_out(3);

uTrue = u_test;
vTrue = v_test;
pTrue = p_test;
R2u = 1 - sum((uTrue - uPred).^2) / sum((uTrue - mean(uTrue)).^2);
R2v = 1 - sum((vTrue - vPred).^2) / sum((vTrue - mean(vTrue)).^2);
R2p = 1 - sum((pTrue - pPred).^2) / sum((pTrue - mean(pTrue)).^2);
fprintf("Test R² (u,v,p): %.4f, %.4f, %.4f\n", R2u, R2v, R2p);

finalNu = extractdata(viscosity);
fprintf("Final learned viscosity (nu): %.5f\n", finalNu);

% function for gradients
function [lossData, lossPhys, resNorms, grads, uN_pred, vN_pred, pN_pred] = ...
    modelGradients(net, viscosity, dlX, dlU, dlV, dlP, dlXf, inN, outN)
    Y    = forward(net, dlX);
    uN_pred   = Y(1,:); 
    vN_pred   = Y(2,:); 
    pN_pred   = Y(3,:);
    lossData = mse(uN_pred, dlU) + mse(vN_pred, dlV) + 0.01*mse(pN_pred, dlP);

    Yf   = forward(net, dlXf);
    uNf  = Yf(1,:); vNf  = Yf(2,:); pNf  = Yf(3,:);
    muO  = outN.mean; sO = outN.std;
    muI  = inN.mean;  sI = inN.std;
    uf   = uNf * sO(1) + muO(1);
    vf   = vNf * sO(2) + muO(2);

    gradU = dlgradient(sum(uNf,"all"), dlXf, 'EnableHigherDerivatives', true);
    u_xN = gradU(1,:);
    u_yN = gradU(2,:);
    u_tN = gradU(3,:);
    u_x = u_xN * (sO(1)/sI(1));
    u_y = u_yN * (sO(1)/sI(2));
    u_t = u_tN * (sO(1)/sI(3));

    gradV = dlgradient(sum(vNf,"all"), dlXf, 'EnableHigherDerivatives', true);
    v_xN = gradV(1,:);
    v_yN = gradV(2,:);
    v_tN = gradV(3,:);
    v_x = v_xN * (sO(2)/sI(1));
    v_y = v_yN * (sO(2)/sI(2));
    v_t = v_tN * (sO(2)/sI(3));

    gradP = dlgradient(sum(pNf,"all"), dlXf, 'EnableHigherDerivatives', true);
    p_x = gradP(1,:) * (sO(3)/sI(1));
    p_y = gradP(2,:) * (sO(3)/sI(2));

    gradU_x = dlgradient(sum(u_xN,"all"), dlXf, 'EnableHigherDerivatives', true);
    u_xx     = gradU_x(1,:) * (sO(1)/(sI(1)^2));

    gradU_y = dlgradient(sum(u_yN,"all"), dlXf, 'EnableHigherDerivatives', true);
    u_yy     = gradU_y(2,:) * (sO(1)/(sI(2)^2));

    gradV_x = dlgradient(sum(v_xN,"all"), dlXf, 'EnableHigherDerivatives', true);
    v_xx     = gradV_x(1,:) * (sO(2)/(sI(1)^2));

    gradV_y = dlgradient(sum(v_yN,"all"), dlXf, 'EnableHigherDerivatives', true);
    v_yy     = gradV_y(2,:) * (sO(2)/(sI(2)^2));

    nu    = viscosity;
    fu    = u_t + uf .* u_x + vf .* u_y + p_x - nu .* (u_xx + u_yy);
    fv    = v_t + uf .* v_x + vf .* v_y + p_y - nu .* (v_xx + v_yy);
    fc    = u_x + v_y;
    lossPhys = mean(fu.^2 + fv.^2 + fc.^2, 'all');
    resNorms = [sqrt(mean(fu.^2)), sqrt(mean(fv.^2)), sqrt(mean(fc.^2))];

    lossTotal = lossData + 0.1 * lossPhys;
    [gradsNet, gradsVisc] = dlgradient(lossTotal, net.Learnables, viscosity);
    grads.net = gradsNet;
    grads.viscosity = gradsVisc;
end

% helper to pack parameters
function theta = packNetworkAndViscosity(net, viscosity)
    values = net.Learnables.Value;
    flatParams = cellfun(@(x) extractdata(x(:)), values, 'UniformOutput', false);
    flatParams = vertcat(flatParams{:});
    theta = [flatParams; extractdata(viscosity)];
end

% helper to unpack parameters
function [net, viscosity] = unpackTheta(theta, params0)
    layers = [3,80,80,80,80,80,3];
    lgraph = layerGraph(featureInputLayer(3,'Normalization','none','Name','in'));
    for i = 2:length(layers)-1
        fcName = sprintf('fc%d',i-1); tanhName = sprintf('tanh%d',i-1);
        lgraph = addLayers(lgraph, [fullyConnectedLayer(layers(i),'Name',fcName); tanhLayer('Name',tanhName)]);
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

    learnables = net.Learnables;
    idx = 1;
    for i = 1:height(learnables)
        sz = size(learnables.Value{i});
        n = prod(sz);
        newVal = reshape(theta(idx:idx+n-1), sz);
        learnables.Value{i} = dlarray(newVal);
        idx = idx + n;
    end
    net = dlnetwork(net.Layers);
    net.Learnables = learnables;

    viscosity = dlarray(theta(end), 'CB');
end

% objective for lbfgs
function [loss, grad] = objectiveLBFGS(theta, params0, inN, outN, x, y, t, u, v, p, Xf)
    persistent callNum
    if isempty(callNum), callNum = 1; else callNum = callNum + 1; end

    [net, viscosity] = unpackTheta(theta, params0);
    dlX  = dlarray([x'; y'; t'], 'CB');
    dlU  = dlarray(u', 'CB');
    dlV  = dlarray(v', 'CB');
    dlP  = dlarray(p', 'CB');
    dlXf = dlarray(Xf', 'CB');
    [lossData, lossPhys, resNorms, grads, ~, ~, ~] = ...
        dlfeval(@modelGradients, net, viscosity, dlX, dlU, dlV, dlP, dlXf, inN, outN);
    lossTotal = lossData + 0.1 * lossPhys;

    flatGrads = cellfun(@(x) extractdata(x(:)), grads.net.Value, 'UniformOutput', false);
    flatGrads = vertcat(flatGrads{:});
    grad = [flatGrads; extractdata(grads.viscosity)];

    loss = double(gather(extractdata(lossTotal)));
    grad = double(gather(grad));
    fprintf("L-BFGS Call %d - Loss=%.4e, Loss_data=%.4e, Loss_phys=%.4e, nu=%.5f\n", ...
            callNum, loss, double(lossData), double(lossPhys), extractdata(viscosity));
end

