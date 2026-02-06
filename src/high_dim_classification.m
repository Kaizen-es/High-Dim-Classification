%{
              High-Dimensional Classification with Fisher LDA
    
    Demonstrates why dimensionality reduction is essential when standard
    Gaussian classifiers fail due to numerical underflow in high-dimensional spaces.
    
 Datasets:
    - Wine Quality - 11 features (direct classification)
    - HAR -  561 features (requires LDA preprocessing)

%}

clear all, close all,

% PART A: WINE QUALITY DATASE


fprintf('WINE QUALITY DATASET CLASSIFICATION\n');


% DATA LOADING
data = readtable('winequality-white.csv', 'Delimiter', ';');
features = table2array(data(:, 1:11));
labels = table2array(data(:, 12));
  

% Transpose data
x = features';
label = labels';

[n, N] = size(x);
uniqueLabels = unique(label);
C = length(uniqueLabels);

fprintf('Dataset loaded:\n');
fprintf('  Samples: %d\n', N);
fprintf('  Features: %d\n', n);
fprintf('  Classes: %d (Quality scores: ', C);
fprintf('%d ', uniqueLabels);
fprintf(')\n\n');

% Parameter estimation
priors = zeros(C, 1);
mu = zeros(n, C);
Sigma = zeros(n, n, C);

for l = 1:C
    classLabel = uniqueLabels(l);
    indl = find(label == classLabel);
    Nl = length(indl);
    x_class = x(:, indl);
    
    priors(l) = Nl / N;
    mu(:, l) = mean(x_class, 2);
    Sigma(:,:, l) = cov(x_class');
    
    fprintf('Class %d: N=%d, Prior=%.4f\n', classLabel, Nl, priors(l));
end

% Regularization decision -
fprintf('\n=== REGULARIZATION DECISION ===\n');
fprintf('Features (n): %d\n', n);
fprintf('Total Samples (N): %d\n', N);
fprintf('Classes (C): %d\n', C);
fprintf('Approximate samples per class: %.0f\n', N/C);
sampleToFeatureRatio = (N/C) / n;
fprintf('Sample-to-feature ratio: %.2f\n', sampleToFeatureRatio);

if sampleToFeatureRatio > 10
    fprintf('→ Ratio > 10:1 (well-conditioned)\n');
    fprintf('→ Decision: NO regularization\n');
    fprintf('→ Rationale: Adding bias when none is needed hurts performance\n');
    applyRegularization = false;
else
    fprintf('→ Ratio < 10:1 (borderline/ill-conditioned)\n');
    fprintf('→ Decision: Apply regularization\n');
    applyRegularization = true;
    alpha = 0.05;
end


% Classification
pxgivenl = zeros(C, N);
for l = 1:C
    pxgivenl(l, :) = evalGaussian(x, mu(:,l), Sigma(:,:,l));
end

px = priors' * pxgivenl;
classPosteriors = pxgivenl .* repmat(priors, 1, N) ./ repmat(px, C, 1);

[~, decisions] = max(classPosteriors, [], 1);
decisionsLabel = uniqueLabels(decisions);

% Confusion Matrix 
ConfusionMatrix = zeros(C, C);
for d = 1:C
    for l = 1:C
        ind_dl = find(decisionsLabel == uniqueLabels(d) & label == uniqueLabels(l));
        indl = find(label == uniqueLabels(l));
        ConfusionMatrix(d, l) = length(ind_dl) / length(indl);
    end
end

fprintf('\nWINE QUALITY RESULTS\n');
fprintf('Confusion Matrix P(D=d|L=l):\n');
fprintf('       ');
for l = 1:C
    fprintf('L=%d    ', uniqueLabels(l));
end
fprintf('\n');
for d = 1:C
    fprintf('D=%d: ', uniqueLabels(d));
    fprintf(' %.4f', ConfusionMatrix(d, :));
    fprintf('\n');
end

Perror_wine = sum(decisionsLabel ~= label) / N;
fprintf('\nP(error): %.4f (%.2f%%)\n', Perror_wine, 100*Perror_wine);

fprintf('\nPer-Class Accuracy:\n');
for l = 1:C
    fprintf('  Quality %d: %.4f (%.2f%%)\n', uniqueLabels(l), ...
            ConfusionMatrix(l, l), 100*ConfusionMatrix(l, l));
end

% PCA VISUALIZATION 
muhat = mean(x, 2);
Sigmahat = cov(x');
xzm = x - muhat * ones(1, N);
[Q, D] = eig(Sigmahat);
[d, ind] = sort(diag(D), 'descend');
Q = Q(:, ind);
D = diag(d);
y = Q' * xzm;

% Plot visualization 
figure(1), clf;
subplot(1, 2, 1);
hold on;
colors = lines(C);
for l = 1:C
    indl = (label == uniqueLabels(l));
    plot(y(1, indl), y(2, indl), '.', 'Color', colors(l, :), ...
         'MarkerSize', 10, 'DisplayName', sprintf('Quality %d', uniqueLabels(l)));
end
xlabel('PC 1'), ylabel('PC 2');
title('Wine Quality: First 2 Principal Components');
legend('Location', 'best');
grid on, axis equal;

subplot(1, 2, 2);
eigenvalues = diag(D);
cumvar = cumsum(eigenvalues) / sum(eigenvalues);
plot(1:length(cumvar), 100*cumvar, 'b-o', 'LineWidth', 2);
xlabel('Number of Principal Components');
ylabel('Cumulative Variance Explained (%)');
title('Wine Quality: PCA Variance Explained');
grid on;

% PART B: HUMAN ACTIVITY RECOGNITION

fprintf('HUMAN ACTIVITY RECOGNITION DATASET\n');

% DATA LOADING 
X_train = load('Train/X_train.txt');
y_train = load('Train/y_train.txt');
X_test = load('Test/X_test.txt');
y_test = load('Test/y_test.txt');
    
features = [X_train; X_test];
labels = [y_train; y_test];
  
fprintf('  Train set: %d samples\n', size(X_train, 1));
fprintf('  Test set: %d samples\n', size(X_test, 1));
fprintf('  Combined: %d samples\n', size(features, 1));


% Transpose data 
x_original = features';
label = labels';

[n, N] = size(x_original);
uniqueLabels = unique(label);
C = length(uniqueLabels);

fprintf('Dataset loaded:\n');
fprintf('  Samples: %d\n', N);
fprintf('  Features: %d\n', n);
fprintf('  Classes: %d (Activities: 1-6)\n\n', C);

activityNames = {'Walking', 'Walking Upstairs', 'Walking Downstairs', ...
                 'Sitting', 'Standing', 'Laying'};

for l = 1:C
    Nl = length(find(label == l));
    fprintf('Activity %d (%s): N=%d, Prior=%.4f\n', l, activityNames{l}, ...
            Nl, Nl/N);
end

% Strategy decision 
fprintf('\n=== CLASSIFICATION STRATEGY ===\n');
fprintf('Original dimensionality: n = %d\n', n);
fprintf('Sample-to-feature ratio: %.2f:1 (severely ill-conditioned)\n', (N/C)/n);
fprintf('\nPROBLEM: Direct Gaussian classification fails due to:\n');
fprintf('  1. Numerical underflow: (2π)^(-561/2) ≈ 10^(-430) < realmin\n');
fprintf('  2. Covariance estimation: Need ~314,721 parameters per class\n');
fprintf('  3. Only ~1,700 samples per class available\n\n');

fprintf('SOLUTION: Dimensionality reduction BEFORE classification\n');
fprintf('Method: Multi-class Fisher LDA \n');

% Multiclass Fisher LDA 
% Reduce from 561D to (C-1)D = 5D space that maximizes class separation


% Estimate overall mean 
mu_overall = mean(x_original, 2);

% Compute within-class and between-class scatter 
% Extended to multi-class case 
Sw = zeros(n, n);  % Within-class scatter
Sb = zeros(n, n);  % Between-class scatter

mu_class = zeros(n, C);
for l = 1:C
    indl = find(label == l);
    Nl = length(indl);
    x_class = x_original(:, indl);
    mu_class(:, l) = mean(x_class, 2);
    
    % Within-class scatter
    Sw = Sw + cov(x_class');
    
    % Between-class scatter
    diff = mu_class(:, l) - mu_overall;
    Sb = Sb + Nl * (diff * diff');
end

% Add regularization to Sw for numerical stability 
eigvals_Sw = eig(Sw);
lambda_Sw = 0.1 * mean(eigvals_Sw(eigvals_Sw > 1e-10));
Sw_reg = Sw + lambda_Sw * eye(n);

% Solve generalized eigenvalue problem: Sb*w = λ*Sw*w
[W, D_lda] = eig(Sb, Sw_reg);

% Sort by eigenvalues (largest = most discriminative) 
[d_lda, ind] = sort(real(diag(D_lda)), 'descend');
W = W(:, ind);

% Keep top (C-1) = 5 discriminant directions 
nProjected = C - 1;
W_lda = W(:, 1:nProjected);

fprintf('  Reduced dimensionality: %d → %d\n', n, nProjected);
fprintf('  Top 5 eigenvalues: ');
fprintf('%.4f ', d_lda(1:5));
fprintf('\n\n');

% Project Data to low-dimensional space 
x = W_lda' * x_original;  % Now x is 5 x 10299


% % Update dimensions 
[n, N] = size(x);  

% Parameter estimation
priors = zeros(C, 1);
mu = zeros(n, C);
Sigma = zeros(n, n, C);

for l = 1:C
    indl = find(label == l);
    Nl = length(indl);
    x_class = x(:, indl);
    
    priors(l) = Nl / N;
    mu(:, l) = mean(x_class, 2);
    Sigma(:,:, l) = cov(x_class');
end

% Regularization in 5D space 
alpha = 0.05;
for l = 1:C
    eigvals = eig(Sigma(:,:,l));
    lambda = alpha * mean(eigvals);
    Sigma(:,:,l) = Sigma(:,:,l) + lambda * eye(n);
end
fprintf('\n');

% Classification
pxgivenl = zeros(C, N);
for l = 1:C
    pxgivenl(l, :) = evalGaussian(x, mu(:,l), Sigma(:,:,l));
end

px = priors' * pxgivenl;
classPosteriors = pxgivenl .* repmat(priors, 1, N) ./ repmat(px, C, 1);

[~, decisions] = max(classPosteriors, [], 1);
decisionsLabel = uniqueLabels(decisions);

% Confusion Matrux 
ConfusionMatrix = zeros(C, C);
for d = 1:C
    for l = 1:C
        ind_dl = find(decisionsLabel == d & label == l);
        indl = find(label == l);
        ConfusionMatrix(d, l) = length(ind_dl) / length(indl);
    end
end

fprintf('\n HUMAN ACTIVITY RECOGNITION RESULTS\n');
fprintf('Confusion Matrix P(D=d|L=l):\n');
fprintf('       ');
for l = 1:C
    fprintf('L=%d    ', l);
end
fprintf('\n');
for d = 1:C
    fprintf('D=%d: ', d);
    fprintf(' %.4f', ConfusionMatrix(d, :));
    fprintf('\n');
end

Perror_har = sum(decisionsLabel ~= label) / N;
fprintf('\nP(error): %.4f (%.2f%%)\n', Perror_har, 100*Perror_har);

fprintf('\nPer-Activity Accuracy:\n');
for l = 1:C
    fprintf('  %s: %.4f (%.2f%%)\n', activityNames{l}, ...
            ConfusionMatrix(l, l), 100*ConfusionMatrix(l, l));
end

% VISUALIZATION in LDA space 
figure(2), clf;
subplot(1, 2, 1);
hold on;
colors = lines(C);
for l = 1:C
    indl = (label == l);
    plot(x(1, indl), x(2, indl), '.', 'Color', colors(l, :), ...
         'MarkerSize', 10, 'DisplayName', activityNames{l});
end
xlabel('LDA Component 1'), ylabel('LDA Component 2');
title('HAR: First 2 LDA Components');
legend('Location', 'best');
grid on, axis equal;

subplot(1, 2, 2);
bar(d_lda(1:nProjected));
xlabel('LDA Component');
ylabel('Eigenvalue (Discriminative Power)');
title('HAR: LDA Eigenvalue Spectrum');
grid on;

% 3D visualization 
figure(3), clf;
hold on;
for l = 1:C
    indl = (label == l);
    plot3(x(1, indl), x(2, indl), x(3, indl), '.', 'Color', colors(l, :), ...
          'MarkerSize', 10, 'DisplayName', activityNames{l});
end
xlabel('LDA 1'), ylabel('LDA 2'), zlabel('LDA 3');
title('HAR: First 3 LDA Components');
legend('Location', 'best');
grid on;
view(45, 30);

fprintf('\nDISCUSSION: MODEL CHOICES AND APPROPRIATENESS\n');
fprintf('========================================\n\n');

fprintf('WINE QUALITY DATASET:\n');
fprintf('- Method: Direct Gaussian classification (no preprocessing)\n');
fprintf('- Dimensionality: n=11 (low-dimensional)\n');
fprintf('- Sample-to-feature ratio: %.1f:1 (excellent)\n', (4898/7)/11);
fprintf('- Regularization: None needed (well-conditioned covariances)\n');
fprintf('- P(error): %.2f%%\n', 100*Perror_wine);
fprintf('- Model appropriateness: Good\n');
fprintf('  * Sufficient samples for full covariance estimation\n');
fprintf('  * Continuous features suit Gaussian assumptions\n');
fprintf('  * Main limitation: High class overlap (subjective labels)\n\n');

fprintf('HUMAN ACTIVITY RECOGNITION DATASET:\n');
fprintf('- Method: Fisher LDA + Gaussian classification\n');
fprintf('- Original dimensionality: n=561 (extremely high-dimensional)\n');
fprintf('- Projected dimensionality: n=5 (C-1 discriminant directions)\n');
fprintf('- Sample-to-feature ratio after LDA: %.1f:1 (excellent)\n', (10299/6)/5);
fprintf('- P(error): %.2f%%\n', 100*Perror_har);
fprintf('- Model appropriateness: Good\n');
fprintf('  * LDA solves numerical underflow problem\n');
fprintf('  * Projects to maximally discriminative subspace\n');
fprintf('  * Gaussian assumption reasonable in LDA space\n');
fprintf('  * Alternative would be k-NN or discriminative methods\n\n');

fprintf('WHY THIS APPROACH WORKS:\n');
fprintf('1. Wine: Low dimensions → Direct Gaussian works\n');
fprintf('2. HAR: High dimensions → LDA preprocessing essential\n');
fprintf('3. LDA reduces 561D → 5D while preserving class separation\n');
fprintf('4. In 5D space, (2π)^(-5/2) ≈ 0.06 (no underflow!)\n');


function g = evalGaussian(x, mu, Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X

[n, N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);

end
