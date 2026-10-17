function results = examples_fitgtg(varargin)
%EXAMPLES_FITGTG Real-data fitting and simulated health-screening workflows.
%   RESULTS = EXAMPLES_FITGTG runs three examples and creates three figures:
%      1. GTG versus TG fits to measured dietary beta-carotene intake.
%      2. Weighted BMI quantile regression with a held-out test set,
%         categorical predictors, confidence intervals and diagnostics.
%      3. Chunked fitting and compact prediction for 50,000 records.
%
%   RESULTS = EXAMPLES_FITGTG('IncludeTall',true) also fits the BMI
%   training table as a tall array, using the current mapreducer.
%   For serial evaluation, call mapreducer(0) before this example.
%
%   RESULTS contains the fitted models, evaluation tables, BMI training
%   and test data, and figure handles. The random number generator state
%   is restored on return. No downloads or file writes are performed.
%
%   Example 1 uses the bundled Plasma_Retinol.txt data from StatLib.
%   Examples 2 and 3 use SIMULATED data, with BMI parameters inspired by
%   Table 6 of Gomez et al. (2024), Mathematics 12(11):1762. They illustrate
%   software capabilities, not clinical prediction or causal effects.
%
%   See also FITGTG, REGRESSIONQUANTILEGTG, DEMO_FITGTG.

parser = inputParser;
addParameter(parser, 'IncludeTall', false, ...
    @(v) isscalar(v) && (islogical(v) || isnumeric(v)) && any(v == [0 1]));
parse(parser, varargin{:});
includeTall = logical(parser.Results.IncludeTall);
state = rng;
restoreRng = onCleanup(@() rng(state));
rng(2024, 'twister');
results = struct('Figures', gobjects(0, 1), 'TallModel', [], ...
    'TallPredictionDifference', []);

%% 1. Measured dietary beta-carotene intake: distribution selection
% BETADIET is dietary intake in mcg/day, not plasma concentration.
% Keep the measured data separate from the simulations below.
dataFile = fullfile(fileparts(mfilename('fullpath')), 'Plasma_Retinol.txt');
betadiet = readBetadiet(dataFile);
MdlDiet = fitgtg(betadiet, 'ResponseName', 'BETADIET');
MdlTG = fitgtg(betadiet, 'ShapeAlpha', 1, 'FitShapeAlpha', false, ...
    'ResponseName', 'BETADIET');
results.DietaryModel = MdlDiet;
results.TruncatedGumbelModel = MdlTG;
results.DietaryData = betadiet;
results.DietaryComparison = table( ...
    [MdlDiet.LogLikelihood; MdlTG.LogLikelihood], ...
    [MdlDiet.ModelCriterion.AIC; MdlTG.ModelCriterion.AIC], ...
    [MdlDiet.ModelCriterion.BIC; MdlTG.ModelCriterion.BIC], ...
    'VariableNames', {'LogLikelihood', 'AIC', 'BIC'}, ...
    'RowNames', {'GTG', 'TG'});
fprintf('\nMeasured dietary beta-carotene: %d observations\n', numel(betadiet));
disp(results.DietaryComparison);
dietQuantiles = [0.1 0.5 0.9];
results.DietaryQuantiles = table(dietQuantiles', ...
    predict(MdlDiet, [], 'Quantiles', dietQuantiles)', ...
    'VariableNames', {'Quantile', 'IntakeMcgPerDay'});
disp(results.DietaryQuantiles);

results.Figures(end+1, 1) = figure('Name', 'Measured beta-carotene intake', ...
    'Position', [100 100 1100 420]);
tiledlayout(1, 2, 'TileSpacing', 'compact');
z = linspace(1, max(betadiet), 600)';
nexttile
histogram(betadiet, 20, 'Normalization', 'pdf', 'DisplayName', 'Measured data');
hold on
plot(z, pdf(MdlDiet, [], z), 'LineWidth', 1.5, 'DisplayName', 'GTG');
plot(z, pdf(MdlTG, [], z), '--', 'LineWidth', 1.5, 'DisplayName', 'TG');
xlabel('Dietary beta-carotene (mcg/day)'); ylabel('Density');
title('Distribution fits'); legend('show'); grid on
nexttile
stairs(sort(betadiet), (1:numel(betadiet))'/numel(betadiet), ...
    'DisplayName', 'Empirical CDF');
hold on
plot(z, cdf(MdlDiet, [], z), 'LineWidth', 1.5, 'DisplayName', 'GTG CDF');
xlabel('Dietary beta-carotene (mcg/day)'); ylabel('Cumulative probability');
title('Check the fitted distribution'); legend('show'); grid on

%% 2. Simulated BMI screening: train/test separation and weighted fitting
% The weights are hypothetical sampling weights depending on sex.
% They demonstrate weighted likelihood, not complex-survey inference.
% lambda and alpha are constant across people in this parametric model.
n = 2500;
Waist = 97 + 12*randn(n, 1);
Sex = categorical(rand(n, 1) < 0.55, [false true], {'male', 'female'});
medianBMI = exp(2.076 + 0.013*Waist + 0.035*(Sex == 'female'));
lambda = 3.504;
alpha = exp(1.309);
unitMedian = RegressionQuantileGTG.gtginv(0.5, 1, lambda, alpha);
BMI = medianBMI.*RegressionQuantileGTG.gtgrnd(1, lambda, alpha, n, 1)/unitMedian;
SamplingWeight = 1 + 0.5*double(Sex == 'female');
screening = table(Waist, Sex, BMI, SamplingWeight);
order = randperm(n);
nTrain = floor(0.75*n);
trainTbl = screening(order(1:nTrain), :);
testTbl = screening(order(nTrain+1:end), :);

% Missing training entries and zero weights are deliberately introduced.
% Test observations remain untouched; no test responses enter estimation.
trainTbl.Waist(11) = NaN;
trainTbl.Sex(23) = categorical(missing);
trainTbl.BMI(37) = NaN;
trainTbl.SamplingWeight(41) = 0;
pList = [0.1 0.5 0.9];
MdlBMI = fitgtg(trainTbl, 'BMI ~ Waist + Sex', 'Weights', 'SamplingWeight', ...
    'Quantiles', pList, 'Standardize', true);
% Reported slopes use standardized predictors (including dummy columns);
% predict accepts the original table and applies the stored transformation.
results.BMIModel = MdlBMI;
results.BMITrainingData = trainTbl;
results.BMITestData = testTbl;
[testPred, crossing] = predict(MdlBMI, testTbl);
testWeight = testTbl.SamplingWeight;
coverage = sum(testWeight.*double(testTbl.BMI <= testPred), 1)/sum(testWeight);
testLoss = loss(MdlBMI, testTbl, 'BMI', 'Weights', 'SamplingWeight');
results.BMIValidation = table(pList', coverage', testLoss', ...
    'VariableNames', {'Quantile', 'WeightedTestCoverage', 'WeightedPinballLoss'});
results.NumCrossingTestRows = nnz(crossing);
fprintf('\nSIMULATED BMI: %d training rows used; %d held-out observations\n', ...
    MdlBMI.NumObservations, height(testTbl));
disp(results.BMIValidation);
disp(MdlBMI.Coefficients);

% Confidence intervals describe uncertainty in the estimated median,
% not the spread of individual BMI values. The 10% and 90% curves describe
% that conditional spread separately.
results.Figures(end+1, 1) = figure('Name', 'Simulated BMI: held-out quantiles', ...
    'Position', [100 100 1100 420]);
tiledlayout(1, 2, 'TileSpacing', 'compact');
waistGrid = linspace(70, 130, 120)';
sexNames = {'male', 'female'};
for j = 1:numel(sexNames)
    profiles = table(waistGrid, repmat(categorical(sexNames(j), categories(Sex)), ...
        numel(waistGrid), 1), 'VariableNames', {'Waist', 'Sex'});
    curves = predict(MdlBMI, profiles);
    [~, ~, medianCI] = predict(MdlBMI, profiles, 'Quantiles', 0.5);
    nexttile
    fill([waistGrid; flipud(waistGrid)], [medianCI(:, 1); flipud(medianCI(:, 2))], ...
        [0.8 0.9 1], 'EdgeColor', 'none', 'DisplayName', '95% CI of median');
    hold on
    sel = testTbl.Sex == sexNames{j};
    scatter(testTbl.Waist(sel), testTbl.BMI(sel), 8, [0.6 0.6 0.6], ...
        'filled', 'DisplayName', 'Held-out simulated data');
    for k = 1:numel(pList)
        plot(waistGrid, curves(:, k), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('Quantile %.1f', pList(k)));
    end
    xlabel('Waist circumference (cm)'); ylabel('BMI (kg/m^2)');
    title(sprintf('Simulated %s participants', sexNames{j}));
    legend('show', 'Location', 'northwest'); grid on
end

% A fitted model can answer probability questions and simulate new responses.
% Here BMI > 30 is only an illustrative threshold, not a clinical assessment.
patients = table([90; 110], categorical(sexNames', categories(Sex)), ...
    'VariableNames', {'Waist', 'Sex'});
[patientMedian, ~, patientCI] = predict(MdlBMI, patients, 'Quantiles', 0.5);
probabilityAbove30 = 1 - cdf(MdlBMI, patients, 30);
drawsPerPatient = 5000;
replicatedPatients = patients(repelem((1:height(patients))', drawsPerPatient), :);
draws = reshape(random(MdlBMI, replicatedPatients), drawsPerPatient, []);
results.PatientPredictions = patients;
results.PatientPredictions.MedianBMI = patientMedian;
results.PatientPredictions.MedianLower95 = patientCI(:, 1);
results.PatientPredictions.MedianUpper95 = patientCI(:, 2);
results.PatientPredictions.ProbabilityAbove30 = probabilityAbove30;
results.PatientPredictions.SimulatedProbabilityAbove30 = mean(draws > 30, 1)';
disp(results.PatientPredictions);

% The normal quantile residuals check the whole conditional distribution.
% Rows excluded during training have NaN residuals and are omitted here.
r = residuals(MdlBMI);
valid = isfinite(r);
rs = sort(r(valid));
normalQuantiles = -sqrt(2)*erfcinv(2*((1:numel(rs))' - 0.5)/numel(rs));
results.Figures(end+1, 1) = figure('Name', 'Simulated BMI: model diagnostics', ...
    'Position', [100 100 950 420]);
tiledlayout(1, 2, 'TileSpacing', 'compact');
nexttile
plot(normalQuantiles, rs, '.'); hold on
plot(normalQuantiles, normalQuantiles, 'k--');
xlabel('Standard normal quantiles'); ylabel('Quantile residuals');
title('Training residual QQ plot'); grid on
nexttile
fittedMedian = predict(MdlBMI, trainTbl, 'Quantiles', 0.5);
plot(fittedMedian(valid), r(valid), '.'); yline(0, 'k--');
xlabel('Fitted median BMI'); ylabel('Quantile residual');
title('Check residual structure'); grid on

%% 3. A larger simulated screening batch: chunking, ridge and compact models
% Explicit L-BFGS avoids a Hessian on each optimization iteration.
% Covariance estimation still evaluates the final full Hessian.
nLarge = 50000;
X = [97 + 12*randn(nLarge, 1), double(rand(nLarge, 1) < 0.55)];
largeMedian = exp(2.076 + X*[0.013; 0.035]);
y = largeMedian.*RegressionQuantileGTG.gtgrnd(1, lambda, alpha, nLarge, 1)/unitMedian;
timer = tic;
MdlLarge = fitgtg(X, y, 'PredictorNames', {'Waist', 'Female'}, ...
    'ResponseName', 'BMI', 'Quantiles', pList, 'ChunkSize', 2048, ...
    'Standardize', true, 'Lambda', 'auto', 'Solver', 'lbfgs');
fitTime = toc(timer);
results.CompactLargeModel = compact(MdlLarge);
results.LargeBatchPredictions = predict(results.CompactLargeModel, X(1:10, :));
results.LargeBatchSummary = table(nLarge, size(X, 2), 2048, fitTime, ...
    MdlLarge.Lambda, string(MdlLarge.Solver), ...
    'VariableNames', {'NumObservations', 'NumPredictors', 'ChunkSize', ...
    'FitTimeSeconds', 'RidgeStrength', 'Solver'});
fprintf('\nLarger SIMULATED screening batch (timing is machine-dependent)\n');
disp(results.LargeBatchSummary);

%% Optional: the same BMI model from a tall table
% This small in-memory-backed tall table makes the example self-contained.
% For genuinely out-of-memory work, replace it with tall(datastore(...)).
% Tall models do not retain training data; use explicit X/Y for evaluation.
if includeTall
    if exist('tall', 'file') == 0
        error('examples_fitgtg:TallUnavailable', 'This MATLAB installation does not support tall arrays.');
    end
    results.TallModel = fitgtg(tall(trainTbl), 'BMI ~ Waist + Sex', ...
        'Weights', 'SamplingWeight', 'Quantiles', pList, 'Standardize', true);
    tallPred = gather(predict(results.TallModel, tall(testTbl)));
    results.TallPredictionDifference = max(abs(tallPred - testPred), [], 'all');
    fprintf('Maximum tall/in-memory prediction difference: %.3g\n', ...
        results.TallPredictionDifference);
end
end

function betadiet = readBetadiet(dataFile)
%READBETADIET Extract column 11 from the bundled, header-prefixed data.
%   Do not substitute simulated values if the measured data are missing.
if ~isfile(dataFile)
    error('examples_fitgtg:MissingData', 'Bundled data file is missing: %s', dataFile);
end
lines = splitlines(string(fileread(dataFile)));
rows = zeros(0, 14);
for k = 1:numel(lines)
    row = sscanf(lines(k), '%f')';
    if numel(row) == 14
        rows(end+1, :) = row; %#ok<AGROW>
    elseif ~isempty(row)
        error('examples_fitgtg:InvalidData', 'Expected 14 numeric columns on data line %d.', k);
    end
end
if size(rows, 1) ~= 315 || any(~isfinite(rows(:, 11)) | rows(:, 11) <= 0)
    error('examples_fitgtg:InvalidData', 'Expected 315 positive, finite BETADIET observations.');
end
betadiet = rows(:, 11);
end
