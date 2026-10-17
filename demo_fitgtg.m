%% Generalized Truncated Gumbel (GTG) Quantile Regression Demo
% This demo illustrates the generalized truncated Gumbel (GTG) distribution
% and the GTG quantile regression model (RGTG) of Gomez et al. (2024),
% implemented in FITGTG and the RegressionQuantileGTG class. The interface
% follows FITRQLINEAR and RegressionQuantileLinear.
%
% The demo contains four parts:
%
% # Shapes of the GTG pdf, cdf and hazard function (Figure 1 in [1]).
% # Application without covariates: dietary beta-carotene (BETADIET) from
%   the Plasma Retinol data set (Section 6.1 and Table 5 in [1]).
% # Application with covariates: quantile regression of body mass index
%   (BMI) on waist circumference and sex (Section 6.2 in [1]), using data
%   simulated from the estimates reported in Table 6 of [1].
% # Large data sets: chunked in-memory computations and tall arrays.
%
% Reference:
%   [1] H. J. Gomez, K. I. Santoro, D. Ayma, I. E. Cortes, D. I. Gallardo
%       and T. M. Magalhaes, "A New Generalization of the Truncated Gumbel
%       Distribution with Quantile Regression and Applications",
%       Mathematics, 12(11), 1762, 2024.

%% 1. Shapes of the GTG distribution
% The GTG(beta,lambda,alpha) distribution extends the left-truncated
% Gumbel (TG) distribution with a shape parameter alpha. For alpha = 1 the
% TG distribution is obtained.
z = linspace(1e-3, 5, 500)';
params = [-1 0.9; -1 2; 0 2; 1 2; 2 3; 1.2 1.5];   % [lambda alpha], beta = 1
figure('Name', 'GTG distribution shapes', 'Position', [100 100 1200 360]);
tiledlayout(1, 3, 'TileSpacing', 'compact');
labels = compose('GTG(1, %g, %g)', params);
nexttile; hold on
for k = 1:size(params, 1)
    plot(z, RegressionQuantileGTG.gtgpdf(z, 1, params(k, 1), params(k, 2)), 'LineWidth', 1.5);
end
title('(a) pdf'); xlabel('z'); ylabel('f(z)'); ylim([0 2]); legend(labels); grid on
nexttile; hold on
for k = 1:size(params, 1)
    plot(z, RegressionQuantileGTG.gtgcdf(z, 1, params(k, 1), params(k, 2)), 'LineWidth', 1.5);
end
title('(b) cdf'); xlabel('z'); ylabel('F(z)'); grid on
nexttile; hold on
for k = 1:size(params, 1)
    f = RegressionQuantileGTG.gtgpdf(z, 1, params(k, 1), params(k, 2));
    F = RegressionQuantileGTG.gtgcdf(z, 1, params(k, 1), params(k, 2));
    plot(z, f./(1 - F), 'LineWidth', 1.5);
end
title('(c) hazard function'); xlabel('z'); ylabel('h(z)'); ylim([0 5]); grid on

%% 2. Application without covariates: dietary beta-carotene
% The variable BETADIET (mcg per day, n = 315) is fitted with the GTG
% model, the TG model (ShapeAlpha fixed at 1) and the Weibull model. When
% the data cannot be downloaded, a sample is simulated from the GTG
% estimates of Table 5 in [1].
betadiet = loadBetadiet();

MdlGTG = fitgtg(betadiet, 'ResponseName', 'BETADIET')
MdlTG  = fitgtg(betadiet, 'ShapeAlpha', 1, 'FitShapeAlpha', false, 'ResponseName', 'BETADIET');
[betaGTG, lambdaGTG, alphaGTG] = gtgParameters(MdlGTG);
[betaTG, lambdaTG] = gtgParameters(MdlTG);

%%
% Compare models with the log-likelihood (LLF), AIC and BIC criteria
% (Table 5 in [1]).
modelNames = {'GTG'; 'TG'; 'Weibull'};
wblParams = wblfit(betadiet);
llWbl = -wbllike(wblParams, betadiet);
LLF = [MdlGTG.LogLikelihood; MdlTG.LogLikelihood; llWbl];
AIC = [MdlGTG.ModelCriterion.AIC; MdlTG.ModelCriterion.AIC; 4 - 2*llWbl];
BIC = [MdlGTG.ModelCriterion.BIC; MdlTG.ModelCriterion.BIC; 2*log(numel(betadiet)) - 2*llWbl];
disp(table(LLF, AIC, BIC, 'RowNames', modelNames))
fprintf('GTG: beta = %.2f, lambda = %.2f, alpha = %.2f\n', betaGTG, lambdaGTG, alphaGTG);
fprintf('TG:  beta = %.2f, lambda = %.2f\n', betaTG, lambdaTG);

%%
% Fitted densities and the fitted GTG cdf (Figure 5 in [1]).
xg = linspace(1, 1e4, 1000)';
figure('Name', 'BETADIET', 'Position', [100 100 1000 380]);
tiledlayout(1, 2, 'TileSpacing', 'compact');
nexttile
histogram(betadiet, 10, 'Normalization', 'pdf', 'FaceColor', [0.85 0.85 0.85]); hold on
plot(xg, RegressionQuantileGTG.gtgpdf(xg, betaGTG, lambdaGTG, alphaGTG), 'k', 'LineWidth', 2);
plot(xg, RegressionQuantileGTG.gtgpdf(xg, betaTG, lambdaTG, 1), 'r', 'LineWidth', 2);
plot(xg, wblpdf(xg, wblParams(1), wblParams(2)), 'g', 'LineWidth', 2);
legend('Data', 'GTG', 'TG', 'Weibull'); xlabel('BETADIET'); ylabel('Density');
title('(a) Fitted densities')
nexttile
ys = sort(betadiet);
stairs(ys, (1:numel(ys))'/numel(ys), 'k', 'LineWidth', 1.5); hold on
plot(xg, RegressionQuantileGTG.gtgcdf(xg, betaGTG, lambdaGTG, alphaGTG), 'b', 'LineWidth', 2);
legend('Empirical', 'GTG', 'Location', 'southeast');
xlabel('BETADIET'); ylabel('Cumulative distribution function'); title('(b) Empirical and fitted cdf')

%% 3. Quantile regression: body mass index
% The BMI is modeled as RGTG with log link for the p-th quantile,
%
%   log(Q_p(BMI | Waist, Sex)) = Bias(p) + Beta1(p)*Waist + Beta2(p)*(Sex == female),
%
% and shape parameters ShapeLambda(p) and log(ShapeAlpha(p)). The data are
% simulated using the median estimates of Table 6 in [1]:
% [Bias Beta] = [2.076 0.013 0.035], lambda = 3.504 and log(alpha) = 1.309.
rng(2024, 'twister');
n = 1743;
Waist = 97 + 14*randn(n, 1);
Sex = categorical(rand(n, 1) < 0.55, [false true], {'male', 'female'});
tau = exp(2.076 + 0.013*Waist + 0.035*(Sex == 'female'));   % true median BMI
lambdaTrue = 3.504;
alphaTrue = exp(1.309);
% Q_u(BMI|x) = tau*(h(lambda,u)/h(lambda,0.5))^(1/alpha), u ~ U(0,1)
h = @(u) RegressionQuantileGTG.hfun(lambdaTrue, u);
BMI = tau.*(h(rand(n, 1))./h(0.5)).^(1/alphaTrue);
tbl = table(Waist, Sex, BMI);

%%
% Fit the RGTG quantile regression model for several quantiles p with a
% single call. Sex is a categorical predictor (dummy coded, reference
% level 'male'). Each quantile is warm started from the previous fit.
pList = 0.1:0.1:0.9;
Mdl = fitgtg(tbl, 'BMI ~ Waist + Sex', 'Quantiles', pList)
MdlMedian = fitgtg(tbl, 'BMI ~ Waist + Sex', 'Quantiles', 0.5);

%%
% Estimates and standard errors across quantiles (compare with Table 6
% in [1]). Only the bias changes with p; the slopes and shape parameters
% are invariant, since the model describes the complete conditional
% distribution.
est = Mdl.Coefficients.Estimate';
se = Mdl.Coefficients.SE';
disp(array2table([pList' est], 'VariableNames', ...
    [{'p'}, Mdl.Coefficients.Properties.RowNames']))

figure('Name', 'Coefficients versus quantile', 'Position', [100 100 1100 330]);
tiledlayout(1, 5, 'TileSpacing', 'compact');
coefLabels = {'Bias(p)', 'Beta_{Waist}(p)', 'Beta_{female}(p)', '\lambda(p)', 'log \alpha(p)'};
for j = 1:5
    nexttile
    errorbar(pList, est(:, j), 1.96*se(:, j), 'o-', 'LineWidth', 1.2);
    xlabel('p'); title(coefLabels{j}); grid on; xlim([0 1])
end

%%
% Predicted conditional quantiles of BMI versus waist circumference. A
% single fitted model predicts every quantile; the third output of
% PREDICT contains 95% confidence intervals.
wGrid = linspace(min(Waist), max(Waist), 100)';
qPlot = [0.1 0.25 0.5 0.75 0.9];
figure('Name', 'BMI quantile curves', 'Position', [100 100 1000 400]);
tiledlayout(1, 2, 'TileSpacing', 'compact');
for s = categories(Sex)'
    nexttile
    sel = Sex == s{1};
    scatter(Waist(sel), BMI(sel), 8, [0.6 0.6 0.6], 'filled', 'MarkerFaceAlpha', 0.4); hold on
    gridTbl = table(wGrid, repmat(categorical(s, categories(Sex)), 100, 1), ...
        'VariableNames', {'Waist', 'Sex'});
    yq = predict(MdlMedian, gridTbl, 'Quantiles', qPlot);
    [~, ~, yci] = predict(MdlMedian, gridTbl);
    plot(wGrid, yq, 'LineWidth', 1.5);
    plot(wGrid, yci, 'k--');
    xlabel('Waist (cm)'); ylabel('BMI (kg/m^2)'); title(s{1}); grid on
    legend([{'Data'}, compose('p = %.2f', qPlot), {'95% CI median', ''}], 'Location', 'northwest');
end

%%
% Model adequacy: normalized quantile residuals (Dunn and Smyth, 1996)
% should be standard normal when the model is correctly specified.
r = residuals(MdlMedian);
rs = sort(r);
nq = -sqrt(2)*erfcinv(2*((1:n)' - 0.5)/n);
figure('Name', 'Quantile residuals', 'Position', [100 100 900 360]);
tiledlayout(1, 2, 'TileSpacing', 'compact');
nexttile
plot(nq, rs, '.'); hold on; plot(nq, nq, 'r', 'LineWidth', 1.5);
xlabel('Theoretical quantiles'); ylabel('Empirical quantiles');
title('Normal QQ-plot of quantile residuals'); grid on
nexttile
plot(resubPredict(MdlMedian), r, '.'); yline(0, 'r');
xlabel('Fitted median BMI'); ylabel('Quantile residual'); title('Residuals versus fitted'); grid on

%%
% Empirical coverage of the predicted quantiles: the fraction of
% observations below the predicted p-th quantile should be close to p.
% The quantile (pinball) loss is returned by LOSS and RESUBLOSS.
coverage = mean(BMI <= resubPredict(Mdl), 1);
disp(table(pList', coverage', resubLoss(Mdl)', ...
    'VariableNames', {'p', 'EmpiricalCoverage', 'QuantileLoss'}))

%% 4. Large data sets
% For in-memory data, FITGTG evaluates the likelihood, gradient and
% Hessian in chunks of rows (option 'ChunkSize'), and the predictor matrix
% is not copied. Here a model with 20 predictors is fitted to one million
% observations. Use 'Verbose',1 to display the iterations.
rng(7, 'twister');
nBig = 1e6;
mBig = 20;
XBig = randn(nBig, mBig);
betaBig = linspace(-0.2, 0.2, mBig)';
yBig = exp(1 + XBig*betaBig).*RegressionQuantileGTG.gtgrnd(1, 1.5, 2, nBig, 1);
tic
MdlBig = fitgtg(XBig, yBig, 'Quantiles', [0.1 0.5 0.9]);
fprintf('Fitted %d observations with %d predictors in %.1f s (%s, %s iterations).\n', ...
    nBig, mBig, toc, MdlBig.Solver, mat2str(MdlBig.ConvergenceInfo.NumIterations));
fprintf('Max. abs. error of Beta: %.2g\n', max(abs(MdlBig.Beta(:, 2) - betaBig)));
CMdlBig = compact(MdlBig);   % remove the training data to save memory

%%
% Data that do not fit in memory can be fitted with tall arrays. All
% computations are reductions over the data. Here the tall array is
% created from an in-memory table and evaluated in the MATLAB session;
% with a parallel pool or a cluster, the passes are distributed. In
% practice, create the tall table from a datastore, for example
% tt = tall(datastore('bmi*.csv')).
mapreducer(0);
tt = tall(tbl);
MdlTall = fitgtg(tt, 'BMI ~ Waist + Sex', 'Quantiles', [0.5 0.9]);
disp(MdlTall.Coefficients)
YTall = gather(predict(MdlTall, head(tt, 5)))

%% Local functions
function betadiet = loadBetadiet()
%LOADBETADIET Load the BETADIET variable of the Plasma Retinol data set.
%   The data are read from Plasma_Retinol.txt in the folder of this demo,
%   which is downloaded from StatLib when missing. If the download fails,
%   a sample of size 315 is simulated from the GTG estimates in Table 5.
folder = fileparts(mfilename('fullpath'));
file = fullfile(folder, 'Plasma_Retinol.txt');
if ~isfile(file)
    try
        websave(file, 'http://lib.stat.cmu.edu/datasets/Plasma_Retinol', ...
            weboptions('Timeout', 30, 'ContentType', 'text'));
    catch err
        warning('demo_fitgtg:Download', ...
            'Could not download data (%s). Using simulated data.', err.message);
        rng(315, 'twister');
        betadiet = RegressionQuantileGTG.gtgrnd(290.83, 2.61, 0.60, 315, 1);
        return
    end
end
lines = splitlines(string(fileread(file)));
vals = zeros(0, 14);
for k = 1:numel(lines)
    v = sscanf(lines(k), '%f')';
    if numel(v) == 14
        vals(end+1, :) = v; %#ok<AGROW>
    end
end
betadiet = vals(:, 11);
end
