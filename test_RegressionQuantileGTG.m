classdef test_RegressionQuantileGTG < matlab.unittest.TestCase
%TEST_REGRESSIONQUANTILEGTG Regression tests for FITGTG and RegressionQuantileGTG.
%   Run the tests with
%
%      results = runtests('test_RegressionQuantileGTG')
%
%   The tests verify the GTG distribution functions, the analytic
%   gradient and Hessian of the log-likelihood, the estimators and
%   solvers, prediction, the fitrqlinear-style interface (tables,
%   formulas, categorical predictors, weights, regularization,
%   standardization, ObservationsIn), large data support (chunked and
%   tall computations), and reproduce published results of Gomez et al.
%   (2024), "A New Generalization of the Truncated Gumbel Distribution
%   with Quantile Regression and Applications", Mathematics 12(11), 1762.
%
%   The Table 5 tests (BETADIET data) read Plasma_Retinol.txt from the
%   folder of this file, or download it from StatLib. They are skipped
%   when the data are not available.
%
%   See also FITGTG, REGRESSIONQUANTILEGTG, DEMO_FITGTG.

    properties (TestParameter)
        % [beta lambda alpha] combinations covering decreasing and
        % unimodal shapes (Figure 1 in [1])
        gtgParams = struct( ...
            'decreasing', [1 -1 0.9], ...
            'unimodal',   [1 1 2], ...
            'peaked',     [2 2 3], ...
            'negLambda',  [0.5 -2 1.5], ...
            'smallAlpha', [3 0 0.5]);
        solver = {'newton', 'bfgs', 'lbfgs'};
    end

    properties
        OriginalWarningState
    end

    methods (TestMethodSetup)
        function saveWarningState(testCase)
            testCase.OriginalWarningState = warning;
            testCase.addTeardown(@() warning(testCase.OriginalWarningState));
        end
    end

    %======================================================================
    % GTG distribution functions
    %======================================================================
    methods (Test)
        function testPdfIntegratesToOne(testCase, gtgParams)
            prm = num2cell(gtgParams);
            area = integral(@(z) RegressionQuantileGTG.gtgpdf(z, prm{:}), 0, Inf);
            testCase.verifyEqual(area, 1, 'AbsTol', 1e-6);
        end

        function testCdfIsIntegralOfPdf(testCase, gtgParams)
            prm = num2cell(gtgParams);
            z = [0.1 0.5 1 2 4];
            for k = 1:numel(z)
                F = integral(@(t) RegressionQuantileGTG.gtgpdf(t, prm{:}), 0, z(k));
                testCase.verifyEqual(RegressionQuantileGTG.gtgcdf(z(k), prm{:}), F, 'AbsTol', 1e-7);
            end
        end

        function testInverseCdfRoundTrip(testCase, gtgParams)
            prm = num2cell(gtgParams);
            p = [1e-6 0.01 0.25 0.5 0.75 0.99 1-1e-6];
            z = RegressionQuantileGTG.gtginv(p, prm{:});
            testCase.verifyEqual(RegressionQuantileGTG.gtgcdf(z, prm{:}), p, 'AbsTol', 1e-10);
            testCase.verifyTrue(all(diff(z) > 0));
        end

        function testSupportBoundaries(testCase)
            testCase.verifyEqual(RegressionQuantileGTG.gtgpdf([-1 0], 1, 1, 2), [0 0]);
            testCase.verifyEqual(RegressionQuantileGTG.gtgcdf([-1 0 Inf], 1, 1, 2), [0 0 1]);
            testCase.verifyEqual(RegressionQuantileGTG.gtginv([0 1], 1, 1, 2), [0 Inf]);
        end

        function testExtremeDistributionRoundTrip(testCase)
            lambda = [-1000 -40 -30 -3 0 4 10];
            p = [0; 1e-12; 0.1; 0.5; 0.9; 1-1e-12; 1];
            for alpha = [0.5 1 2]
                z = RegressionQuantileGTG.gtginv(p, 1.7, lambda, alpha);
                testCase.verifyTrue(isreal(z));
                testCase.verifyEqual(z(1, :), zeros(size(lambda)));
                testCase.verifyEqual(z(end, :), inf(size(lambda)));
                testCase.verifyEqual(RegressionQuantileGTG.gtgcdf(z, 1.7, lambda, alpha), ...
                    repmat(p, 1, numel(lambda)), 'AbsTol', 5e-12);
                testCase.verifyEqual(RegressionQuantileGTG.gtgpdf(Inf, 1.7, lambda, alpha), ...
                    zeros(size(lambda)));
            end
            testCase.verifyTrue(isnan(RegressionQuantileGTG.gtgpdf(Inf, NaN, 1, 2)));
            testCase.verifyTrue(isnan(RegressionQuantileGTG.gtgpdf(NaN, 1, 1, 2)));
            testCase.verifyTrue(isnan(RegressionQuantileGTG.gtginv(NaN, 1, 4, 2)));
            testCase.verifyTrue(isnan(RegressionQuantileGTG.gtgcdf(NaN, 1, -40, 2)));
            testCase.verifyTrue(isnan(RegressionQuantileGTG.gtgcdf(1, NaN, 1, 2)));
        end

        function testWeibullLimitDistribution(testCase)
            z = [0; 0.1; 0.5; 1; 3; Inf];
            beta = 1.7; alpha = 2;
            F = -expm1(-(z/beta).^alpha);
            testCase.verifyEqual(RegressionQuantileGTG.gtgcdf(z, beta, [-1000 -40], alpha), ...
                repmat(F, 1, 2), 'AbsTol', 1e-14);
            p = [0.1; 0.5; 0.9];
            [h, dh, d2h] = RegressionQuantileGTG.hfun([-1000 -40], p);
            testCase.verifyEqual(h, repmat(-log1p(-p), 1, 2), 'AbsTol', 1e-14);
            testCase.verifyEqual(dh, zeros(3, 2));
            testCase.verifyEqual(d2h, zeros(3, 2));
            y = [0.1; 0.5; 1; 3];
            theta = [log(beta) + log(-log1p(-0.5))/alpha; -1000; log(alpha)];
            [ll, g, H] = RegressionQuantileGTG.loglikelihood(theta, zeros(4, 0), y, 0.5);
            testCase.verifyEqual(ll, sum(log(RegressionQuantileGTG.gtgpdf(y, beta, -1000, alpha))), ...
                'AbsTol', 1e-10);
            testCase.verifyTrue(all(isfinite([ll; g; H(:)])));
        end

        function testAlphaOneIsTruncatedGumbel(testCase)
            % Remark 1: GTG(beta,lambda,1) = TG(beta,lambda), Eq. (1)
            beta = 2.5; lambda = 0.7; y = linspace(0.01, 10, 50);
            g = @(u) exp(-u - exp(-u));
            G = @(u) exp(-exp(-u));
            fTG = g(y/beta - lambda)./(beta*(1 - G(-lambda)));
            testCase.verifyEqual(RegressionQuantileGTG.gtgpdf(y, beta, lambda, 1), fTG, 'RelTol', 1e-12);
        end

        function testStochasticRepresentation(testCase)
            % Z = beta*Y^(1/alpha) with Y ~ TG(1,lambda)
            beta = 1.7; lambda = 0.4; alpha = 2.2; z = [0.3 1 2.5];
            Fz = RegressionQuantileGTG.gtgcdf(z, beta, lambda, alpha);
            Fy = RegressionQuantileGTG.gtgcdf((z/beta).^alpha, 1, lambda, 1);
            testCase.verifyEqual(Fz, Fy, 'RelTol', 1e-12);
        end

        function testQuartilesCorollary1(testCase)
            beta = 2; lambda = 1.3; alpha = 1.7;
            G = @(u) exp(-exp(-u));
            Ginv = @(p) -log(-log(p));
            q = beta*(lambda + Ginv([0.25 0.5 0.75] + [0.75 0.5 0.25]*G(-lambda))).^(1/alpha);
            testCase.verifyEqual(RegressionQuantileGTG.gtginv([0.25 0.5 0.75], beta, lambda, alpha), ...
                q, 'RelTol', 1e-12);
        end

        function testSkewnessKurtosisTables1And2(testCase)
            % Skewness and kurtosis of GTG(1,lambda,alpha), Tables 1 and 2
            cases = [ 0   1   1.6782  7.2726
                      0   0.5 5.5570 62.8258
                     -2   1   1.9502  8.7063
                     -2   0.5 6.4487 83.4229
                     -1.5 1   1.9190  8.5270];
            warning('off', 'MATLAB:integral:MinStepSize');
            for k = 1:size(cases, 1)
                l = cases(k, 1); a = cases(k, 2);
                mk = @(r) integral(@(u) RegressionQuantileGTG.gtginv(u, 1, l, a).^r, 0, 1, ...
                    'RelTol', 1e-9, 'AbsTol', 1e-12);
                m1 = mk(1); m2 = mk(2); m3 = mk(3); m4 = mk(4);
                v = m2 - m1^2;
                skew = (m3 - 3*m1*m2 + 2*m1^3)/v^1.5;
                kurt = (m4 - 4*m1*m3 + 6*m1^2*m2 - 3*m1^4)/v^2;
                testCase.verifyEqual(skew, cases(k, 3), 'AbsTol', 1e-3);
                testCase.verifyEqual(kurt, cases(k, 4), 'RelTol', 1e-3);
            end
        end

        function testGtgstatMatchesQuantileIntegral(testCase)
            [m, v] = RegressionQuantileGTG.gtgstat([1 2], [0 1], [1 2]);
            testCase.verifyEqual(m(1), 1.26020201078938, 'RelTol', 1e-8);
            testCase.verifyEqual(v(1), 1.23164666695424, 'RelTol', 1e-7);
            z = RegressionQuantileGTG.gtginv(((1:1e5) - 0.5)/1e5, 2, 1, 2);
            testCase.verifyEqual(m(2), mean(z), 'RelTol', 1e-4);
        end

        function testRandomNumbers(testCase)
            rng(7, 'twister');
            z = RegressionQuantileGTG.gtgrnd(1.5, 0.8, 1.7, 2e5, 1);
            testCase.verifySize(z, [2e5 1]);
            zs = sort(z);
            Femp = ((1:numel(zs))' - 0.5)/numel(zs);
            ks = max(abs(RegressionQuantileGTG.gtgcdf(zs, 1.5, 0.8, 1.7) - Femp));
            testCase.verifyLessThan(ks, 1.36/sqrt(numel(zs)));
            testCase.verifySize(RegressionQuantileGTG.gtgrnd(1, 1, [1 2; 3 4]), [2 2]);
            testCase.verifySize(RegressionQuantileGTG.gtgrnd(1, 1, 1, 3), [3 3]);
        end

        function testHfunctionDerivatives(testCase)
            lambda = [-30 -3 0 2 15]; p = [0.1 0.5 0.9]';
            [~, dh, d2h] = RegressionQuantileGTG.hfun(lambda, p);
            d = 1e-5;
            [hp, dhp] = RegressionQuantileGTG.hfun(lambda + d, p);
            [hm, dhm] = RegressionQuantileGTG.hfun(lambda - d, p);
            testCase.verifyEqual(dh, (hp - hm)/(2*d), 'AbsTol', 1e-6);
            testCase.verifyEqual(d2h, (dhp - dhm)/(2*d), 'AbsTol', 1e-6);
            % h(lambda,p) tends to -log(1-p) when lambda -> -Inf
            testCase.verifyEqual(RegressionQuantileGTG.hfun(-40, p), -log(1 - p), 'RelTol', 1e-6);
        end
    end

    %======================================================================
    % Likelihood
    %======================================================================
    methods (Test)
        function testLogLikelihoodMatchesDensity(testCase)
            rng(3, 'twister');
            n = 30; X = randn(n, 1); y = exp(randn(n, 1));
            theta = [0.2; -0.4; 1.1; 0.3]; p = 0.7;
            ll = RegressionQuantileGTG.loglikelihood(theta, X, y, p, []);
            alpha = exp(theta(4)); lambda = theta(3);
            beta = exp(theta(1) + X*theta(2))*RegressionQuantileGTG.hfun(lambda, p)^(-1/alpha);
            ref = sum(log(RegressionQuantileGTG.gtgpdf(y, beta, lambda, alpha)));
            testCase.verifyEqual(ll, ref, 'RelTol', 1e-12);
            % Weighted log-likelihood
            w = rand(n, 1);
            llw = RegressionQuantileGTG.loglikelihood(theta, X, y, p, w);
            testCase.verifyEqual(llw, sum(w.*log(RegressionQuantileGTG.gtgpdf(y, beta, lambda, alpha))), 'RelTol', 1e-12);
        end

        function testAnalyticGradientAndHessian(testCase)
            rng(1, 'twister');
            n = 50; X = randn(n, 2); y = -2*log(rand(n, 1));
            theta = [0.3; 0.1; -0.2; 1.2; 0.4];
            for p = [0.1 0.5 0.9]
                for lam = [-4 1.2 6]
                    th = theta; th(4) = lam;
                    [~, g, H] = RegressionQuantileGTG.loglikelihood(th, X, y, p, []);
                    gNum = zeros(5, 1); HNum = zeros(5);
                    for j = 1:5
                        e = zeros(5, 1); e(j) = 1e-6;
                        [lp, gp] = RegressionQuantileGTG.loglikelihood(th + e, X, y, p, []);
                        [lm, gm] = RegressionQuantileGTG.loglikelihood(th - e, X, y, p, []);
                        gNum(j) = (lp - lm)/2e-6;
                        HNum(:, j) = (gp - gm)/2e-6;
                    end
                    testCase.verifyEqual(g, gNum, 'AbsTol', 1e-5*max(1, norm(g)));
                    testCase.verifyEqual(H, HNum, 'AbsTol', 1e-5*max(1, max(abs(H(:)))));
                    testCase.verifyEqual(H, H', 'AbsTol', 1e-10*max(abs(H(:))));
                end
            end
        end
    end

    %======================================================================
    % Fitting and prediction
    %======================================================================
    methods (Test)
        function testRegressionSnapshot(testCase)
            % Baseline estimates for the simulation scenario of Eq. (16)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(500, 0.5, 1989);
            Mdl = fitgtg(X, y);
            testCase.verifyClass(Mdl, 'RegressionQuantileGTG');
            expEst = [2.06143271267402 0.014194534996068 0.0380005405718565 ...
                4.20445921501228 1.20168482258872]';
            expSE = [0.00576178186083538 0.00176526194707233 0.00697961263015503 ...
                0.668005303926512 0.15348242403518]';
            testCase.verifyEqual(Mdl.Coefficients.Estimate, expEst, 'RelTol', 1e-5);
            testCase.verifyEqual(Mdl.Coefficients.SE, expSE, 'RelTol', 1e-3);
            testCase.verifyEqual(Mdl.LogLikelihood, -466.852226511059, 'AbsTol', 1e-6);
            testCase.verifyEqual(Mdl.Bias, expEst(1), 'RelTol', 1e-5);
            testCase.verifyEqual(Mdl.Beta, expEst(2:3), 'RelTol', 1e-5);
            testCase.verifyEqual(Mdl.ShapeLambda, expEst(4), 'RelTol', 1e-5);
            testCase.verifyEqual(Mdl.ShapeAlpha, exp(expEst(5)), 'RelTol', 1e-5);
            testCase.verifyEqual(Mdl.Coefficients.Properties.RowNames, ...
                {'(Intercept)'; 'x1'; 'x2'; 'ShapeLambda'; 'log(ShapeAlpha)'});
            testCase.verifyEqual(Mdl.ConvergenceInfo.TerminationCode, 1);
            testCase.verifyEqual(Mdl.NumObservations, 500);
            testCase.verifyEqual(Mdl.ModelCriterion.AIC, -2*Mdl.LogLikelihood + 10, 'RelTol', 1e-12);
            testCase.verifyEqual(Mdl.ModelCriterion.BIC, -2*Mdl.LogLikelihood + 5*log(500), 'RelTol', 1e-12);
        end

        function testSolversAgree(testCase, solver)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(500, 0.5, 1989);
            ref = fitgtg(X, y, 'Solver', 'newton');
            Mdl = fitgtg(X, y, 'Solver', solver, 'GradientTolerance', 1e-8, 'BetaTolerance', 1e-10);
            testCase.verifyEqual(Mdl.Solver, solver);
            testCase.verifyEqual(Mdl.LogLikelihood, ref.LogLikelihood, 'AbsTol', 1e-7);
            testCase.verifyEqual(Mdl.Coefficients.Estimate, ref.Coefficients.Estimate, 'RelTol', 1e-4);
        end

        function testParameterRecovery(testCase)
            % Estimates are within 3 standard errors of the true values
            trueTheta = [2.071; 0.013; 0.034; 3.881; 1.220];
            for p = [0.1 0.9]
                [X, y] = test_RegressionQuantileGTG.simulateScenario16(3000, p, 11);
                Mdl = fitgtg(X, y, 'Quantiles', p);
                z = abs(Mdl.Coefficients.Estimate - trueTheta)./Mdl.Coefficients.SE;
                testCase.verifyLessThan(z, 3);
            end
        end

        function testMultipleQuantiles(testCase)
            % The RGTG model describes the same distribution for every p:
            % only the bias changes, by (log h(lambda,p2)-log h(lambda,p1))/alpha.
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(800, 0.5, 5);
            pq = [0.1 0.5 0.9];
            Mdl = fitgtg(X, y, 'Quantiles', pq);
            testCase.verifyEqual(Mdl.Quantiles, pq);
            testCase.verifySize(Mdl.Beta, [2 3]);
            testCase.verifySize(Mdl.Bias, [1 3]);
            testCase.verifySize(Mdl.Coefficients.Estimate, [5 3]);
            testCase.verifySize(Mdl.CoefficientCovariance, [5 5 3]);
            testCase.verifyEqual(Mdl.LogLikelihood, Mdl.LogLikelihood(1)*ones(1, 3), 'AbsTol', 1e-6);
            testCase.verifyEqual(Mdl.Beta, repmat(Mdl.Beta(:, 1), 1, 3), 'RelTol', 1e-4);
            testCase.verifyEqual(Mdl.ShapeLambda, Mdl.ShapeLambda(1)*ones(1, 3), 'RelTol', 1e-4);
            h = @(p) RegressionQuantileGTG.hfun(Mdl.ShapeLambda(1), p);
            testCase.verifyEqual(Mdl.Bias(3) - Mdl.Bias(1), ...
                (log(h(0.9)) - log(h(0.1)))/Mdl.ShapeAlpha(1), 'AbsTol', 1e-5);
            % Separate fits agree with the joint fit
            M9 = fitgtg(X, y, 'Quantiles', 0.9);
            testCase.verifyEqual(M9.Coefficients.Estimate, Mdl.Coefficients.Estimate(:, 3), 'RelTol', 1e-4);
            % Warm start: later quantiles need (almost) no iterations
            testCase.verifyLessThanOrEqual(Mdl.ConvergenceInfo.NumIterations(2:3), 2);
        end

        function testPredict(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(600, 0.5, 21);
            Mdl = fitgtg(X, y, 'Quantiles', [0.1 0.5 0.9]);
            YFit = predict(Mdl, X(1:10, :));
            testCase.verifySize(YFit, [10 3]);
            testCase.verifyTrue(all(YFit(:, 1) < YFit(:, 2) & YFit(:, 2) < YFit(:, 3)));
            testCase.verifyEqual(resubPredict(Mdl), predict(Mdl, X), 'RelTol', 1e-12);
            [y5, crossing, YCI] = predict(Mdl, X(1:10, :), 'Quantiles', 0.5);
            testCase.verifyEqual(y5, YFit(:, 2), 'RelTol', 1e-12);
            testCase.verifyFalse(any(crossing));
            testCase.verifySize(YCI, [10 2]);
            testCase.verifyTrue(all(YCI(:, 1) < y5 & y5 < YCI(:, 2)));
            [~, ~, YCI99] = predict(Mdl, X(1:10, :), 'Quantiles', 0.5, 'Alpha', 0.01);
            testCase.verifyTrue(all(YCI99(:, 1) < YCI(:, 1) & YCI(:, 2) < YCI99(:, 2)));
            [~, ~, YCIall] = predict(Mdl, X(1:10, :));
            testCase.verifySize(YCIall, [10 2 3]);
            % Predicted quantile is the conditional icdf, also for quantiles not fitted
            q = predict(Mdl, X(1:10, :), 'Quantiles', 0.75);
            testCase.verifyEqual(cdf(Mdl, X(1:10, :), q), 0.75*ones(10, 1), 'AbsTol', 1e-10);
            % Empirical coverage
            cover = mean(y <= predict(Mdl, X));
            testCase.verifyEqual(cover, Mdl.Quantiles, 'AbsTol', 0.05);
        end

        function testLossAndResubLoss(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(400, 0.5, 22);
            Mdl = fitgtg(X, y, 'Quantiles', [0.25 0.75]);
            YFit = predict(Mdl, X);
            r = y - YFit;
            ref = mean(max([0.25 0.75].*r, ([0.25 0.75] - 1).*r));
            testCase.verifyEqual(loss(Mdl, X, y), ref, 'RelTol', 1e-12);
            testCase.verifyEqual(resubLoss(Mdl), ref, 'RelTol', 1e-12);
            testCase.verifySize(loss(Mdl, X, y, 'Quantiles', 0.75), [1 1]);
            mse = loss(Mdl, X, y, 'LossFun', @(Y, YF, W, q) sum(W.*(Y - YF).^2)/sum(W));
            testCase.verifyEqual(mse, mean((y - YFit).^2), 'RelTol', 1e-12);
            w = rand(400, 1);
            lw = loss(Mdl, X, y, 'Weights', w);
            testCase.verifyEqual(lw, sum(w.*max([0.25 0.75].*r, ([0.25 0.75] - 1).*r))/sum(w), 'RelTol', 1e-12);
        end

        function testConditionalDistributionMethods(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(400, 0.5, 8);
            Mdl = fitgtg(X, y);
            [b, l, a] = gtgParameters(Mdl, X(1:3, :));
            testCase.verifyEqual(pdf(Mdl, X(1:3, :), y(1:3)), RegressionQuantileGTG.gtgpdf(y(1:3), b, l, a));
            testCase.verifyEqual(Mdl.LogLikelihood, sum(log(pdf(Mdl, X, y))), 'RelTol', 1e-10);
            rng(4, 'twister');
            ysim = random(Mdl);
            testCase.verifySize(ysim, size(y));
            testCase.verifyTrue(all(ysim > 0));
            r = residuals(Mdl);
            testCase.verifyEqual(mean(r), 0, 'AbsTol', 0.15);
            testCase.verifyEqual(std(r), 1, 'AbsTol', 0.1);
            testCase.verifyEqual(residuals(Mdl, 'Type', 'raw'), y - resubPredict(Mdl), 'RelTol', 1e-12);
            testCase.verifyEqual(residuals(Mdl, X, y), r, 'RelTol', 1e-12);
            ci = coefCI(Mdl);
            est = Mdl.Coefficients.Estimate;
            testCase.verifySize(ci, [5 2]);
            testCase.verifyTrue(all(ci(:, 1) < est & est < ci(:, 2)));
            ci99 = coefCI(Mdl, 0.01);
            testCase.verifyTrue(all(ci99(:, 1) < ci(:, 1)));
        end

        function testCompactModel(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(300, 0.5, 23);
            Mdl = fitgtg(X, y);
            testCase.verifyEqual(Mdl.X, X);
            testCase.verifyEqual(Mdl.Y, y);
            testCase.verifyEqual(sum(Mdl.W), 1, 'AbsTol', 1e-12);
            C = compact(Mdl);
            testCase.verifyEmpty(C.X);
            testCase.verifyEmpty(C.Y);
            testCase.verifyEqual(predict(C, X(1:5, :)), predict(Mdl, X(1:5, :)));
            testCase.verifyError(@() resubPredict(C), 'RegressionQuantileGTG:NoTrainingData');
            testCase.verifyError(@() residuals(C), 'RegressionQuantileGTG:NoTrainingData');
        end

        function testMissingValuesAreExcluded(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(300, 0.5, 10);
            Xn = X; Xn(5, 1) = NaN; yn = y; yn(9) = NaN;
            Mdl = fitgtg(Xn, yn);
            keep = [1:4 6:8 10:300];
            ref = fitgtg(X(keep, :), y(keep));
            testCase.verifyEqual(Mdl.NumObservations, 298);
            testCase.verifyEqual(Mdl.Coefficients.Estimate, ref.Coefficients.Estimate, 'RelTol', 1e-8);
            testCase.verifyEqual(find(~Mdl.RowsUsed), [5; 9]);
            YFit = resubPredict(Mdl);
            testCase.verifyTrue(isnan(YFit(5)));
            testCase.verifyEqual(resubLoss(Mdl), loss(ref, X(keep, :), y(keep)), 'RelTol', 1e-8);
        end

        function testNoPredictorsAndNoBias(testCase)
            rng(12, 'twister');
            y = RegressionQuantileGTG.gtgrnd(2, 1, 1.5, 2000, 1);
            Mdl = fitgtg(y, 'Quantiles', 0.75);
            testCase.verifyEmpty(Mdl.Beta);
            testCase.verifySize(predict(Mdl), [1 1]);
            testCase.verifyEqual(resubPredict(Mdl), predict(Mdl)*ones(2000, 1), 'RelTol', 1e-12);
            testCase.verifyEqual(predict(Mdl), RegressionQuantileGTG.gtginv(0.75, 2, 1, 1.5), 'RelTol', 0.05);
            x = 1 + rand(2000, 1);
            M0 = fitgtg(x, y.*x, 'FitBias', false);
            testCase.verifyEqual(M0.Coefficients.Properties.RowNames, {'x1'; 'ShapeLambda'; 'log(ShapeAlpha)'});
            testCase.verifyEqual(M0.Bias, 0);
            testCase.verifySize(M0.Beta, [1 1]);
        end

        function testRowVectorResponsesAndWeights(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(300, 0.5, 103);
            w = 1 + mod((1:300)', 3);
            X(5, 1) = NaN; y(9) = NaN; w(12) = 0;
            ref = fitgtg(X, y, 'Weights', w, 'Quantiles', [0.2 0.8]);
            for cs = [37 300]
                Mdl = fitgtg(X, y.', 'Weights', w.', 'ChunkSize', cs, 'Quantiles', [0.2 0.8]);
                testCase.verifyEqual(Mdl.Coefficients.Estimate, ref.Coefficients.Estimate, 'AbsTol', 1e-8);
                testCase.verifyEqual(Mdl.RowsUsed, ref.RowsUsed);
                testCase.verifyEqual(Mdl.W, ref.W);
                testCase.verifySize(Mdl.Y, [300 1]);
                Mc = fitgtg(X.', y.', 'Weights', w.', 'ObservationsIn', 'columns', ...
                    'ChunkSize', cs, 'Quantiles', [0.2 0.8]);
                testCase.verifyEqual(resubPredict(Mc), resubPredict(ref), 'AbsTol', 1e-8);
                testCase.verifyEqual(resubLoss(Mc), resubLoss(ref), 'AbsTol', 1e-8);
            end
            M0 = fitgtg(y, 'Weights', w);
            Mrow = fitgtg(y.', 'Weights', w.');
            testCase.verifyEqual(Mrow.Coefficients.Estimate, M0.Coefficients.Estimate);
            tbl = array2table(X);
            Mt = fitgtg(tbl, y.', 'Weights', w.');
            Mx = fitgtg(X, y, 'Weights', w);
            testCase.verifyEqual(Mt.Coefficients.Estimate, Mx.Coefficients.Estimate);
        end

        function testExcludedRowsInLossAndResiduals(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(300, 0.5, 104);
            w = ones(300, 1);
            w(1) = 0; X(2, 1) = Inf; w(3) = NaN; w(4) = 0; y(4) = 0; y(5) = NaN;
            keep = (6:300)';
            Mdl = fitgtg(X, y, 'Weights', w, 'Quantiles', [0.25 0.75]);
            ref = fitgtg(X(keep, :), y(keep), 'Quantiles', [0.25 0.75]);
            testCase.verifyEqual(find(~Mdl.RowsUsed), (1:5)');
            testCase.verifyEqual(resubLoss(Mdl), resubLoss(ref), 'AbsTol', 1e-8);
            mse = @(Y,YFit,W,q) sum(W.*(Y - YFit).^2)/sum(W);
            testCase.verifyEqual(resubLoss(Mdl, 'LossFun', mse), ...
                resubLoss(ref, 'LossFun', mse), 'AbsTol', 1e-8);
            for type = {'raw', 'quantile'}
                r = residuals(Mdl, 'Type', type{1});
                testCase.verifyTrue(all(isnan(r(1:5, :)), 'all'));
                testCase.verifyEqual(r(keep, :), residuals(ref, 'Type', type{1}), 'AbsTol', 1e-8);
            end
            rnew = residuals(Mdl, X(1, :), y(1));
            testCase.verifyTrue(isfinite(rnew));
            testCase.verifyEqual(loss(Mdl, X, y, 'Weights', w.*double(Mdl.RowsUsed)), ...
                resubLoss(Mdl), 'AbsTol', 1e-12);
        end

        function testSingleLevelCategoricalValidity(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(300, 0.5, 105);
            g = categorical(repmat({'a'}, 300, 1)); g(3) = categorical(missing);
            tbl = table(X(:, 1), g, y, 'VariableNames', {'x', 'g', 'y'});
            keep = [1:2 4:300]';
            Mdl = fitgtg(tbl, 'y ~ x + g');
            ref = fitgtg(X(keep, 1), y(keep));
            testCase.verifyEqual(Mdl.NumObservations, 299);
            testCase.verifyEqual(Mdl.Coefficients.Estimate, ref.Coefficients.Estimate, 'AbsTol', 1e-8);
            bad = table(0, categorical({'unknown'}), 'VariableNames', {'x', 'g'});
            testCase.verifyTrue(isnan(predict(Mdl, bad)));
            M0 = fitgtg(tbl, 'y ~ g');
            ref0 = fitgtg(y(keep));
            testCase.verifyEqual(M0.Coefficients.Estimate, ref0.Coefficients.Estimate, 'AbsTol', 1e-8);
            yp = resubPredict(M0);
            testCase.verifyTrue(isnan(yp(3)));
            testCase.verifyTrue(isnan(predict(M0, bad)));
            testCase.verifyTrue(isnan(pdf(M0, bad, 1)));
            testCase.verifyTrue(isnan(cdf(M0, bad, 1)));
            testCase.verifyEqual(predict(M0), resubPredict(M0));
            testCase.verifyError(@() predict(M0, table(0, 'VariableNames', {'x'})), ...
                'RegressionQuantileGTG:InvalidX');
            codes = ones(300, 1); codes(3) = NaN;
            Mn = fitgtg(codes, y, 'CategoricalPredictors', 'all');
            testCase.verifyEqual(Mn.Coefficients.Estimate, ref0.Coefficients.Estimate, 'AbsTol', 1e-8);
            testCase.verifyTrue(isnan(predict(Mn, 2)));
            Mc = fitgtg(codes.', y.', 'CategoricalPredictors', 'all', 'ObservationsIn', 'columns');
            testCase.verifyEqual(resubPredict(Mc), resubPredict(Mn));
            tbl.g(:) = categorical(missing);
            testCase.verifyError(@() fitgtg(tbl, 'y ~ g'), 'RegressionQuantileGTG:NoData');
            testCase.verifyError(@() fitgtg(tbl(1:5, :), y), 'RegressionQuantileGTG:SizeMismatch');
        end

        function testWeightsAndLossValidation(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(100, 0.5, 106);
            Mdl = fitgtg(X, y);
            for value = {-1, Inf, 1i}
                w = ones(100, 1); w(7) = value{1};
                testCase.verifyError(@() fitgtg(X, y, 'Weights', w), ...
                    'RegressionQuantileGTG:InvalidWeights');
                testCase.verifyError(@() loss(Mdl, X, y, 'Weights', w), ...
                    'RegressionQuantileGTG:InvalidWeights');
                tbl = table(X(:, 1), y, w, 'VariableNames', {'x', 'y', 'w'});
                testCase.verifyError(@() fitgtg(tbl, 'y', 'Weights', 'w'), ...
                    'RegressionQuantileGTG:InvalidWeights');
                testCase.verifyError(@() loss(Mdl, X, y, 'LossFun', 'not-a-loss'), ...
                    'RegressionQuantileGTG:InvalidLossFun');
            end
            testCase.verifyError(@() loss(Mdl, X, y, 'Weights', ones(99, 1)), ...
                'RegressionQuantileGTG:SizeMismatch');
            testCase.verifyEqual(loss(Mdl, X, y, 'LossFun', "QUANTILE"), loss(Mdl, X, y));
        end

        function testInitialValueOptions(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(400, 0.5, 14);
            M1 = fitgtg(X, y);
            M2 = fitgtg(X, y, 'Bias', 2, 'Beta', [0; 0], 'ShapeLambda', 3, 'ShapeAlpha', 3);
            testCase.verifyEqual(M2.LogLikelihood, M1.LogLikelihood, 'AbsTol', 1e-6);
            testCase.verifyEqual(M2.ModelParameters.InitialBeta, [0; 0]);
            testCase.verifyError(@() fitgtg(X, y, 'Beta', [1 2 3]), 'RegressionQuantileGTG:InvalidOption');
        end

        function testIterationLimitAndHistory(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(400, 0.5, 15);
            testCase.verifyWarning(@() fitgtg(X, y, 'IterationLimit', 1, 'Solver', 'bfgs'), ...
                'RegressionQuantileGTG:NotConverged');
            warning('off', 'RegressionQuantileGTG:NotConverged');
            Mdl = fitgtg(X, y, 'IterationLimit', 1, 'Solver', 'bfgs');
            testCase.verifyEqual(Mdl.ConvergenceInfo.TerminationCode, 0);
            testCase.verifyEqual(Mdl.ConvergenceInfo.NumIterations, 1);
            M2 = fitgtg(X, y);
            hist = M2.ConvergenceInfo.History{1};
            testCase.verifyTrue(all(diff(hist.Objective) <= 0));
            testCase.verifyEqual(hist.Objective(end), M2.ConvergenceInfo.Objective, 'RelTol', 1e-14);
        end
    end

    %======================================================================
    % fitrqlinear-style interface
    %======================================================================
    methods (Test)
        function testTableInputSyntaxes(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(300, 0.5, 9);
            tbl = table(X(:, 1), X(:, 2), y, 'VariableNames', {'Logis', 'Bern', 'Resp'});
            MX = fitgtg(X, y, 'Quantiles', 0.3);
            MT = fitgtg(tbl, 'Resp', 'Quantiles', 0.3);
            testCase.verifyEqual(MT.Coefficients.Estimate, MX.Coefficients.Estimate, 'RelTol', 1e-8);
            testCase.verifyEqual(MT.PredictorNames, {'Logis', 'Bern'});
            testCase.verifyEqual(MT.ResponseName, 'Resp');
            testCase.verifyEqual(MT.Coefficients.Properties.RowNames, ...
                {'(Intercept)'; 'Logis'; 'Bern'; 'ShapeLambda'; 'log(ShapeAlpha)'});
            testCase.verifyEqual(predict(MT, tbl(1:4, :)), predict(MX, X(1:4, :)), 'RelTol', 1e-8);
            testCase.verifyEqual(loss(MT, tbl, 'Resp'), loss(MX, X, y), 'RelTol', 1e-8);
            % Formula and table with separate response
            MF = fitgtg(tbl, 'Resp ~ Logis + Bern', 'Quantiles', 0.3);
            testCase.verifyEqual(MF.Coefficients.Estimate, MX.Coefficients.Estimate, 'RelTol', 1e-8);
            MY = fitgtg(tbl(:, {'Logis', 'Bern'}), y, 'Quantiles', 0.3);
            testCase.verifyEqual(MY.Coefficients.Estimate, MX.Coefficients.Estimate, 'RelTol', 1e-8);
            % Predictor subset and formula without bias
            M1 = fitgtg(tbl, 'Resp ~ Logis', 'Quantiles', 0.3);
            testCase.verifyEqual(M1.PredictorNames, {'Logis'});
            M2 = fitgtg(tbl, 'Resp', 'PredictorNames', {'Logis'}, 'Quantiles', 0.3);
            testCase.verifyEqual(M2.Coefficients.Estimate, M1.Coefficients.Estimate, 'RelTol', 1e-8);
            M3 = fitgtg(tbl, 'Resp ~ Logis - 1');
            testCase.verifyFalse(M3.ModelParameters.FitBias);
            testCase.verifyError(@() fitgtg(tbl), 'RegressionQuantileGTG:MissingResponse');
        end

        function testCategoricalPredictors(testCase)
            rng(31, 'twister');
            n = 900;
            x = randn(n, 1);
            g = categorical(randi(3, n, 1), 1:3, {'a', 'b', 'c'});
            eta = 1 + 0.2*x + 0.3*(g == 'b') - 0.4*(g == 'c');
            y = exp(eta).*RegressionQuantileGTG.gtgrnd(1, 1, 2, n, 1);
            tbl = table(x, g, y);
            Mdl = fitgtg(tbl, 'y');
            testCase.verifyEqual(Mdl.CategoricalPredictors, 2);
            testCase.verifyEqual(Mdl.ExpandedPredictorNames, {'x', 'g_b', 'g_c'});
            D = [x, double(g == 'b'), double(g == 'c')];
            ref = fitgtg(D, y);
            testCase.verifyEqual(Mdl.Coefficients.Estimate, ref.Coefficients.Estimate, 'RelTol', 1e-8);
            testCase.verifyEqual(predict(Mdl, tbl(1:6, :)), predict(ref, D(1:6, :)), 'RelTol', 1e-8);
            % Numeric codes declared as categorical
            Mc = fitgtg([x double(g)], y, 'CategoricalPredictors', 2);
            testCase.verifyEqual(Mc.Coefficients.Estimate, ref.Coefficients.Estimate, 'RelTol', 1e-8);
            % Unknown level gives NaN prediction
            t1 = tbl(1, :); t1.g = categorical({'z'});
            testCase.verifyTrue(isnan(predict(Mdl, t1)));
        end

        function testObservationsInColumns(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(300, 0.5, 24);
            ref = fitgtg(X, y);
            Mdl = fitgtg(X', y, 'ObservationsIn', 'columns');
            testCase.verifyEqual(Mdl.Coefficients.Estimate, ref.Coefficients.Estimate, 'RelTol', 1e-10);
            testCase.verifyEqual(predict(Mdl, X(1:5, :)), predict(ref, X(1:5, :)), 'RelTol', 1e-10);
            testCase.verifyEqual(predict(Mdl, X(1:5, :)', 'ObservationsIn', 'columns'), ...
                predict(ref, X(1:5, :)), 'RelTol', 1e-10);
            testCase.verifyEqual(resubPredict(Mdl), resubPredict(ref), 'RelTol', 1e-10);
            testCase.verifyEqual(resubLoss(Mdl), resubLoss(ref), 'RelTol', 1e-10);
        end

        function testWeights(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(300, 0.5, 25);
            rng(2, 'twister');
            w = randi(3, 300, 1);
            Mw = fitgtg(X, y, 'Weights', w);
            idx = repelem((1:300)', w);
            Mr = fitgtg(X(idx, :), y(idx));
            testCase.verifyEqual(Mw.Coefficients.Estimate, Mr.Coefficients.Estimate, 'RelTol', 1e-5);
            % Weights are normalized to sum to the number of observations
            testCase.verifyEqual(Mw.LogLikelihood*sum(w)/300, Mr.LogLikelihood, 'RelTol', 1e-8);
            testCase.verifyEqual(Mw.W, w/sum(w), 'RelTol', 1e-12);
            % Zero weights exclude observations; table weight variable
            w0 = ones(300, 1); w0(1:10) = 0;
            M0 = fitgtg(X, y, 'Weights', w0);
            ref = fitgtg(X(11:end, :), y(11:end));
            testCase.verifyEqual(M0.Coefficients.Estimate, ref.Coefficients.Estimate, 'RelTol', 1e-8);
            tbl = table(X(:, 1), X(:, 2), w, y, 'VariableNames', {'a', 'b', 'w', 'y'});
            Mt = fitgtg(tbl, 'y', 'Weights', 'w');
            testCase.verifyEqual(Mt.PredictorNames, {'a', 'b'});
            testCase.verifyEqual(Mt.Coefficients.Estimate, Mw.Coefficients.Estimate, 'RelTol', 1e-8);
        end

        function testRegularization(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(400, 0.5, 26);
            X = [X, randn(400, 3)];
            M0 = fitgtg(X, y);
            M1 = fitgtg(X, y, 'Lambda', 0.01);
            M2 = fitgtg(X, y, 'Lambda', 1);
            testCase.verifyLessThan(norm(M1.Beta), norm(M0.Beta));
            testCase.verifyLessThan(norm(M2.Beta), norm(M1.Beta));
            testCase.verifyEqual(M1.Lambda, 0.01);
            testCase.verifyLessThan(M1.LogLikelihood, M0.LogLikelihood);
            % Stationarity of the penalized objective: -logL/N + Lambda/2*|Beta|^2
            th = [M1.Bias; M1.Beta; M1.ShapeLambda; log(M1.ShapeAlpha)];
            [~, g] = RegressionQuantileGTG.loglikelihood(th, X, y, 0.5, []);
            gp = -g/400 + 0.01*[0; M1.Beta; 0; 0];
            testCase.verifyLessThan(norm(gp, Inf), 1e-5);
            Ma = fitgtg(X, y, 'Lambda', 'auto');
            testCase.verifyEqual(Ma.Lambda, 1/400);
        end

        function testStandardize(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(400, 0.5, 27);
            X(:, 1) = 1000 + 100*X(:, 1);
            ref = fitgtg(X, y);
            Mdl = fitgtg(X, y, 'Standardize', true);
            testCase.verifyEqual(Mdl.Mu, mean(X), 'RelTol', 1e-12);
            testCase.verifyEqual(Mdl.Sigma, std(X, 1), 'RelTol', 1e-10);
            testCase.verifyEqual(Mdl.Beta, ref.Beta.*Mdl.Sigma', 'RelTol', 1e-5);
            testCase.verifyEqual(Mdl.LogLikelihood, ref.LogLikelihood, 'AbsTol', 1e-8);
            testCase.verifyEqual(predict(Mdl, X(1:5, :)), predict(ref, X(1:5, :)), 'RelTol', 1e-6);
            [~, ~, ci1] = predict(Mdl, X(1:5, :));
            [~, ~, ci0] = predict(ref, X(1:5, :));
            testCase.verifyEqual(ci1, ci0, 'RelTol', 1e-4);
        end

        function testResponseTransform(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(200, 0.5, 28);
            Mdl = fitgtg(X, y, 'ResponseTransform', @log);
            ref = fitgtg(X, y);
            testCase.verifyEqual(predict(Mdl, X(1:5, :)), log(predict(ref, X(1:5, :))), 'RelTol', 1e-12);
        end

        function testTruncatedGumbelModel(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(300, 0.5, 29);
            Mdl = fitgtg(X, y, 'ShapeAlpha', 1, 'FitShapeAlpha', false);
            testCase.verifyFalse(Mdl.FitShapeAlpha);
            testCase.verifyEqual(Mdl.ShapeAlpha, 1);
            testCase.verifyTrue(isnan(Mdl.Coefficients{'log(ShapeAlpha)', 'SE'}));
            testCase.verifyEqual(Mdl.ModelCriterion.AIC, -2*Mdl.LogLikelihood + 2*4, 'RelTol', 1e-12);
            full = fitgtg(X, y);
            testCase.verifyGreaterThan(full.LogLikelihood, Mdl.LogLikelihood);
        end

        function testInvalidInputs(testCase)
            testCase.verifyError(@() fitgtg([1; 2; -1]), 'RegressionQuantileGTG:InvalidY');
            testCase.verifyError(@() fitgtg([1; 2; 0]), 'RegressionQuantileGTG:InvalidY');
            testCase.verifyError(@() fitgtg([1; 2; Inf]), 'RegressionQuantileGTG:InvalidY');
            testCase.verifyError(@() fitgtg(rand(5, 1), rand(4, 1) + 1), 'RegressionQuantileGTG:SizeMismatch');
            testCase.verifyError(@() fitgtg(rand(5, 1), rand(5, 1) + 1, 'Weights', ones(4, 1)), ...
                'RegressionQuantileGTG:SizeMismatch');
            testCase.verifyError(@() fitgtg(rand(10, 1) + 1, 'Quantiles', 1.5), 'RegressionQuantileGTG:InvalidQuantiles');
            testCase.verifyError(@() fitgtg(rand(10, 1) + 1, 'FitBias', false), 'RegressionQuantileGTG:NoCoefficients');
            testCase.verifyError(@() fitgtg(rand(10, 2), rand(10, 1) + 1, 'PredictorNames', {'a'}), ...
                'RegressionQuantileGTG:InvalidPredictorNames');
            testCase.verifyError(@() fitgtg(rand(10, 1) + 1, 'NoSuchOption', 1), 'RegressionQuantileGTG:UnknownParameter');
            testCase.verifyError(@() fitgtg(rand(10, 1) + 1, 'Solver', 'sgd'), 'RegressionQuantileGTG:InvalidOption');
            testCase.verifyError(@() fitgtg(rand(10, 1) + 1, 'Lambda', -1), 'RegressionQuantileGTG:InvalidOption');
            testCase.verifyError(@() fitgtg(nan(10, 1)), 'RegressionQuantileGTG:NoData');
            Mdl = fitgtg(rand(50, 1), rand(50, 1) + 1);
            testCase.verifyError(@() predict(Mdl, rand(3, 2)), 'RegressionQuantileGTG:InvalidX');
            testCase.verifyError(@() predict(Mdl, rand(3, 1), 'Quantiles', 0), 'RegressionQuantileGTG:InvalidQuantiles');
        end

        function testExamplesWorkflow(testCase)
            figuresBefore = findall(groot, 'Type', 'figure');
            testCase.addTeardown(@() close(setdiff(findall(groot, 'Type', 'figure'), figuresBefore)));
            visibility = get(groot, 'DefaultFigureVisible');
            set(groot, 'DefaultFigureVisible', 'off');
            testCase.addTeardown(@() set(groot, 'DefaultFigureVisible', visibility));
            state = rng;
            results = examples_fitgtg;
            testCase.verifyEqual(rng, state);
            testCase.verifyEqual(numel(results.DietaryData), 315);
            testCase.verifyEqual(results.DietaryModel.LogLikelihood, -2672.896, 'AbsTol', 1e-3);
            testCase.verifySize(results.DietaryQuantiles, [3 2]);
            testCase.verifyEqual(results.BMIModel.NumObservations, ...
                height(results.BMITrainingData) - 4);
            testCase.verifyEqual(height(results.BMITestData), 625);
            testCase.verifyEqual(results.NumCrossingTestRows, 0);
            testCase.verifyTrue(all(results.BMIModel.ConvergenceInfo.TerminationCode > 0));
            testCase.verifyLessThan(abs(results.BMIValidation.WeightedTestCoverage - ...
                results.BMIValidation.Quantile), 0.08);
            testCase.verifyTrue(all(isfinite(results.BMIValidation.WeightedPinballLoss)));
            patients = results.PatientPredictions;
            testCase.verifyTrue(all(patients.MedianLower95 <= patients.MedianBMI & ...
                patients.MedianBMI <= patients.MedianUpper95));
            testCase.verifyTrue(all(patients.ProbabilityAbove30 >= 0 & patients.ProbabilityAbove30 <= 1));
            testCase.verifyLessThan(abs(patients.SimulatedProbabilityAbove30 - ...
                patients.ProbabilityAbove30), 0.03);
            testCase.verifyEqual(results.LargeBatchSummary.NumObservations, 50000);
            testCase.verifyEqual(results.LargeBatchSummary.Solver, "lbfgs");
            testCase.verifyEqual(results.LargeBatchSummary.RidgeStrength, 1/50000);
            testCase.verifyEmpty(results.CompactLargeModel.X);
            testCase.verifySize(results.LargeBatchPredictions, [10 3]);
            testCase.verifyTrue(all(isfinite(results.LargeBatchPredictions), 'all'));
            testCase.verifyTrue(all(diff(results.LargeBatchPredictions, 1, 2) > 0, 'all'));
            testCase.verifyEmpty(results.TallModel);
            testCase.verifyEqual(numel(results.Figures), 3);
            testCase.verifyTrue(all(isgraphics(results.Figures, 'figure')));
        end
    end

    %======================================================================
    % Large data
    %======================================================================
    methods (Test)
        function testChunkedEqualsDirect(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(500, 0.5, 30);
            y(7) = NaN;
            ref = fitgtg(X, y, 'Quantiles', [0.2 0.8]);
            Mdl = fitgtg(X, y, 'Quantiles', [0.2 0.8], 'ChunkSize', 37);
            testCase.verifyEqual(Mdl.Coefficients.Estimate, ref.Coefficients.Estimate, 'RelTol', 1e-10);
            testCase.verifyEqual(Mdl.Coefficients.SE, ref.Coefficients.SE, 'RelTol', 1e-8);
            testCase.verifyEqual(Mdl.LogLikelihood, ref.LogLikelihood, 'RelTol', 1e-12);
            Mc = fitgtg(X', y, 'ObservationsIn', 'columns', 'ChunkSize', 37, 'Quantiles', [0.2 0.8]);
            testCase.verifyEqual(Mc.Coefficients.Estimate, ref.Coefficients.Estimate, 'RelTol', 1e-10);
        end

        function testSinglePrecisionPredictors(testCase)
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(400, 0.5, 32);
            Xs = single(X);
            Mdl = fitgtg(Xs, y, 'ChunkSize', 100);
            ref = fitgtg(double(Xs), y);
            testCase.verifyEqual(Mdl.Coefficients.Estimate, ref.Coefficients.Estimate, 'RelTol', 1e-10);
            testCase.verifyClass(predict(Mdl, Xs(1:3, :)), 'double');
        end

        function testManyPredictorsUsesLbfgs(testCase)
            rng(33, 'twister');
            n = 3000; m = 120;
            X = randn(n, m);
            b = 0.05*randn(m, 1);
            y = exp(1 + X*b).*RegressionQuantileGTG.gtgrnd(1, 1, 2, n, 1);
            Mdl = fitgtg(X, y, 'GradientTolerance', 1e-7);
            testCase.verifyEqual(Mdl.Solver, 'lbfgs');
            testCase.verifyGreaterThan(Mdl.ConvergenceInfo.TerminationCode, 0);
            ref = fitgtg(X, y, 'Solver', 'newton');
            testCase.verifyEqual(Mdl.LogLikelihood, ref.LogLikelihood, 'AbsTol', 1e-4);
            testCase.verifyEqual(Mdl.Beta, ref.Beta, 'AbsTol', 1e-3);
        end

        function testTallArrays(testCase)
            testCase.assumeTrue(exist('tall', 'file') > 0, 'Tall arrays are not available.');
            mr = gcmr('nocreate');
            mapreducer(0);
            if ~isempty(mr)
                testCase.addTeardown(@() mapreducer(mr));
            end
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(400, 0.5, 34);
            g = categorical(X(:, 2), [0 1], {'no', 'yes'});
            tbl = table(X(:, 1), g, y, 'VariableNames', {'z1', 'g', 'y'});
            ref = fitgtg(tbl, 'y ~ z1 + g', 'Quantiles', [0.5 0.9]);
            tt = tall(tbl);
            Mdl = fitgtg(tt, 'y ~ z1 + g', 'Quantiles', [0.5 0.9]);
            testCase.verifyEmpty(Mdl.X);
            testCase.verifyEqual(Mdl.NumObservations, 400);
            testCase.verifyEqual(Mdl.Coefficients.Estimate, ref.Coefficients.Estimate, 'RelTol', 1e-4);
            testCase.verifyEqual(Mdl.LogLikelihood, ref.LogLikelihood, 'AbsTol', 1e-6);
            [YFit, ~, YCI] = predict(Mdl, tt);
            [YFit, YCI] = gather(YFit, YCI);
            [YRef, ~, YCIRef] = predict(ref, tbl);
            testCase.verifyEqual(YFit, YRef, 'RelTol', 1e-4);
            testCase.verifyEqual(YCI, YCIRef, 'RelTol', 1e-3);
            testCase.verifyEqual(loss(Mdl, tt, 'y'), loss(ref, tbl, 'y'), 'RelTol', 1e-4);
            % Numeric tall arrays
            Mn = fitgtg(tall(X), tall(y));
            refn = fitgtg(X, y);
            testCase.verifyEqual(Mn.Coefficients.Estimate, refn.Coefficients.Estimate, 'RelTol', 1e-4);
        end

        function testTallCorrectnessEdges(testCase)
            testCase.assumeTrue(exist('tall', 'file') > 0, 'Tall arrays are not available.');
            mr = gcmr('nocreate');
            mapreducer(0);
            if ~isempty(mr), testCase.addTeardown(@() mapreducer(mr)); end
            [X, y] = test_RegressionQuantileGTG.simulateScenario16(200, 0.5, 107);
            g = categorical(repmat({'a'}, 200, 1)); g(3) = categorical(missing);
            tbl = table(g, y);
            ref = fitgtg(tbl, 'y');
            Mdl = fitgtg(tall(tbl), 'y');
            testCase.verifyEqual(Mdl.NumObservations, 199);
            testCase.verifyEqual(Mdl.LogLikelihood, ref.LogLikelihood, 'AbsTol', 1e-6);
            bad = table(categorical(["a"; "unknown"; missing]), 'VariableNames', {'g'});
            [yp, ~, ci] = predict(Mdl, tall(bad));
            [yp, ci] = gather(yp, ci);
            testCase.verifyTrue(isfinite(yp(1)));
            testCase.verifyTrue(all(isnan(yp(2:3))));
            testCase.verifyTrue(all(isnan(ci(2:3, :)), 'all'));
            X(1, 1) = Inf;
            w = ones(200, 1); w(1) = 0; w(2) = NaN;
            Mn = fitgtg(X, y, 'Weights', w);
            testCase.verifyEqual(loss(Mn, tall(X), tall(y), 'Weights', tall(w)), ...
                resubLoss(Mn), 'AbsTol', 1e-10);
            w(5) = -1;
            testCase.verifyError(@() fitgtg(tall(X), tall(y), 'Weights', tall(w)), ...
                'RegressionQuantileGTG:InvalidWeights');
            testCase.verifyError(@() loss(Mn, tall(X), tall(y), 'Weights', tall(w)), ...
                'RegressionQuantileGTG:InvalidWeights');
            z = tall([0; 0.1; 1; Inf]);
            F = gather(RegressionQuantileGTG.gtgcdf(z, 1, -40, 2));
            testCase.verifyEqual(F, -expm1(-[0; 0.1; 1; Inf].^2), 'AbsTol', 1e-14);
        end
    end

    %======================================================================
    % Published results (Application 1, Table 5)
    %======================================================================
    methods (Test)
        function testBetadietTable5(testCase)
            x = test_RegressionQuantileGTG.loadBetadiet();
            testCase.assumeNotEmpty(x, 'Plasma Retinol data not available.');
            testCase.verifyNumElements(x, 315);

            % GTG model: beta = 290.83 (126.04), lambda = 2.61 (0.34),
            % alpha = 0.60 (0.08), LLF = -2672.896, AIC = 5351.792
            Mdl = fitgtg(x);
            [b, l, a] = gtgParameters(Mdl);
            testCase.verifyEqual(b, 290.83, 'AbsTol', 0.05);
            testCase.verifyEqual(l, 2.61, 'AbsTol', 0.005);
            testCase.verifyEqual(a, 0.60, 'AbsTol', 0.005);
            testCase.verifyEqual(Mdl.LogLikelihood, -2672.896, 'AbsTol', 1e-3);
            testCase.verifyEqual(Mdl.ModelCriterion.AIC, 5351.792, 'AbsTol', 2e-3);
            % Standard errors by the delta method
            C = Mdl.CoefficientCovariance;
            [h, dh] = RegressionQuantileGTG.hfun(l, Mdl.Quantiles);
            Jb = b*[1, -dh/(h*a), log(h)/a];
            testCase.verifyEqual(sqrt(Jb*C*Jb'), 126.04, 'RelTol', 0.01);
            testCase.verifyEqual(Mdl.Coefficients{'ShapeLambda', 'SE'}, 0.34, 'AbsTol', 0.005);
            testCase.verifyEqual(a*Mdl.Coefficients{'log(ShapeAlpha)', 'SE'}, 0.08, 'AbsTol', 0.005);

            % TG model (alpha = 1): beta = 1041.09, lambda = 1.46, LLF = -2682.618
            mTG = fitgtg(x, 'ShapeAlpha', 1, 'FitShapeAlpha', false);
            [bTG, lTG, aTG] = gtgParameters(mTG);
            testCase.verifyEqual(aTG, 1);
            testCase.verifyEqual(bTG, 1041.09, 'AbsTol', 0.1);
            testCase.verifyEqual(lTG, 1.46, 'AbsTol', 0.005);
            testCase.verifyEqual(mTG.LogLikelihood, -2682.618, 'AbsTol', 1e-3);
            testCase.verifyEqual(mTG.ModelCriterion.AIC, 5369.236, 'AbsTol', 2e-3);
        end

        function testQuartileStartBetadiet(testCase)
            % The quartile based initial point (Section 3.2) matches the
            % sample quartiles and is close to the ML estimates.
            x = test_RegressionQuantileGTG.loadBetadiet();
            testCase.assumeNotEmpty(x, 'Plasma Retinol data not available.');
            [prm, info] = RegressionQuantileGTG.gtgstart(x);
            testCase.verifyNotEmpty(info.QuartileSolutions);
            q = RegressionQuantileGTG.gtginv([0.25 0.5 0.75], info.QuartileSolutions(1, 1), ...
                info.QuartileSolutions(1, 2), info.QuartileSolutions(1, 3));
            testCase.verifyEqual(q, info.Quartiles, 'RelTol', 1e-6);
            testCase.verifyLessThan(RegressionQuantileGTG.gtglike(prm, x) - 2672.896, 5);
        end
    end

    %======================================================================
    % Helpers
    %======================================================================
    methods (Static)
        function [X, y] = simulateScenario16(n, p, seed)
            % Simulation scenario of Eq. (16) in [1]:
            % log(tau_i(p)) = 2.071 + 0.013*z1 + 0.034*z2, lambda = 3.881,
            % log(alpha) = 1.220, z1 ~ Logistic(0,1), z2 ~ Bernoulli(0.6).
            rng(seed, 'twister');
            u = rand(n, 1);
            z1 = log(u./(1 - u));
            z2 = double(rand(n, 1) < 0.6);
            tau = exp(2.071 + 0.013*z1 + 0.034*z2);
            h = @(q) RegressionQuantileGTG.hfun(3.881, q);
            y = tau.*(h(rand(n, 1))./h(p)).^(1/exp(1.220));
            X = [z1 z2];
        end

        function x = loadBetadiet()
            % BETADIET from the Plasma Retinol data set, or [] if unavailable.
            persistent cache
            if ~isempty(cache)
                x = cache;
                return
            end
            x = [];
            file = fullfile(fileparts(mfilename('fullpath')), 'Plasma_Retinol.txt');
            try
                if ~isfile(file)
                    websave(file, 'http://lib.stat.cmu.edu/datasets/Plasma_Retinol', ...
                        weboptions('Timeout', 30, 'ContentType', 'text'));
                end
                lines = splitlines(string(fileread(file)));
                vals = zeros(0, 14);
                for k = 1:numel(lines)
                    v = sscanf(lines(k), '%f')';
                    if numel(v) == 14
                        vals(end+1, :) = v; %#ok<AGROW>
                    end
                end
                x = vals(:, 11);
                cache = x;
            catch
            end
        end
    end
end
