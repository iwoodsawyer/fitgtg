function Mdl = fitgtg(varargin)
%FITGTG Fit generalized truncated Gumbel (GTG) quantile regression model.
%   Mdl = FITGTG(Tbl,ResponseVarName) returns a RegressionQuantileGTG
%   model fitted to the predictors in table Tbl and the positive response
%   Tbl.ResponseVarName. The reparameterized GTG (RGTG) quantile
%   regression model of Gomez et al. (2024),
%
%      log(Q_p(Y|x)) = Bias + x*Beta,   Y|x ~ GTG(beta(x), lambda, alpha),
%
%   is fitted by maximum likelihood for the median (p = 0.5).
%
%   Mdl = FITGTG(Tbl,formula) uses a formula string 'Y ~ x1 + x2 + ...'
%   to select the response and predictors. Add '- 1' to drop the bias.
%
%   Mdl = FITGTG(Tbl,Y) uses all variables of Tbl as predictors and the
%   vector Y as response.
%
%   Mdl = FITGTG(X,Y) uses the numeric predictor matrix X (N-by-P, or
%   P-by-N with 'ObservationsIn','columns') and the response vector Y.
%   Row-vector responses and weights are converted to column vectors.
%
%   Mdl = FITGTG(Y) fits the GTG(beta,lambda,alpha) distribution to Y
%   (model without predictors). Use GTGPARAMETERS(Mdl) to obtain the
%   estimates of beta, lambda and alpha.
%
%   X, Y and Tbl can be tall arrays. Then all computations are reductions
%   over the data, which are evaluated with a few passes per iteration.
%   For in-memory data, the computations are done in chunks of rows
%   ('ChunkSize'), so that no copies of large predictor matrices are
%   made. Rows with missing values are not used, including missing values
%   in categorical predictors with only one observed level. Unknown
%   categorical levels in prediction data produce NaN. Y must be positive.
%
%   Mdl = FITGTG(...,Name,Value) specifies options using one or more
%   name-value arguments:
%
%   Model:
%      'Quantiles'          - Vector of probabilities in (0,1). A model is
%                             fitted for each quantile. Default: 0.5.
%      'FitBias'            - Include a bias (intercept) term.
%                             Default: true.
%      'FitShapeAlpha'      - Estimate the shape parameter alpha. Use
%                             'ShapeAlpha',1,'FitShapeAlpha',false to fit
%                             the truncated Gumbel (TG) model.
%                             Default: true.
%      'Lambda'             - Ridge (L2) penalty strength on Beta (not on
%                             Bias or shape parameters): the objective is
%                             -logL/N + Lambda/2*sum(Beta.^2). "auto"
%                             uses 1/N. Default: 0 (maximum likelihood).
%      'Weights'            - Real, nonnegative, finite weights (vector
%                             or name of a table variable). Zero and NaN
%                             weights exclude observations. Negative and
%                             infinite weights are rejected. Weights are
%                             normalized to sum to the number of observations.
%                             Default: ones.
%
%   Data:
%      'CategoricalPredictors' - Indices, logical vector, names or "all".
%                             Categorical predictors are dummy coded with
%                             the first level as reference. Default:
%                             non-numeric table variables.
%      'ObservationsIn'     - "rows" (default) or "columns" for X.
%      'PredictorNames'     - Names of the predictors (or the predictor
%                             variables of Tbl).
%      'ResponseName'       - Name of the response. Default: "Y".
%      'ResponseTransform'  - Transformation of predicted responses:
%                             "none" (default) or a function handle.
%      'Standardize'        - Standardize the predictors. Beta then
%                             applies to (X-Mu)./Sigma. Default: false.
%      'ChunkSize'          - Number of rows per chunk for in-memory
%                             data. Default: "auto".
%
%   Initial values:
%      'Beta', 'Bias'       - Initial linear coefficients and bias.
%      'ShapeLambda'        - Initial value of lambda.
%      'ShapeAlpha'         - Initial (or, with 'FitShapeAlpha',false,
%                             fixed) value of alpha.
%                             Default: least squares on log(Y) and the
%                             quartile estimator of [1] on the residuals.
%
%   Optimization:
%      'Solver'             - "auto" (default), "newton", "bfgs" or
%                             "lbfgs". "auto" uses "newton" for up to 100
%                             parameters and "lbfgs" otherwise.
%      'IterationLimit'     - Maximum number of iterations. Default: 1000.
%      'GradientTolerance'  - Tolerance on the gradient (infinity norm).
%                             Default: 1e-6.
%      'BetaTolerance'      - Tolerance on the relative change of the
%                             coefficients. Default: 1e-6.
%      'HessianHistorySize' - History size of "lbfgs". Default: 15.
%      'Verbose'            - 0 (default) or 1 to display iterations.
%
%   Example: Median and 90% quantile regression
%      rng default
%      n = 500;
%      waist = 97 + 13*randn(n,1);
%      sex = categorical(rand(n,1) < 0.6, [false true], {'male','female'});
%      tau = exp(2.076 + 0.013*waist + 0.035*(sex == 'female'));
%      e = RegressionQuantileGTG.gtgrnd(1, 3.5, exp(1.3), n, 1);
%      bmi = tau.*e/RegressionQuantileGTG.gtginv(0.5, 1, 3.5, exp(1.3));
%      tbl = table(waist, sex, bmi);
%      Mdl = fitgtg(tbl, 'bmi ~ waist + sex', 'Quantiles', [0.5 0.9])
%      YFit = predict(Mdl, tbl(1:5,:))
%
%   Reference:
%      [1] H. J. Gomez, K. I. Santoro, D. Ayma, I. E. Cortes,
%          D. I. Gallardo and T. M. Magalhaes, "A New Generalization of
%          the Truncated Gumbel Distribution with Quantile Regression and
%          Applications", Mathematics, 12(11), 1762, 2024.
%
%   See also REGRESSIONQUANTILEGTG, EXAMPLES_FITGTG, DEMO_FITGTG,
%            FITRQLINEAR, TALL.

Mdl = RegressionQuantileGTG.fit(varargin{:});
end
