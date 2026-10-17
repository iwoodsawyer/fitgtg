# GTG Quantile Regression for MATLAB

Fit generalized truncated Gumbel (GTG) distributions and parametric quantile
regression models for **strictly positive responses**. The implementation
follows Gomez et al. (2024), *A New Generalization of the Truncated Gumbel
Distribution with Quantile Regression and Applications*.

The entry point, [`fitgtg`](fitgtg.m), returns a
[`RegressionQuantileGTG`](RegressionQuantileGTG.m) model with MATLAB-style
`predict`, `loss`, `residuals` and `compact` methods. Interfaces are inspired
by `fitrqlinear` and `RegressionQuantileLinear`; this is not a drop-in
replacement for those classes.

## Highlights

- GTG maximum-likelihood estimation and the truncated Gumbel (TG) special case.
- Numeric matrices, tables, additive formulas and categorical predictors.
- Multiple quantiles, conditional probabilities, densities and random responses.
- Observation weights, optional ridge regularization and predictor standardization.
- Analytic gradients and Hessians; Newton, BFGS and L-BFGS solvers.
- Chunked in-memory fitting and tall-array fitting and prediction.
- Coefficient tables, Wald confidence intervals, information criteria and
  convergence diagnostics.
- Regression tests against the paper's likelihood and BETADIET estimates,
  plus numerical, interface and large-data checks.

## Installation and requirements

Download or clone this folder, then add it to your MATLAB path:

```matlab
addpath('C:\Source\fitgtg')   % Replace with your installation folder.
help fitgtg
help RegressionQuantileGTG
```

For MATLAB File Exchange, download/unzip the submission or install it using
the available installation interface, then ensure the folder is on the path.
There is no compilation or external R dependency.

- **Validated release:** MATLAB R2025b. Earlier releases have not been certified.
- The optimizers are implemented here and do not call Optimization Toolbox solvers.
- The existing [`demo_fitgtg.m`](demo_fitgtg.m) uses Statistics and Machine
  Learning Toolbox functions for its Weibull comparison.
- Optional tall workflows require support for the tall operations used here.
  Parallel execution additionally depends on your MATLAB installation and
  Parallel Computing Toolbox configuration. Serial examples can use `mapreducer(0)`.

## Quick start

```matlab
rng default
n = 1000;
X = randn(n,2);
Y = exp(1 + X*[0.3; -0.2]) .* ...
    RegressionQuantileGTG.gtgrnd(1,2,2,n,1);

Mdl = fitgtg(X,Y,'Quantiles',[0.1 0.5 0.9]);
[YFit,crossing,YCI] = predict(Mdl,X(1:5,:));
disp(Mdl.Coefficients)
```

`YFit` is 5-by-3, with columns in the requested quantile order. `crossing`
is a logical indicator per observation. `YCI` is 5-by-2-by-3: lower/upper
confidence bounds for each estimated conditional quantile.

**These are confidence intervals for the quantile curve, not individual
prediction intervals.** For example, the fitted 0.1 and 0.9 quantiles form
a model-based central 80% interval for a future response, without separately
accounting for parameter-estimation uncertainty.

## Realistic examples and plots

Run the self-contained workflows in [`examples_fitgtg.m`](examples_fitgtg.m):

```matlab
results = examples_fitgtg;
disp(results.BMIValidation)
disp(results.PatientPredictions)

% Optional serial tall-table version of the same BMI fit:
mapreducer(0)
resultsTall = examples_fitgtg('IncludeTall',true);
```

The function restores the caller's random-number generator state and makes
no downloads or file writes. It returns models, data, evaluation tables and
three figure handles.

| Workflow | Data | Capabilities |
|---|---|---|
| Dietary beta-carotene intake | 315 measured observations in the bundled StatLib dataset | GTG/TG comparison, AIC/BIC, quantiles, density and CDF plots |
| BMI screening | Clearly labeled simulated data inspired by the paper's Table 6 | Held-out evaluation, categorical predictors, hypothetical sampling weights, missing values, confidence bands, threshold probabilities, simulation and residual diagnostics |
| Larger screening batch | 50,000 simulated observations | Chunked fitting, L-BFGS, automatic ridge strength and compact prediction |
| Optional BMI tall-table fit | The same simulated training table | Tall fitting/prediction and comparison with the in-memory model |

The simulated BMI workflows are software demonstrations, **not validated
clinical models or evidence of causal effects**. The example weights do not
provide complex-survey variance estimation.

The original [`demo_fitgtg.m`](demo_fitgtg.m) also illustrates distribution
shapes, published fits and a one-million-row simulation.

## Supported fitting forms

```matlab
Mdl = fitgtg(X,Y);
Mdl = fitgtg(Tbl,'ResponseVariable');
Mdl = fitgtg(Tbl,'ResponseVariable ~ x1 + x2');
Mdl = fitgtg(Tbl,Y);
Mdl = fitgtg(Y);                         % Distribution fit, no predictors.
Mdl = fitgtg(X.',Y,'ObservationsIn','columns');
```

Formulas support additive terms and `- 1` or `+ 0` to omit the intercept.
Interactions, polynomial terms and formula shorthand such as `.` are not
implemented; construct those predictor columns explicitly.

Responses and weights can be row or column vectors. Table variables named
as responses or weights are excluded from the default predictor set.
Categorical predictors use their first encoded training level as reference
(category order for categorical arrays, sorted levels for numeric codes).
See `ExpandedPredictorNames` to identify the resulting columns.

### Important options

| Option | Default | Meaning |
|---|---|---|
| `Quantiles` | `0.5` | Distinct probabilities strictly between zero and one |
| `FitBias` | `true` | Include an intercept |
| `ShapeLambda` | Estimated starting value | Initial GTG shape parameter lambda |
| `ShapeAlpha` | Estimated starting value | Initial alpha, or fixed alpha if `FitShapeAlpha=false` |
| `FitShapeAlpha` | `true` | Estimate alpha |
| `Lambda` | `0` | Ridge strength; `'auto'` uses `1/NumObservations` |
| `Weights` | Ones | Numeric vector or table variable name |
| `Standardize` | `false` | Report coefficients for standardized predictors |
| `CategoricalPredictors` | Automatic for nonnumeric table variables | Indices, names, logical mask or `'all'` |
| `Solver` | `'auto'` | Newton for at most 100 fitted parameters, otherwise L-BFGS |
| `ChunkSize` | `'auto'` | In-memory row-block size |
| `IterationLimit` | `1000` | Maximum solver iterations |
| `GradientTolerance` | `1e-6` | Gradient infinity-norm stopping tolerance |
| `BetaTolerance` | `1e-6` | Relative parameter-step stopping tolerance |
| `HessianHistorySize` | `15` | L-BFGS history length |

See `help fitgtg` for initial coefficient values, naming, response transforms
and other options. **`Lambda` is ridge regularization, not the distribution's
shape lambda; use `ShapeLambda` for the latter.**

To fit the TG special case:

```matlab
MdlTG = fitgtg(Y,'ShapeAlpha',1,'FitShapeAlpha',false);
```

## Prediction, evaluation and model inspection

```matlab
YFit = predict(Mdl,Xnew);
Y75 = predict(Mdl,Xnew,'Quantiles',0.75); % Need not be a fitted quantile.
L = loss(Mdl,Xtest,Ytest);               % Mean pinball loss per quantile.
Ltrain = resubLoss(Mdl);
R = residuals(Mdl);                     % Normalized quantile residuals.
CI = coefCI(Mdl);                       % Coefficient Wald intervals.
[beta,lambda,alpha] = gtgParameters(Mdl,Xnew);
F = cdf(Mdl,Xnew,Ynew);
f = pdf(Mdl,Xnew,Ynew);
Ysim = random(Mdl,Xnew);
CMdl = compact(Mdl);
```

For a response-only fit, `predict(Mdl)` returns one quantile row. To specify
other probabilities, use `predict(Mdl,[],'Quantiles',[0.25 0.75])`.

Useful properties include `Beta`, `Bias`, `ShapeLambda`, `ShapeAlpha`,
`Coefficients`, `CoefficientCovariance`, `LogLikelihood`, `ModelCriterion`,
`ConvergenceInfo`, `RowsUsed`, and `Mu`/`Sigma` when standardization is enabled.
Coefficient intervals use `[Bias; Beta; ShapeLambda; log(ShapeAlpha)]`, with
the intercept omitted when `FitBias=false`. Fixed alpha has no estimated SE.

The static distribution utilities are `gtgpdf`, `gtgcdf`, `gtginv`, `gtgrnd`,
`gtgstat`, `gtglike` and `gtgstart`.

## Model assumptions and interpretation

The fitted relationship is

```text
log(Q_p(Y | x)) = Bias(p) + x * Beta(p)
```

The conditional distribution is GTG with positive scale, real-valued
`ShapeLambda`, and positive `ShapeAlpha`. Alpha is optimized on its log scale.
The shape parameters are constant across observations.

This is **parametric maximum-likelihood quantile regression**, not direct
pinball-loss minimization. Pinball loss is supplied as an evaluation measure.
With an unpenalized free intercept and the same fitted distribution,
changing the reference quantile changes the intercept but not the slopes
or shapes. This permits exact distribution-preserving warm starts across
quantiles. Without a free intercept, that equivalence need not hold.

The minimized objective is

```text
-weightedLogLikelihood / NumObservations + Lambda/2 * sum(Beta.^2)
```

Weights are normalized to sum to the number of retained observations.
Coefficient covariance uses the final objective Hessian; with ridge
regularization it includes the penalty curvature. Wald intervals are
asymptotic approximations, not bootstrap or survey-design intervals.
Likelihood-based criteria count fitted parameters and are most directly
interpretable for unregularized fits.

### Missing values and convergence

- Responses in retained rows must be positive and finite.
- Missing responses, invalid numeric predictors, and missing categorical
  predictors are excluded from fitting.
- Zero or NaN weights exclude observations; negative, infinite or nonreal
  weights are errors.
- Unknown categorical levels produce NaN predictions, including for
  predictors that had only one observed training level.
- Training residuals are NaN for excluded rows.
- Inspect `ConvergenceInfo.TerminationCode`, `GradientNorm` and
  `TerminationStatus`; warnings should not be ignored.
- Near the Weibull limit (`ShapeLambda` tends to negative infinity), shape
  identifiability and Hessian-based standard errors can be unreliable.

## Large datasets

Numeric in-memory predictor data are processed in row chunks rather than
copied into a complete standardized design matrix. Table/categorical
encoding can still allocate expanded predictor columns. The training data
remain in an ordinary fitted model; use `compact` to remove them.

For data already exposed through a MATLAB datastore:

```matlab
ds = tabularTextDatastore('screening.csv'); % Your own data file(s).
tt = tall(ds);
MdlTall = fitgtg(tt,'BMI ~ Waist','Quantiles',[0.1 0.5 0.9]);
YFitTall = predict(MdlTall,tt);
YFit = gather(YFitTall);                  % Only gather what fits in memory.
```

Tall fitting gathers small reductions, not the entire dataset. Tall
initialization uses approximate residual quantiles, so results can differ
slightly from in-memory fits. Tall models do not retain training data:
evaluate with explicit predictors and responses instead of `resubPredict`,
`resubLoss` or training-only `residuals`.

Scaling to many **rows** does not imply unlimited scaling to many
**predictors**. The final covariance calculation forms a dense Hessian;
coefficient transforms and covariance storage also require quadratic
memory in the expanded predictor count. L-BFGS avoids repeated Hessians
during optimization but does not remove these final costs.

## Testing

```matlab
results = runtests('test_RegressionQuantileGTG');
assert(all([results.Passed]))
```

Tests cover densities/CDFs, support boundaries, the Weibull limit, analytic
derivatives, solver agreement, paper estimates, weights, regularization,
standardization, missing values, row/column orientations and tall arrays.
Optional tall tests are skipped when the necessary capability is unavailable.

## Scope and limitations

- `CrossVal`, `OptimizeHyperparameters`, `UseParallel` and
  `PredictionForMissingValue` are not implemented.
- Data splitting and regularization selection must be performed explicitly.
- Arbitrary response transforms affect prediction output, not the fitted
  GTG distribution. Probability/density methods use the original response scale.
- This folder contains no assigned software license. Before publishing a
  GitHub or File Exchange release, choose a code license and verify permission
  to redistribute the accompanying data and article PDF. The bundled data
  header explicitly says **"Authorization: Contact Authors"**.

## Citation and data attribution

Gomez, H. J.; Santoro, K. I.; Ayma, D.; Cortes, I. E.; Gallardo, D. I.;
Magalhaes, T. M. (2024). *A New Generalization of the Truncated Gumbel
Distribution with Quantile Regression and Applications*. Mathematics,
12(11), 1762. https://doi.org/10.3390/math12111762

The measured example uses BETADIET (column 11) from the
[StatLib Plasma Retinol dataset](https://lib.stat.cmu.edu/datasets/Plasma_Retinol),
provided in [`Plasma_Retinol.txt`](Plasma_Retinol.txt). Its header describes
315 participants and cites Nierenberg et al. (1989), *Determinants of plasma
levels of beta-carotene and retinol*, American Journal of Epidemiology,
130:511-521. Dietary intake and plasma concentrations are different variables.

### Suggested File Exchange summary

**Title:** Generalized Truncated Gumbel Quantile Regression

**Description:** MATLAB implementation of GTG distribution fitting and
parametric quantile regression for positive responses, with table/formula
interfaces, categorical predictors, weighted likelihood, ridge
regularization, multiple conditional quantiles, confidence intervals,
distribution utilities, chunked computation and tall-array support.
Includes regression tests and real-data/simulated examples with plots.

**Suggested keywords:** quantile regression, generalized truncated Gumbel,
positive response, maximum likelihood, distribution fitting, tall arrays,
biostatistics.
