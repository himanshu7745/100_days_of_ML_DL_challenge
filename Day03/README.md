# Day 03 — Resampling & Validation

This folder contains notebooks that explore resampling methods and model validation strategies commonly used in statistical learning and machine learning: the bootstrap, validation set, k-fold cross-validation, and leave-one-out cross-validation (LOOCV).

## 1 — Bootstrap: idea and mathematics

The bootstrap is a general-purpose, computer-intensive method for estimating the sampling distribution of a statistic when the analytic form is difficult or unknown.

Let the data be $\mathcal{D}=\{x_1,\dots,x_n\}$ and let $\hat\theta = s(\mathcal{D})$ be a statistic of interest (mean, median, model coefficient, prediction error, etc.). The bootstrap approximates the sampling distribution of $\hat\theta$ by resampling from the empirical distribution $\hat F$ that places mass $1/n$ on each observed point.

Algorithm (nonparametric bootstrap):
1. For $b=1,\dots,B$:
   - Draw a bootstrap sample $\mathcal{D}^*_b$ of size $n$ by sampling with replacement from $\mathcal{D}$.
   - Compute $\hat\theta^*_b = s(\mathcal{D}^*_b)$.
2. Use the empirical distribution of $\{\hat\theta^*_b\}_{b=1}^B$ to estimate variance, standard error, and confidence intervals.

Bootstrap standard error estimate:
$$\widehat{\mathrm{SE}}_{\text{boot}}(\hat\theta) = \sqrt{\frac{1}{B-1}\sum_{b=1}^B\big(\hat\theta^*_b - \bar{\hat\theta}^*\big)^2},$$
where $\bar{\hat\theta}^* = \frac{1}{B}\sum_b\hat\theta^*_b$.

Basic percentile confidence interval (approximate):
- Order the bootstrap replicates and take quantiles. For a two-sided $(1-\alpha)$ CI, use the $\alpha/2$ and $1-\alpha/2$ empirical quantiles of $\{\hat\theta^*_b\}$.

Bias-corrected and accelerated (BCa) intervals provide improved coverage by adjusting for bias and skewness; see references below. The BCa interval requires estimating the bias-correction constant $z_0$ and acceleration $a$.

Practical notes:
- Choose $B$ large enough (commonly 1000–10000) depending on desired accuracy and compute budget.
- For complex statistics (e.g., medians, quantiles, non-smooth functions), bootstrap often performs well where analytic approximations fail.
- For dependent data (time series), use block bootstrap variants.

## 2 — Cross-validation: validation set, k-fold, and LOOCV

Cross-validation estimates the expected generalization error of a learning algorithm by training and testing on multiple splits of the data.

Validation set method:
- Split the data into a training set (size $n_{\text{train}}$) and a validation set (size $n_{\text{val}}$).
- Train the model on the training set and evaluate on the validation set. This gives a single estimate of test error but is high-variance (sensitive to the split).

k-fold cross-validation:

- Partition data into \(k\) roughly equal folds.
- For each fold \(j = 1,\ldots,k\), train on the other \(k-1\) folds and evaluate on fold \(j\).

The CV estimate is the average error across folds:

$$
\widehat{\mathrm{Err}}_{\text{k-fold}}
=
\frac{1}{k}
\sum_{j=1}^{k}
\mathrm{Err}^{(j)}
$$

- Common choices are \(k=5\) or \(k=10\), which balance bias, variance, and computational cost.
- Small \(k\) generally leads to higher bias and lower variance.
- Large \(k\) generally leads to lower bias but higher variance and computational expense.

Leave-one-out CV (LOOCV): special case $k=n$ where each test fold is a single observation. LOOCV has low bias but can have high variance and is computationally expensive for large $n$.

Relationship to bias–variance:
- Cross-validation error is an (approximate) unbiased estimate of generalization error for model selection, but it can be noisy. Using more folds reduces bias but may increase variance.

Choosing between bootstrap and CV:
- Use CV for model selection and estimating predictive performance.
- Use bootstrap to estimate standard errors/uncertainty of a statistic or when analytic variance formulas are unavailable.

## 3 — Mathematical examples

Example: bootstrap standard error for sample mean $\bar X$.
- Statistic: $\hat\theta = \bar X = \frac{1}{n}\sum_{i=1}^n x_i$.
- The bootstrap replicates $\bar X^*_b$ approximate the sampling distribution of $\bar X$. For IID data, bootstrap SE converges to the true SE $\sigma/\sqrt{n}$ as $n,B\to\infty$.

Example: k-fold CV estimate of mean-squared error for regression.
- For fold $j$ with test indices $T_j$, compute MSE_j = (1/|T_j|) \sum_{i\in T_j} (y_i - \hat y_i^{(-j)})^2.
- Average over folds to get the final CV MSE.

## 4 — Implementation recipes (Python)

Bootstrap (simple NumPy example):

```python
import numpy as np

def bootstrap_stat(data, stat_func, B=2000, random_state=None):
    rng = np.random.default_rng(random_state)
    n = len(data)
    reps = np.empty(B)
    for b in range(B):
        sample = rng.choice(data, size=n, replace=True)
        reps[b] = stat_func(sample)
    se = reps.std(ddof=1)
    return reps, se

# Example: bootstrap mean
# data = np.array([...])
# reps, se = bootstrap_stat(data, np.mean, B=2000)
```

k-fold cross-validation (scikit-learn):

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

X, y = ...
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse = -scores.mean()
```

LOOCV using scikit-learn: set `cv=n` or use `LeaveOneOut()` from `sklearn.model_selection`.

## 5 — Practical tips and pitfalls
- Always randomize/shuffle data before splitting, unless the data are time-ordered.
- For classification tasks, use stratified k-fold to preserve class proportions in each fold.
- When tuning hyperparameters, nest cross-validation: use inner CV for hyperparameter selection and outer CV for performance estimation.
- Beware of data leakage: do preprocessing (scaling, feature selection) inside each CV fold, not before splitting.
- For small datasets, LOOCV can be useful, but note the variance and computational cost.
- When reporting results, include not only point estimates (CV error) but also variability (e.g., standard error across folds or bootstrap CI).
