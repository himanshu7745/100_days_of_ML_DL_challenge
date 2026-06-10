# Day 04 — Shrinkage and Subset Selection

This directory contains notebooks and experiments demonstrating two families of approaches for controlling model complexity in linear models: shrinkage (regularization) and subset selection. The goals of this README are to: (1) explain the core concepts in depth, (2) give mathematical intuition and practical guidance, and (3) point to the notebooks that implement and illustrate these methods.

Notebooks
- [Shrinkage: ridge_regression.ipynb](Day04/Shrinkage/ridge_regression.ipynb)
- [Subset selection: best_subset_selection.ipynb](Day04/subset_selection/best_subset_selection.ipynb)
- [Subset selection: forward_stepwise_selection.ipynb](Day04/subset_selection/forward_stepwise_selection.ipynb)
- [Subset selection: backward_stepwise_selection.ipynb](Day04/subset_selection/backward_stepwise_selection.ipynb)

Overview
--------

When building predictive models, overfitting occurs when a model captures noise in the training data rather than the underlying signal. Two widely used strategies to reduce overfitting in linear models are:

- Shrinkage (regularization): keep all predictors but constrain coefficient sizes via a penalty.
- Subset selection: choose a smaller subset of predictors to include in the model.

Both approaches trade bias for variance and aim to reduce expected test error. Which approach to prefer depends on goals (interpretability vs predictive performance), the number of predictors $p$, and the underlying signal-to-noise ratio.

Shrinkage (Regularization)
--------------------------

Shrinkage methods add a penalty to the ordinary least squares (OLS) loss. For linear regression with design matrix $X$ and target $y$, OLS minimizes the residual sum of squares:

$$
L_{OLS}(\beta)=\|y - X\beta\|_2^2.
$$

Ridge regression adds an $L_2$ penalty:

$$
L_{ridge}(\beta)=\|y - X\beta\|_2^2 + \lambda \sum_{j=1}^p \beta_j^2.
$$

Closed-form solution (after centering and optionally standardizing predictors):

$$
\hat{\beta}_{ridge} = (X^{T}X + \lambda I)^{-1} X^{T}y.
$$

Key properties and intuition:
- The penalty shrinks coefficients toward zero; larger $\lambda$ -> more shrinkage.
- Ridge never sets coefficients exactly to zero (unless $\lambda\to\infty$) but reduces variance.
- Useful when predictors are many and/or highly collinear. The $\lambda I$ term stabilizes the inversion of $X^TX$.

Lasso (for comparison)
- Lasso uses an $L_1$ penalty: $\lambda \sum_j |\beta_j|$.
- Encourages sparsity: it can set some $\beta_j$ exactly to zero, performing variable selection.
- Not implemented in the Day04 notebooks, but conceptually important when comparing shrinkage vs subset selection.

Selecting $\lambda$
- Choose $\lambda$ via cross-validation (k-fold CV) to minimize estimated test error.
- Standardize predictors before applying ridge/lasso so the penalty treats features comparably.

Practical notes for Ridge
- Center the response and predictors. Standardize predictors (zero mean, unit variance) when features are on different scales.
- Interpret coefficient magnitudes with caution after shrinkage: coefficients are biased toward zero.
- Use CV to pick $\lambda$ and examine coefficient paths as $\lambda$ varies.

Subset Selection Methods
------------------------

These methods search for a subset of predictors that gives a good balance between fit and complexity.

1) Best Subset Selection
- Fit all possible models with up to $k$ predictors and choose the model with best performance (e.g., lowest CV error or lowest AIC/BIC) for each $k$.
- Computationally expensive: $\binom{p}{k}$ models; typically feasible only for small $p$ (e.g., $p \lesssim 30$) or with heuristics.

2) Forward Stepwise Selection
- Start with no predictors. Repeatedly add the predictor that yields the largest improvement in fit (e.g., reduces RSS the most), until a stopping rule is met.
- Greedy and much faster than best subset; may miss the globally optimal subset.

3) Backward Stepwise Selection
- Start with the full model and iteratively remove the least useful predictor until a stopping rule is met.
- Requires starting with a model of moderate dimensionality (cannot run backward when $p>n$ unless using regularization or screening first).

Model Selection Criteria
- Cross-Validation: directly estimates test error and is robust. Prefer k-fold CV for predictive performance.
- Adjusted $R^2$: penalizes model size mildly; useful for quick comparisons.
- AIC / BIC: information criteria that penalize complexity; BIC penalizes more strongly (favors smaller models when $n$ large).

Bias–Variance Tradeoff
----------------------

- Subset selection typically produces models with low variance (fewer predictors) but higher bias if informative features are dropped.
- Shrinkage reduces variance by shrinking coefficients but introduces bias toward zero; it often yields better predictive performance when many small effects exist.
- In practice, use CV to compare approaches directly on predictive metrics.

When to Use Which
------------------
- If interpretability with a small set of predictors is the priority, subset selection or Lasso are appropriate.
- If multicollinearity is a concern or there are many correlated predictors, ridge can stabilize estimates and improve predictive performance.
- If $p$ is large relative to $n$, prefer regularization (ridge/lasso/elastic net) over exhaustive best subset.

Practical Checklist
-------------------
- Always split data into training and test (or use nested CV) when doing model selection.
- Standardize predictors for shrinkage methods.
- Use CV to choose tuning parameters (e.g., $\lambda$) and to compare models.
- Inspect coefficient paths or inclusion frequencies across CV folds to understand stability.



References and Further Reading
------------------------------
- Hastie, Tibshirani, Friedman — "The Elements of Statistical Learning" (chapters on model selection and regularization).
- James, Witten, Hastie, Tibshirani — "An Introduction to Statistical Learning" (practical introductions and R examples).
