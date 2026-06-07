# Day 01 - California Housing Linear Regression

This day focuses on building and comparing linear regression models for the California housing dataset.

## What I Implemented

1. A NumPy-based linear regression from scratch.
	- Model training is done with gradient descent.
	- The notebook also evaluates the fitted model with error and goodness-of-fit metrics.
2. A scikit-learn linear regression baseline.
	- Uses `LinearRegression` as the reference implementation.
	- The workflow follows the closed-form linear regression approach used by scikit-learn.
	- The model is evaluated with the same metrics for comparison.

## Dataset

The project uses the California housing dataset stored in:

- `dataset/housing.csv.csv`

## Notebooks

- `projects/california_housing_scratch.ipynb`
- `projects/califronia_housing_scikit_learn.ipynb`

## Workflow

Both notebooks follow the same basic pipeline:

1. Load the housing data.
2. Split the data into training and testing sets.
3. Scale features where needed.
4. Train the regression model.
5. Evaluate the predictions.

## Metrics

The models are evaluated using:

- RMS / RMSE
- R2 score

## Performance Comparison

The Multiple Linear Regression model was implemented from scratch using NumPy and optimized using Gradient Descent. Its performance was compared with Scikit-Learn's `LinearRegression` model on the California Housing dataset.

| Model | R² Score |
|---|---:|
| Custom NumPy Implementation | 0.6109 |
| Scikit-Learn LinearRegression | 0.6401 |

The custom implementation achieved an R² score of 0.6109, indicating that approximately 61.09% of the variance in house prices was explained by the model. Scikit-Learn's implementation achieved a slightly higher R² score of 0.6401.

The performance difference is primarily due to the optimization methods used by the two approaches. The custom implementation relies on Gradient Descent, which iteratively updates model parameters and may not reach the exact optimal solution within a limited number of epochs. In contrast, Scikit-Learn's `LinearRegression` uses an efficient least-squares solver based on numerical linear algebra techniques such as Singular Value Decomposition (SVD), enabling it to obtain the optimal coefficients directly.

Despite the small performance gap, the custom implementation successfully demonstrates the underlying mathematics of Multiple Linear Regression, including feature normalization, gradient computation, parameter updates, loss minimization, and model evaluation. The results obtained are close to those of the optimized library implementation, validating the correctness of the algorithm.

## Notes

- The scratch implementation is useful for understanding how gradient descent updates the parameters step by step.
- The scikit-learn version provides a clean baseline for comparison.
- The R2 values show that both approaches reach a similar level of explanatory power on this dataset.
