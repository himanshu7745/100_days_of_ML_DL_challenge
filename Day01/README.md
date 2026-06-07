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

### NumPy Scratch Model

- MSE: `0.36309082577908286`
- RMS / RMSE: `0.6025701833`
- R2 Score: `0.6369091742209172`

### scikit-learn Model

- MSE: `4921881237.628148`
- RMS / RMSE: `70156.1204573639`
- R2 Score: `0.6400865688993735`

## Notes

- The scratch implementation is useful for understanding how gradient descent updates the parameters step by step.
- The scikit-learn version provides a clean baseline for comparison.
- The R2 values show that both approaches reach a similar level of explanatory power on this dataset.
