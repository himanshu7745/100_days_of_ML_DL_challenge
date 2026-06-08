# Day 02 – Classification Models and Probabilistic Learning

## Overview

On Day 02 of my Machine Learning and Deep Learning journey, I explored classification algorithms based on probability theory and statistical modeling. The focus was on understanding how different models estimate class membership and make predictions using probabilistic principles.

The following algorithms were studied and implemented:

* Logistic Regression
* Linear Discriminant Analysis (LDA)
* Quadratic Discriminant Analysis (QDA)

All models were implemented from scratch using NumPy and compared against their Scikit-Learn counterparts.

---

## Topics Covered

### 1. Logistic Regression

#### Concepts Learned

* Binary Classification
* Sigmoid (Logistic) Function
* Log-Odds and Logit Function
* Maximum Likelihood Estimation (MLE)
* Binary Cross-Entropy Loss
* Gradient Descent Optimization

#### Key Formula

```math
\sigma(z)=\frac{1}{1+e^{-z}}
```

where

```math
z = w^T x + b
```

Binary Cross Entropy Loss:

```math
J=-\sum [y\log(p)+(1-y)\log(1-p)]
```

#### Implementation

Implemented Logistic Regression from scratch using:

* NumPy
* Gradient Descent
* Binary Cross Entropy Loss

#### Visualizations

* Training Loss Curve
* Confusion Matrix
* Accuracy Evaluation

---

### 2. Linear Discriminant Analysis (LDA)

#### Concepts Learned

* Generative Classification Models
* Bayes Theorem
* Gaussian Distribution Assumption
* Shared Covariance Matrix
* Linear Decision Boundary

#### Assumptions

* Features follow Gaussian Distribution
* Classes have different means
* Classes share a common covariance matrix

#### Discriminant Function

```math
\delta_k(x)=x^T\Sigma^{-1}\mu_k-\frac{1}{2}\mu_k^T\Sigma^{-1}\mu_k+\log(\pi_k)
```

#### Implementation

Implemented LDA from scratch by:

* Computing Class Means
* Computing Shared Covariance Matrix
* Computing Prior Probabilities
* Applying Discriminant Functions

#### Visualizations

* LDA Projection Histogram
* Confusion Matrix
* Accuracy Evaluation

---

### 3. Quadratic Discriminant Analysis (QDA)

#### Concepts Learned

* Class-Specific Covariance Matrices
* Quadratic Decision Boundaries
* Gaussian Class Modeling

#### Difference from LDA

LDA:

```math
\Sigma_0 = \Sigma_1
```

QDA:

```math
\Sigma_0 \neq \Sigma_1
```

This results in a quadratic decision boundary.

#### Discriminant Function

```math
\delta_k(x)=
-\frac{1}{2}\log|\Sigma_k|
-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)
+\log(\pi_k)
```

#### Implementation

Implemented QDA from scratch by:

* Computing Individual Covariance Matrices
* Computing Matrix Determinants
* Computing Inverse Covariance Matrices
* Evaluating Class Probabilities

#### Visualizations

* Confusion Matrix
* Accuracy Evaluation

---

## Dataset Used

### Breast Cancer Wisconsin Dataset

Features: 30

Classes:

* Malignant (0)
* Benign (1)

Dataset Source:

```python
from sklearn.datasets import load_breast_cancer
```

---

## Model Comparison

Models Evaluated:

1. Logistic Regression
2. Linear Discriminant Analysis (LDA)
3. Quadratic Discriminant Analysis (QDA)

Evaluation Metrics:

* Accuracy
* Confusion Matrix
* Visualization Analysis

---

## Graphs Generated

### Logistic Regression

* Binary Cross Entropy Loss Curve
* Confusion Matrix

### LDA

* LD1 Projection Histogram
* Confusion Matrix

### QDA

* Confusion Matrix

### Overall Comparison

* Accuracy Comparison Bar Chart

---

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 98.24%   |
| LDA                 | 95.61%   |
| QDA                 | 95.61%   |

---

## Key Learnings

* Logistic Regression directly models (P(y|x)).
* LDA models (P(x|y)) and assumes a shared covariance matrix.
* QDA models (P(x|y)) using separate covariance matrices.
* Maximum Likelihood Estimation forms the foundation of Logistic Regression.
* LDA and QDA are generative classifiers.
* Logistic Regression is a discriminative classifier.
* QDA can model more complex class boundaries than LDA.

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn



---

## Learning Journey

```text
Day 01 → Linear Regression
Day 02 → Logistic Regression, LDA, QDA\
```
