# Breast Cancer Classification and Evaluation
Breast cancer classification and evaluation of classification algorithms using k-fold Cross-Validation.

## Data set
The data set used is [Wisconsin Breast Cancer (Original) Data Set](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original) by **UC Irvine Machine Learning Repository**.

## Classification algorithms
- Discriminant Analysis

- K-Nearest Neighbors

- Naive Bayes

- Support Vector Machine

- Decision Tree

## Evaluation
| Classification Algorithm | Accuracy | Sensitivity | Specificity |
| :------------: | :------------: | :------------: | :------------: |
| **Discriminant Analysis** (Linear) | 0.959943 | 0.978166 | 0.925311 |
| **Discriminant Analysis** (Mahalanobis) | 0.899857 | 0.847162 | 1.000000 |
| **K-Nearest Neighbor** (NumNeighbors = 5) | 0.965665 | 0.971616 | 0.954357 |
| **K-Nearest Neighbor** (NumNeighbors = 25) | 0.962804 | 0.978166 | 0.933610 |
| **Naive Bayes** (Gaussian Distribution) | 0.959943 | 0.951965 | 0.975104 |
| **Naive Bayes** (Kernel Distribution) | 0.964235 | 0.971616 | 0.950207 |
| **Support Vector Machine** (BoxConstraint = 1) | 0.967096 | 0.973799 | 0.954357 |
| **Support Vector Machine** (BoxConstraint = 10) | 0.962804 | 0.967249 | 0.954357 |
| **Decision Tree** (AlgorithmForCategorical = Exact) | 0.928469 | 0.941048 | 0.904564 |
| **Decision Tree** (AlgorithmForCategorical = PCA) | 0.942775 | 0.956332 | 0.917012 |