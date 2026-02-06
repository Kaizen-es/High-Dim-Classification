# High-Dimensional Classification with Fisher LDA
Gaussian classification in high-dimensional feature spaces, demonstrating why dimensionality reduction is essential when standard methods fail.

## Issue
Gaussian classifiers break in high-dimensional spaces. With 561 features, the normalization constant (2π)^(-561/2) ≈ 10^(-430) causes numerical underflow which is far below what computers can represent.
Regularization didn't help since this is a numerical not statistical issue.

## Solution
Fisher LDA was used to project the data to a lower-dimensional space while preserving class separation.
For 6-class classification, LDA reduces 561 dimensions to 5 dimensions. In 5D, (2π)^(-5/2) ≈ 0.06. Therefore, there is no underflow and classification works.

## Results
The HAR (Human Activity Recognition) and Wine Quality classification achieved an error rate of 2.37% and 56.94% respectively.

## Datasets
UCI HAR Dataset — 10,299 samples, 561 features, 6 activity classes
UCI Wine Quality — 4,898 samples, 11 features, 7 quality classes

## Future Work
Python implementation with scikit-learn
Comparison with PCA (unsupervised) vs LDA (supervised) dimensionality reduction

