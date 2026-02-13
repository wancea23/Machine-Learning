# Changelog

## [Unreleased]

## 2026-01-18
### Notion Journaling
- Added K-Means++ algorithm in Python.
- Added the code to documentation for the project.
- Improved and modified documentation.

## 2026-01-20
### K-Means++ Implementation Documented
- Full algorithm breakdown with step-by-step explanations in Notion
- Key insights documented:
  - Probabilistic centroid initialization (roulette-wheel selection)
  - Why farthest points have higher but not guaranteed probability
  - Cumulative probability scanning mechanism
  - Convergence logic and movement threshold
- Visual examples created for probability distributions and cluster evolution

## 2026-01-23
- Played, explained and displayed examples of usage of K-Means Algorithms

## 2026-01-25
### Supervised Learning Documented
- Explained the introduction of the supervised learning
- Explained what is Linear Regression
- Used and exaplined mathematical terms

## 2026-01-28
- Explained what is Gradient Descent
- Exlpained on a deeper level the math behind the Gradient Descent Convergence

## 2026-02-4
- Explained what are Hyperparameters in Linear Regression
- Explained what is SGD, BATCH and Epochs
- Did some analysis and comparison between these paramters

## 2026-02-5
### Learning Libraries
- Understood the fundamentals of NumPy

## 2026-02-6
- Understood the fundamentals of Pandas

## 2026-02-8
- Understood the fundamentals of Matplotlib

## 2026-02-9
### Chicago Taxi Fare Prediction Model
- Created the visuals in order to pick the best feature
- Picked Trip_Miles

## 2026-02-13
- Learned the concept of PyTorch library and tensors
- Created a NumPy implementation for the Chicago Taxi Fare Prediction Model of linear regression with manual gradient descent (full batch, fixed iterations)
- Created a PyTorch implementation for the Chicago Taxi Fare Prediction Model with automatic differentiation and dynamic convergence detection
- Implemented GPU acceleration support with automatic fallback to CPU
- Achieved final model: FARE = 2.2795 * TRIP_MILES + 5.0000 with MSE = 14.0254