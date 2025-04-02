# Estimators from text modules

This folder contains the code with the definition of the classes used to implement the estimators from text module.
All the classes have to generalize the `BaseEstimatorFromText` class, containing the following methods:

- train
- predict

At the moment, the following latent traits estimator classes are defined:

- `FeatureEngAndRegressionEstimatorFromText`: an EstimatorFromText object which is made of one or more Pipelines, each 
  made of two modules (a feature engineering component and a regression component) to estimate one latent traits. Each 
  module can be made of one or more components (to perform the estimation with an ensemble) 
- `MajorityEstimatorFromText`