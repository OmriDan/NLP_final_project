# `FeatureEngineeringModule`

This folder contains the code with the definition of the `FeatureEngineeringModule` class and the classes that are used 
to implement the components it is made of.

---

## `FeatureEngineeringModule` class

Objects of this class are used to implement feature engineering modules.
They contain the following attribute:

- `components`: a list of the components that the feature engineering module is made of

They contain the following methods:

- `add_component`: add one component to the list of components
- `add_list_components`: add a list of components to the list of components defined in the module
- `fit_transform`: fit the module and transform the input data (similar to fit_transform in sklearn vectorizers)
- `transform`: should only be used after fit_transform, transform the input data

---

## Component objects

All the components classes have to generalize the `BaseFeatEngComponent` class, and they contain the following methods:

- `fit_transform`
- `transform`

At the moment, the following component classes are defined:

- `IRFeaturesComponent`: computes IR-based features from the input text, using a sklearn vectorizer
- `LinguisticFeaturesComponent`: computes linguistic features
- `ReadabilityFeaturesComponent`: computes features based on readability measures
