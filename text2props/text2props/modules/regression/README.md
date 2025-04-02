# `RegressionModule`

This folder contains the code with the definition of the `RegressionModule` class and the classes that are used 
to implement the components it is made of.

---

## `RegressionModule` class

Objects of this class are used to implement regression modules.
They contain the following attribute:

- `components`: a list of the components that the regression module is made of

They contain the following methods:

- `add_component`: add one component to the list of components
- `add_list_components`: add a list of components to the list of components defined in the module
- `train`: train the module
- `predict`: perform the prediction
- `randomized_cv_train`: trains (with randomized cross validation) the components that the module is made of

---

## Component objects

All the components classes have to generalize the `BaseRegressionComponent` class, and they contain the following 
methods:

- `train`
- `predict`
- `randomized_cv_train`

At the moment, the following component classes are defined:

- `SklearnRegressionComponent`: regression component that is made of a sklearn regressor object
