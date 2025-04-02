This folder contains the definition of the `Text2PropsModel` class, which is the class that must be used to implement
models with `text2props`.
`Text2PropsModel` objects are made of one two modules:

1. latent traits calibrator: performs the estimation of ground truth latent traits
2. estimator from text: performs the estimation of latent traits from textual information

---

# `Text2PropsModel` class

This is the class that must be used to implement models with the `text2props` framework.

Objects of this class have the following attributes:

- `n_latent_traits`: the number of latent traits that are estimated by the model.
- `latent_traits`: the name of the latent traits that are estimated by the model.
- `latent_traits_calibrator`: the latent traits calibrator module, to perform the estimation of ground truth latent 
  traits.
- `estimator_from_text`: the estimator from text object, to perform the estimation of latent traits from textual info
- `ground_truth_latent_traits`: dictionary containing the estimated latent traits. Specifically, it is a nested dict and
  the structure of the dictionary is as follows: the first key represents the name of the latent trait (even if there is 
  only one latent trait), the second key is the question ID. 

Objects of this class have the following methods:

- `calibrate_latent_traits`: performs the estimation of ground truth latent traits using the latent traits estimator 
  module defined in the model
- `train`: trains the model, by calibrating the latent traits and training the estimator from text object
- `predict`: perform the prediction
- `randomized_cv_train`: performs the training of the estimator from text object with randomized cross validation. 
  Also, this method returns the score obtained with the best performing configuration, so that they can be compared with
  other models to perform model selection.
- `get_calibrated_latent_traits`
- `compute_error_metrics_latent_traits_estimation`
- `store_calibrated_latent_traits`
