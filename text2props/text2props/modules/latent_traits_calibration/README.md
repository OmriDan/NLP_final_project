# Latent traits calibration modules

This folder contains the code with the definition of the classes used to implement the latent traits calibration module.
All the classes have to generalize the `BaseLatentTraitsCalibrator` class, containing the following methods:

- get_n_latent_traits: returns the number of latent traits in the estimator
- get_name_latent_traits: returns the name of the latent traits
- get_calibrated_latent_traits: returns the dictionary of the estimated latent traits
- calibrate_latent_traits: performs the estimation of the ground truth latent traits and returns the dictionary 
    containing the estimated values.

At the moment, the following latent traits estimator classes are defined:

- `KnownParametersCalibrator`: degenerate estimator, to used if the latent traits are already known
- `IRTCalibrator`: performs the estimation using Item Response Theory (1 parameter model or 2 parameters model). The 
default arguments implement the 1PM model, with `difficulty_range = [-5; +5]` and discrimination = 1.0. In order to use
the 2PM it is necessary to specify a range for the discrimination.
- `WrongnessCalibrator`: computes the wrongness of each question (i.e. fraction of wrong answers divided by the total 
    number of answers)
