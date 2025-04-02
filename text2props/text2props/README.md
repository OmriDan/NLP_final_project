# *text2props*

Models implemented with this framework estimate questions' latent traits using the text of the
questions and, in the case of Multiple Choice Questions (MCQ), the text of the possible choices.
Using `text2props`, one or more latent traits can be estimated at the same time.

The models implemented with `text2props` have two components:

1. *latent traits calibrator* - performs question calibration, by estimating ground truth latent traits, which are then
used to train and evaluate the estimators that estimate latent traits from textual information.
2. *estimator of latent traits form text* - performs the actual estimation of the latent traits from text. This is the
part of which is used to calibrate previously unseen questions.

---

While implementing a model with *text2props*, the steps to follow are the following:
* define the `Calibrator` object for latent traits calibration
* define the `EstimatorFromText` object
* define the `Text2PropsModel` object, containing the two objects initialized before
* at this point all the actions (e.g. train, predict) are performed at `Text2PropsModel` granularity

---

This folder is organized as follows:

- `text2props/data_validation/` contains methods to validate the data (e.g., check whether a DF has the required columns)
- `text2props/evaluation/` contains the code for evaluating the models.
- `text2props/model/` contains the definition of the classes used to implement the models.
- `text2props/modules/` contains the definition of the classes used to implement the modules and their components.
- `text2props/utils/` contains some utility code used in other parts of the package.
- `text2props/constants.py` contains the definition of the constants