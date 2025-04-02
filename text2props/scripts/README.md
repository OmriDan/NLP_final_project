# Example scripts

* `example_r2de.py`: implementation of the R2DE model with *text2props*. [R2DE](https://arxiv.org/abs/2001.07569) 
consists in a IRT estimation with a two-parameters model for the ground truth latent traits, followed by two Random 
Forest models for the estimation of latent traits from textual information. 
* `example_wrongness_majority.py`: implementation of a simple model that performs the estimation of the wrongness of 
each question (i.e. fraction of wrong answers) and the estimation from text with a majority model (i.e. for all the 
questions the same difficulty is predicted, and that is the average wrongness of all the training questions).