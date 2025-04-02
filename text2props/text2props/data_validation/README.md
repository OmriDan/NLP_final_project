# data_validation

This folder contain the source code of the methods that are used to validate the input
data.
For instance, code to check that a pandas DataFrame contains the required columns, or 
that an object is of a specific type.

Data validation methods implemented:

* `check_answers_df_columns`: checks whether the input dataframe has the columns required 
    to be an `answers_df`
* `check_question_df_columns`: checks whether the input dataframe has the columns required 
    to be a `question_df`
* `check_questions_lt_columns(input_df)`: checks whether `input_df` has the columns 
    required to be a  questions' latent traits dataframe (i.e. [question id, latent 
    trait])
