from ._base import BaseFeatEngComponent
from text2props.constants import (
    Q_TEXT,
    MODELS_PATH,
)
import gensim.models
import numpy as np
import os
from scipy.sparse import coo_matrix
from ..utils import gen_wrong_answers_dict, gen_correct_answers_dict, concatenate_answers_text_into_question_text_df


class Word2VecFeaturesComponent(BaseFeatEngComponent):

    def __init__(
            self,
            model=None,
            min_count: int = 5,
            size: int = 100,
            workers: int = 3,
            seed: int = 1,
            input_model_path: str = MODELS_PATH,
            input_model_name: str = None,
            output_model_path: str = MODELS_PATH,
            output_model_name: str = None,
            concatenate_correct: bool = False,
            concatenate_wrong: bool = False,
    ):
        """
        :param model: if available, this components can be initialized with a already trained word2vec model
        :param min_count: a training parameter of gensim.models.Word2Vec, it is used for pruning the internal dictionary
        :param size: a training parameter of gensim.models.Word2Vec, it is the number of dimensions of the space that
            the word2vec model maps into
        :param workers: a training parameter of gensim.models.Word2Vec, it is used for training parallelization, to
            speed up training
        :param seed: a training parameter of gensim.models.Word2Vec, it is the random seed
        :param input_model_path: the path of the folder where the model to load is memorized
        :param input_model_name: the name of the model to load
        :param output_model_path: the path of the folder where the model has to be saved
        :param output_model_name: the name of the model to save
        """
        self.model = model
        self.min_count = min_count
        self.size = size
        self.workers = workers
        self.seed = seed
        self.input_model_path = input_model_path
        self.input_model_name = input_model_name
        self.output_model_path = output_model_path
        self.output_model_name = output_model_name
        self.correct_text_dict = None
        self.wrong_text_dict = None
        self.concatenate_correct = concatenate_correct
        self.concatenate_wrong = concatenate_wrong

    def fit_transform(self, input_df):
        self.correct_text_dict = gen_correct_answers_dict(input_df)
        self.wrong_text_dict = gen_wrong_answers_dict(input_df)
        local_df = input_df.copy()
        local_df[Q_TEXT] = concatenate_answers_text_into_question_text_df(
            local_df, self.correct_text_dict, self.wrong_text_dict, self.concatenate_correct, self.concatenate_wrong)
        if self.input_model_name is None:
            sentences = [gensim.utils.simple_preprocess(q_text) for q_text in local_df[Q_TEXT]]
            self.model = gensim.models.Word2Vec(
                sentences=sentences, min_count=self.min_count, vector_size=self.size, seed=self.seed, workers=self.workers,
            )
            if self.output_model_name is not None:
                self.model.save(os.path.join(self.output_model_path, self.output_model_name))
        else:
            self.model = gensim.models.Word2Vec.load(os.path.join(self.input_model_path, self.input_model_name))
        return self.transform(input_df)

    def transform(self, input_df):
        self.correct_text_dict.update(gen_correct_answers_dict(input_df))
        self.wrong_text_dict.update(gen_wrong_answers_dict(input_df))
        local_df = input_df.copy()
        local_df[Q_TEXT] = concatenate_answers_text_into_question_text_df(
            local_df, self.correct_text_dict, self.wrong_text_dict, self.concatenate_correct, self.concatenate_wrong)
        results = np.empty([len(local_df.index), self.size])
        for idx, text in enumerate(local_df[Q_TEXT].values):
            results[idx, :] = np.mean(
                [self.model.wv[x] if x in self.model.wv.key_to_index.keys() else np.zeros(self.size) for x in text.split(' ')],
                axis=0
            )
        return coo_matrix(results)
