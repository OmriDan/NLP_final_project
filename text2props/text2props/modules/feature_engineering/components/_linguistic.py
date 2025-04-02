import nltk.data
import numpy as np
import pandas as pd
import string
import textstat
from typing import Optional
from scipy.sparse import coo_matrix
from text2props.constants import Q_TEXT, Q_ID
from ..utils import gen_correct_answers_dict, gen_wrong_answers_dict
from . import BaseFeatEngComponent

LEXICON_COUNT_QUESTION = 'lexicon_count_question'
LEXICON_COUNT_ANSWERS = 'lexicon_count_correct_choices'
LEXICON_COUNT_DISTRACTORS = 'lexicon_count_wrong_choices'
SENTENCE_COUNT_QUESTION = 'sentence_count_question'
SENTENCE_COUNT_ANSWERS = 'sentence_count_correct_choices'
SENTENCE_COUNT_DISTRACTORS = 'sentence_count_wrong_choices'
AVG_WORD_LENGTH_QUESTION = 'avg_word_len_question'
RATIO_LENGTH_QUESTION_ANSWERS = 'ratio_len_question_correct_choices'
RATIO_LEN_QUESTION_DISTRACTORS = 'ratio_len_question_wrong_choices'
CHARACTER_COUNT_QUESTION = 'character_count_question'
AVG_SENTENCE_LENGTH = 'avg_sentence_length'
NUM_LONG_WORDS = 'num_long_words'
NUM_SHORT_WORDS = 'num_short_words'
AVG_NUM_LONG_WORDS_PER_SENTENCE = 'avg_num_long_words_per_sentence'
AVG_NUM_SHORT_WORDS_PER_SENTENCE = 'avg_num_short_words_per_sentence'
AVG_NOUNS_PER_SENTENCE = 'avg_nouns_per_sentence'
AVG_VERBS_PER_SENTENCE = 'avg_verbs_per_sentence'

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


class LinguisticFeaturesComponent(BaseFeatEngComponent):

    def __init__(self, version: Optional[int] = 1):
        self.version = version

    def fit_transform(self, input_df: pd.DataFrame) -> coo_matrix:
        """
        Transforms the input data and returns a sparse matrix. It returns a sparse matrix, although these features are
        not sparse for consistency reasons with the other components.
        :param input_df:
        :return:
        """
        return self.transform(input_df)

    def transform(self, input_df: pd.DataFrame) -> coo_matrix:
        """
        It computes and returns the linguistic features from the input DF. The DF must include the following attributes
        in its columns: Q_TEXT, Q_ID
        :param input_df:
        :return:
        """
        if Q_TEXT not in input_df.columns:
            raise ValueError("Q_TEXT should be in input_df.columns")
        if Q_ID not in input_df.columns:
            raise ValueError("Q_ID should be in input_df.columns")

        correct_ans_text_dict = gen_correct_answers_dict(input_df)
        wrong_ans_text_dict = gen_wrong_answers_dict(input_df)

        df = pd.DataFrame()

        for q_id, q_text in input_df[[Q_ID, Q_TEXT]].values:
            q_text_no_punctuation = q_text.translate(str.maketrans('', '', string.punctuation))
            list_words = q_text_no_punctuation.split(' ')
            answers_text = correct_ans_text_dict[q_id]
            distractors_text = wrong_ans_text_dict[q_id]

            try: lexicon_count_question = textstat.lexicon_count(q_text)
            except TypeError: lexicon_count_question = 1
            try: lexicon_count_answers = np.mean([textstat.lexicon_count(x) for x in answers_text])
            except TypeError: lexicon_count_answers = 1
            try: lexicon_count_distractors = np.mean([textstat.lexicon_count(x) for x in distractors_text])
            except TypeError: lexicon_count_distractors = 1

            try: sentence_count_question = textstat.sentence_count(q_text)
            except TypeError: sentence_count_question = 1
            try: sentence_count_answer = np.mean([textstat.sentence_count(x) for x in answers_text])
            except TypeError: sentence_count_answer = 1
            try: sentence_count_distractors = np.mean([textstat.sentence_count(x) for x in distractors_text])
            except TypeError: sentence_count_distractors = 1

            new_row = pd.DataFrame([{
                LEXICON_COUNT_QUESTION: lexicon_count_question,
                LEXICON_COUNT_ANSWERS: lexicon_count_answers,
                LEXICON_COUNT_DISTRACTORS: lexicon_count_distractors,
                SENTENCE_COUNT_QUESTION: sentence_count_question,
                SENTENCE_COUNT_ANSWERS: sentence_count_answer,
                SENTENCE_COUNT_DISTRACTORS: sentence_count_distractors,
                AVG_WORD_LENGTH_QUESTION: np.mean([len(x) for x in list_words]),
                RATIO_LENGTH_QUESTION_ANSWERS: (1 + lexicon_count_question) / (1 + lexicon_count_answers),
                RATIO_LEN_QUESTION_DISTRACTORS: (1 + lexicon_count_question) / (1 + lexicon_count_distractors),
            }])
            df = pd.concat([df, new_row], ignore_index=True)

            if self.version > 1:
                tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                list_sentences_question = tokenizer.tokenize(q_text)

                words_list = nltk.word_tokenize(q_text)
                tagged_words_list = nltk.pos_tag(words_list)

                # number of characters
                new_row[CHARACTER_COUNT_QUESTION] = len(q_text)
                # average sentence length
                new_row[AVG_SENTENCE_LENGTH] = np.mean([len(sentence) for sentence in list_sentences_question])

                # nouns per sentence
                new_row[AVG_NOUNS_PER_SENTENCE] = np.mean([len([x for x in nltk.pos_tag(nltk.word_tokenize(s)) if x[1] in {'NN', 'NNS'}]) for s in list_sentences_question])
                # verbs per sentence
                new_row[AVG_VERBS_PER_SENTENCE] = np.mean([len([x for x in nltk.pos_tag(nltk.word_tokenize(s)) if x[1] in {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}]) for s in list_sentences_question])

                # Number of long words (7 characters or more) in the question
                new_row[NUM_LONG_WORDS] = len([x for x in list_words if len(x) >= 7])
                # Number of short words in the question
                new_row[NUM_SHORT_WORDS] = len([x for x in list_words if len(x) < 7])
                # Average number of long words per sentence
                new_row[AVG_NUM_LONG_WORDS_PER_SENTENCE] = np.mean([len([w for w in s if len(w) >= 7]) for s in list_sentences_question])
                # Average number of short words per sentence
                new_row[AVG_NUM_SHORT_WORDS_PER_SENTENCE] = np.mean([len([w for w in s if len(w) < 7]) for s in list_sentences_question])

        return coo_matrix(df.values)

# other features that might be added in the futures;
#   - Average number of syllables per word
#   - number of double consonants
#   - number of vowels
