from typing import Dict, List, Iterable

import pandas as pd
from text2props.constants import Q_ID, Q_TEXT, CORRECT_TEXTS, WRONG_TEXTS
import re
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer
from text2props.utils.nlp import ENGLISH_STOP_WORDS


def gen_correct_answers_dict(input_df: pd.DataFrame) -> Dict[str, List[str]]:
    correct_ans_text_dict = dict()
    for q_id, list_correct_ans in input_df[[Q_ID, CORRECT_TEXTS]].values:
        correct_ans_text_dict[q_id] = list_correct_ans
    return correct_ans_text_dict


def gen_wrong_answers_dict(input_df: pd.DataFrame) -> Dict[str, List[str]]:
    wrong_ans_text_dict = dict()
    for q_id, list_wrong_ans in input_df[[Q_ID, WRONG_TEXTS]].values:
        wrong_ans_text_dict[q_id] = list_wrong_ans
    return wrong_ans_text_dict


def concatenate_answers_text_into_question_text_df(
        input_df: pd.DataFrame,
        correct_texts: Dict[str, List[str]] = None,
        wrong_texts: Dict[str, List[str]] = None,
        correct: bool = True,
        wrong: bool = True
) -> Iterable[str]:
    if correct:
        input_df[Q_TEXT] = input_df.apply(
            lambda r: r[Q_TEXT] + ' ' + correct_texts[r[Q_ID]] if type(correct_texts[r[Q_ID]]) == str else r[Q_TEXT],
            axis=1
        )
    if wrong:
        input_df[Q_TEXT] = input_df.apply(
            lambda r: r[Q_TEXT] + ' ' + wrong_texts[r[Q_ID]] if type(wrong_texts[r[Q_ID]]) == str else r[Q_TEXT],
            axis=1
        )
    return input_df[Q_TEXT]


def vectorizer_text_preprocessor(
        text: str,
        uncased: bool = True,
        remove_stop_words: bool = True,
        remove_html_tags: bool = False,
        remove_numbers: bool = True,
        remove_punctuation: bool = True,
        perform_stemming: bool = True
) -> str:
    if uncased:
        text = text.lower()
    if remove_stop_words:
        text = stop_words_removal(text)
    if remove_html_tags:
        text = html_tags_removal(text)
    if remove_numbers:
        text = numbers_removal(text)
    if remove_punctuation:
        text = punctuation_removal(text)
    if perform_stemming:
        text = stemming(text)
    return text


def html_tags_removal(text: str) -> str:
    return re.sub("<.*?>", " ", text)


def numbers_removal(text: str) -> str:
    return ' '.join([x for x in text.split(' ') if not x.isdigit()])


def punctuation_removal(text: str) -> str:
    return ''.join([char if char not in string.punctuation else ' ' for char in text])


def stop_words_removal(text: str, stop_words=ENGLISH_STOP_WORDS) -> str:
    return ' '.join([word for word in text.split() if word not in stop_words])


def stemming(text: str, stemmer=PorterStemmer()) -> str:
    return ' '.join([stemmer.stem(word) for word in text.split()])


def lemmatization(text: str, lemmatizer=WordNetLemmatizer()) -> str:
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
