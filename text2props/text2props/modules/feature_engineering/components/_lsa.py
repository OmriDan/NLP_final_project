from ._base import BaseFeatEngComponent
from text2props.constants import Q_TEXT

from gensim import corpora
from gensim.matutils import sparse2full
from gensim.models import LsiModel, TfidfModel
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from scipy.sparse import coo_matrix
from text2props.utils.nlp import ENGLISH_STOP_WORDS


class LSAFeaturesComponent(BaseFeatEngComponent):

    def __init__(self, model: dict = None, number_of_topics: int = 5):
        """
        :param model: if available, this components can be initialized with a already trained model
        :param number_of_topics:
        """
        self.is_pretrained = model is not None
        if not self.is_pretrained:
            self.model = dict()
        self.number_of_topics = number_of_topics

    def fit(self, input_df):
        if not self.is_pretrained:
            clean_texts = _preprocess_data(input_df[Q_TEXT].values)
            dictionary = corpora.Dictionary(clean_texts)
            corpus = [dictionary.doc2bow(text) for text in clean_texts]
            tfidf_model = TfidfModel(corpus)
            corpus_tfidf = tfidf_model[corpus]
            lsi_model = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=self.number_of_topics)
            self.model = {'lsi': lsi_model, 'tfidf': tfidf_model, 'dictionary': dictionary}
        else:
            raise NotImplementedError
        return

    def transform(self, input_df):
        clean_texts = _preprocess_data(input_df[Q_TEXT].values)
        corpus = [self.model['dictionary'].doc2bow(text) for text in clean_texts]
        corpus_tfidf = self.model['tfidf'][corpus]
        corpus_lsi = [sparse2full(self.model['lsi'][item], self.number_of_topics) for item in corpus_tfidf]
        return coo_matrix(corpus_lsi)

    def fit_transform(self, input_df) -> coo_matrix:
        self.fit(input_df)
        return self.transform(input_df)


def _preprocess_data(item_list):
    """
    Preprocess the text (tokenize, removing stopwords, and stemming)
    :param item_list: list of documents
    :return: preprocessed text
    """
    tokenizer = RegexpTokenizer(r'\w+')
    p_stemmer = PorterStemmer()
    texts = []
    for text in item_list:
        raw = text.lower()
        tokens = tokenizer.tokenize(raw)
        stopped_tokens = [i for i in tokens if i not in ENGLISH_STOP_WORDS]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        texts.append(stemmed_tokens)
    return texts
