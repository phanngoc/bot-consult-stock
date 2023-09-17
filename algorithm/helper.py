import numpy as np

def isNumpy(x):
    return isinstance(x, np.ndarray)


from pyvi import ViTokenizer, ViPosTagger
from underthesea import word_tokenize
import pandas as pd
import re

class NLP():
    COMMON_WORD_PATH = 'data/cmword.csv'
    stopwords = []

    def __init__(self, text = None):
        self.text = text
        self.__set_stopwords()

    def __set_stopwords(self):
        try:
            with open(self.COMMON_WORD_PATH, 'r', encoding="utf-8") as f:
                stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
                self.stopwords = stopwords
        except FileNotFoundError:
            pass

    def segmentation(self, engine='underthesea'):
        if engine == 'pyvi':
            return ViTokenizer.tokenize(self.text)
        elif engine == 'underthesea':
            try:
                return word_tokenize(self.text, format="text")
            except (TypeError, AttributeError):
                return ''

    def split_words(self):
        text = self.segmentation()
        try:
            t = [str(x.strip('0123456789%@$.,=+-!;/()*"&^:#|\n\t\' ').lower()) for x in text.split()]
            filtered_list = [item for item in t if item != ""]
            return filtered_list
        except TypeError:
            return []

    def get_words_feature(self):
        split_words = self.split_words()
        return [word for word in split_words if word not in self.stopwords]

    def get_string_sanitize(self):
        words = self.get_words_feature()
        return ' '.join(words)

    def remove_common_word(self):
        pf_cm_words = pd.read_csv(self.COMMON_WORD_PATH)
        pf_cm_words.columns = ['regex', 'rep']
        rep = pd.Series(pf_cm_words.rep.values,index=pf_cm_words.regex.values).to_dict()
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep.get(re.escape(m.group(0)), ''), self.text)
        return text


# text = """Bộ Công Thương chịu trách nhiệm trước Chính phủ quản lý nhà nước về điện lực, 
# trong đó có giá điện nên Bộ Tài chính đề nghị không quy định trách nhiệm phối hợp rà soát của bộ này khi điều chỉnh giá."""

# nlp = NLP(text)
# print(nlp.get_words_feature())
# print(nlp.get_string_sanitize())