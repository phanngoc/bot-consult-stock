import array
from re import T
from matplotlib import transforms
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import embedding
from torch.nn.functional import pad
from underthesea import classify, word_tokenize, sent_tokenize
from pickle import load
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from helper import isNumpy, NLP
from transformers import AutoModel, AutoTokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Lambda, Embedding, LSTM, SimpleRNN, GRU, Input, Bidirectional
from keras.preprocessing.text import Tokenizer
import pandas as pd

class TfidfTransform:
    data_origin = None
    def __init__(self):
        print('init tfidf', self.data_origin)
        self.vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.9, max_features = 2000000, ngram_range = (1, 1))
        if self.data_origin is not None:
            self.vectorizer.fit(self.data_origin)
        pass

    def fit_transform(self, data):
        dataSeris = pd.Series(data)
        data = dataSeris.fillna('').astype(str)
        self.data_origin = data
        try:
            advancedtrain = self.vectorizer.fit_transform(data)
        except Exception as e:
            print('e', e)
            return None
        self.data_transformed = advancedtrain.toarray()
        return self.data_transformed

    def transform(self, data):
        advancedtrain = self.vectorizer.transform(data)
        self.data_transformed = advancedtrain.toarray()
        return self.data_transformed
    
    def inverse_transform(self, data):
        return self.vectorizer.inverse_transform(data)
    
    def save_model(self, path):
        joblib.dump(self.vectorizer, path)

    def load_model(self, path):
        self.vectorizer = joblib.load(path)
    
class PhobertTransform:
    max_len = 300
    data_transformed = None

    def __init__(self, type_extract='mean'):
        self.type_extract = type_extract
        self.model = AutoModel.from_pretrained("vinai/phobert-base-v2")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)

    def fit_transform(self, sentences):
        embeddingArr = []
        for sentence in sentences:
            try:
                encodedText = self.tokenizer.encode_plus(
                    sentence, padding=True, truncation=True, 
                    add_special_tokens=True,
                    return_attention_mask=True,
                    max_length=self.max_len, return_tensors='pt')
            except Exception as e:
                print('e', e, sentence)
                return None
            
            encodedTextInputIds = encodedText['input_ids']
            embeddedTitle = self.model(torch.tensor(encodedTextInputIds))
            embedding = self.extract_hidden_state(embeddedTitle, self.type_extract)
            embeddingArr.append(embedding.ravel())
        
        self.data_transformed = np.array(embeddingArr)
        return self.data_transformed
    
    def save_model(self, path):
        joblib.dump(self.data_transformed, path)

    def load_model(self, path):
        self.data_transformed = joblib.load(path)

    def transform(self, data):
        return self.fit_transform(data)

    def extract_hidden_state(self, outputs, type='view'):
        # 2 way for get hidden state: outputs[0], outputs['last_hidden_state']
        with torch.no_grad():
            if type == 'view':
                hiddenSize = outputs[0].size()[0]
                # flatten dimension 1 and 2, and hold batch dimension 0
                embeddings = outputs[0].view(hiddenSize, -1).numpy()
            elif type == 'mean':
                hiddenSize = outputs[0].size()[0]
                embeddings = outputs[0].mean(axis=1).numpy()

        return embeddings

    def inverse_transform(self, data):
        return self.vectorizer.inverse_transform(data)

from keras_preprocessing.sequence import pad_sequences

class SequenceTransform:
    max_len = 300
    def __init__(self, max_len=300):
        self.max_len = max_len
        pass

    def fit_transform(self, data):
        self.tokenizer = Tokenizer(num_words=10000)
        self.tokenizer.fit_on_texts(data)
        data_transformed = self.tokenizer.texts_to_sequences(data)
        data_transformed = pad_sequences(data_transformed, maxlen = self.max_len, padding='post', truncating='post')
        return data_transformed

    def transform(self, data):
        data_transformed = self.tokenizer.texts_to_sequences(data)
        data_transformed = pad_sequences(data_transformed, maxlen = self.max_len, padding='post', truncating='post')
        return data_transformed
    
    def inverse_transform(self, data):
        return self.vectorizer.inverse_transform(data)
    
    def save_model(self, path):
        joblib.dump(self.tokenizer, path)

    def load_model(self, path):
        self.tokenizer = joblib.load(path)


def clear_text_vi(texts = []):
    clear_ld = lambda t: NLP(t).get_string_sanitize()
    vec_fn = np.vectorize(clear_ld)
    return vec_fn(texts)
    
import os
import joblib

class RegressionTextToPrice:
    max_len = 300
    path_cached = './cached/'
    valid_algorithm = ['randomforest', 'svr', 'linear', 'nbsvm', 'logistic', 'mlp']
    valid_transform_type = ['phobert', 'tfidf', 'sequence']

    def __init__(self, transform_type='phobert', algorithm='svr'):
        self.transform_type = transform_type
        self.algorithm = algorithm

        if self.algorithm not in self.valid_algorithm:
            raise Exception('algorithm not valid. Valid param is:', self.valid_algorithm)

        if self.transform_type not in self.valid_transform_type:
            raise Exception('transform_type not valid. Valid param is:', self.valid_transform_type)

        if self.transform_type == 'phobert':
            self.transformer = PhobertTransform()
        elif self.transform_type == 'tfidf':
            self.transformer = TfidfTransform()
        elif self.transform_type == 'sequence':
            self.transformer = SequenceTransform()

    def fit(self, x_train, y_train, cache=False):
        if cache and self.load_model():
            print('load cache')
            return
        self.x_train_raw = np.array(x_train) if isNumpy(x_train) else x_train
        self.y_train_raw = np.array(y_train) if isNumpy(y_train) else y_train

        self.x_train_raw = clear_text_vi(self.x_train_raw)

        self.x_train = self.transformer.fit_transform(self.x_train_raw)
        self.y_train = self.y_train_raw
        
        if self.algorithm == 'svr':
            self.regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        elif self.algorithm == 'linear':
            self.regressor = LinearRegression()
        elif self.algorithm == 'randomforest':
            self.regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.algorithm == 'nbsvm':
            self.regressor = LinearSVC(C=0.01)
        elif self.algorithm == 'logistic':
            self.regressor = GradientBoostingClassifier()
        elif self.algorithm == 'mlp': # @TODO algorithm == 'mlp' not work in current. Need to fix
            print('mlp', self.x_train.shape, self.y_train.shape)
            self.y_train = np.expand_dims(self.y_train, axis=1)
            self.scaler = MinMaxScaler(feature_range = (0,1))
            y_train_scaled = self.scaler.fit_transform(self.y_train)
            print('y_train_scaled', y_train_scaled.shape, y_train_scaled[:2])
            self.regressor = Sequential()
            self.regressor.add(LSTM(units = 50,return_sequences = True, input_shape = (self.x_train.shape[1],1)))
            self.regressor.add(Dropout(0.2))
            self.regressor.add(LSTM(units = 50,return_sequences = True))
            self.regressor.add(Dropout(0.2))
            self.regressor.add(LSTM(units = 50,return_sequences = True))
            self.regressor.add(Dropout(0.2))
            self.regressor.add(LSTM(units = 50))
            self.regressor.add(Dropout(0.2))

            self.regressor.add(Dense(units = 1))

            self.regressor.compile(optimizer = 'adam',loss = 'mean_squared_error')
            self.regressor.fit(self.x_train, y_train_scaled, epochs = 30, batch_size = 32)

            return

        self.regressor.fit(self.x_train, self.y_train)  # ravel y_train to convert to 1D array
        self.save_model() if cache else None

    def get_path_cached(self, type='model'):
        if type == 'model':
            path = f"{self.path_cached}regressor_{self.algorithm}.model"
        elif type == 'transformer':
            path = f"{self.path_cached}transformer_{self.transform_type}.bin"
        return path

    def type_model(self):
        if self.algorithm == 'mlp':
            return 'keras'
        else:
            return 'sklearn'

    def save_model(self):
        type = self.type_model()
        path_model_cached = self.get_path_cached('model')
        if type == 'sklearn':
            joblib.dump(self.regressor, path_model_cached)
        elif type == 'keras':
            self.regressor.save(path_model_cached)

        path_transform_cached = self.get_path_cached('transformer')
        self.transformer.save_model(path_transform_cached)

    def load_model(self):
        type = self.type_model()
        path_model_cached = self.get_path_cached('model')
        path_transform_cached = self.get_path_cached('transformer')
        if (not os.path.exists(path_model_cached) or not os.path.exists(path_transform_cached)):
            return False

        if type == 'sklearn':
            self.regressor = joblib.load(path_model_cached)
        elif type == 'keras':
            self.regressor = load_model(path_model_cached)

        path_transform_cached = self.get_path_cached('transformer')
        self.transformer.load_model(path_transform_cached)
        return True

    def predict(self, x_test):
        x_test = self.transformer.transform(x_test)
        result = self.regressor.predict(x_test)
        return result
    
    @staticmethod
    def trim_words(sentence, num_word=200):
        # Split the sentence into words
        words = sentence.split()
        # Trim the sentence to 5 words
        trimmed_sentence = " ".join(words[:num_word])
        return trimmed_sentence


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
