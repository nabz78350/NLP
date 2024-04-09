# Load, explore and plot data
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from params import *
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout, GlobalAveragePooling1D, Flatten, SpatialDropout1D, Bidirectional
import re 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords


## for knn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

## for xgb 
from nltk.corpus import stopwords
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## sklearn models 

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
## for MLP

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout, GlobalAveragePooling1D, Flatten, SpatialDropout1D, Bidirectional
class DataClass :
    
    def __init__(self,use_prediction : bool = True,
                path_dir : str = PATH_DATA_TREATED,
                custom : bool = True,
                use_enhanced : str = "lstm",

                 ):
        
        
        self.use_prediction = use_prediction 
        self.features = ['link', 'employer', 'occupation', 'name_sex','firstname_lower']
        self.path_dir = path_dir
        self.custom = custom
        self.use_enhanced = use_enhanced
        self.import_data()
    
    def import_data(self):
        
        if self.use_prediction:
            path = os.path.join(self.path_dir,'data_prediction.pq')
        else:
            path = os.path.join(self.path_dir,'data_groundtruth.pq')
        self.data = pd.read_parquet(path)
        
        self.freq_name = pd.read_csv(os.path.join(PATH_DATA,'firstname_with_sex.csv'),sep = ";")
        self.freq_name['total'] = self.freq_name[['male','female']].sum(1)
    
        if self.use_prediction:
            self.enhanced_index = self.data[self.data['firstname_lower']!= self.data['firstname_lower_enhanced']].index.tolist()
        else :
            self.enhanced_index = []
        
    def merge_features(self,output_col : str = 'X'):
        
        if self.use_enhanced == "enhanced":
            print('modifying features')
            print(self.features)
            self.features.remove('firstname_lower')
            self.features.append('firstname_lower_enhanced')
            print(self.features)
        elif self.use_enhanced == "lstm":
            print('modifying features')
            print(self.features)
            self.features.remove('firstname_lower')
            self.features.append('firstname_lower_lstm')
            print(self.features)
        elif self.use_enhanced == "fuzzy":
            print('modifying features')
            print(self.features)
            self.features.remove('firstname_lower')
            self.features.append('firstname_lower_fuzzy')
            print(self.features)
        else :
            pass
        merged = self.data[self.features].fillna('').apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        merged =  pd.DataFrame(merged,columns = [output_col]) 
        nltk.download('stopwords')
        s = stopwords.words('french')
        ps = nltk.wordnet.WordNetLemmatizer()
        for i in merged.index.tolist():
            review = re.sub('[^a-zA-Z]', ' ', merged.loc[i,'X'])
            review = review.lower()
            review = review.split()
            review = [ps.lemmatize(word) for word in review if not word in s]
            review = ' '.join(review)
            merged.loc[i, 'X'] = review
        return merged 
    
    
    def create_dataset(self):
        X = self.merge_features()
        Y = self.data[['target']]
        assert X.shape == Y.shape 
        self.data_model = X.join(Y)
        self.data_model.columns = ['message','label']
        
    @staticmethod    
    def create_unbalanced_dataset(data:pd.DataFrame,
                                  multiplier:int = 3,
                                  class_oversampled:str ="zeros"):
    
        count_zeros = data['label'].value_counts()[0]
        count_ones = data['label'].value_counts()[1]
        valid = min(count_zeros,count_ones)
        
        
        zeros = data[data['label']==0].sample(n=valid)
        ones = data[data['label']==1].sample(n=valid)
        if class_oversampled == "zeros":
            zeros_df = [zeros for _ in range(multiplier-1)]
            ones_df = ones 
            zeros_df = pd.concat(zeros_df,axis=0)
        else :
            ones_df = [ones for _ in range(multiplier-1)]
            zeros_df = zeros
            ones_df = pd.concat(ones_df,axis=0)
        unbalanced_data = pd.concat([zeros_df,ones_df],axis=0)
        unbalanced_data['label'].value_counts() / unbalanced_data.shape[0]
        return unbalanced_data
                
        
class DataModel:
    
    def __init__(self,
                 data:pd.DataFrame,
                custom_test_index : list = [],
                max_len : int = 50 ,
                use_enhanced : str = "lstm",
                custom : bool = False,
                trunc_type : str = 'post',
                padding_type: str  = 'post',
                oov_tok : str = '<OOV>', 
                vocab_size : int = 500):
    
    
        self.data = data
        self.custom_test_index = custom_test_index
        self.max_len = max_len
        self.custom = custom
        self.use_enhanced = use_enhanced
        self.trunc_type = trunc_type
        self.padding_type = padding_type
        self.oov_tok = oov_tok
        self.vocab_size = vocab_size
        
        
    def split_dataset(self,test_size :float = 0.3):
        if self.custom :
            custom_test = self.data.loc[self.custom_test_index]
            self.x_test = custom_test['message']
            self.y_test = custom_test['label'].values 
            custom_train = self.data.drop(self.custom_test_index)
            self.x_train = custom_train['message']
            self.y_train = custom_train['label'].values 
        
        else :
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data['message'],
                                                                                self.data['label'].values,
                                                                                test_size=test_size, random_state=434)
        self.train = pd.DataFrame(self.x_train,columns = ['message'])
        self.train['label'] = self.y_train
        
        self.test = pd.DataFrame(self.x_test,columns = ['message'])
        self.test['label'] = self.y_test
    def create_tokenizer(self):
        self.tokenizer = Tokenizer(num_words = self.vocab_size, 
                                    char_level = False,
                                    oov_token = self.oov_tok)
        
        self.tokenizer.fit_on_texts(self.x_train)
    
    def create_padding(self):
        
        self.split_dataset()
        self.create_tokenizer()
        
        self.training_sequences = self.tokenizer.texts_to_sequences(self.x_train)
        self.training_padded = pad_sequences(self.training_sequences,
                                        maxlen = self.max_len,
                                        padding = self.padding_type,
                                        truncating = self.trunc_type)


        self.testing_sequences = self.tokenizer.texts_to_sequences(self.x_test)
        self.testing_padded = pad_sequences(self.testing_sequences,
                                    maxlen = self.max_len,
                                    padding = self.padding_type,
                                    truncating = self.trunc_type)


        print('Shape of training tensor: ', self.training_padded.shape)
        print('Shape of testing tensor: ', self.testing_padded.shape)
        
        

class KnnModel :
    
    def __init__(self,
                 x_train : pd.DataFrame,
                 y_train : np.array,
                 x_test : pd.DataFrame,
                 y_test : np.array,
                 model_args:dict):
        self.model_args = model_args 
        
        nltk.download('stopwords')
        french_stopwords = stopwords.words('french')
        self.count_vect = CountVectorizer(stop_words=french_stopwords)
        self.tfidf_transformer = TfidfTransformer()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def fit(self):
        X_train_counts = self.count_vect.fit_transform(self.x_train)
        X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)
        knn = KNeighborsClassifier(**self.model_args)
        
        self.model =  knn.fit(X_train_tfidf, self.y_train)
        self.y_pred_train  = self.model.predict(X_train_tfidf)
        self.accuracy_train = accuracy_score(self.y_train,self.y_pred_train)
            
    def predict(self) :
        X_new_counts = self.count_vect.transform(self.x_test)
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
        self.y_pred_test = self.model.predict(X_new_tfidf)
        self.accuracy_test = accuracy_score(self.y_test,self.y_pred_test)


class NaiveBayesModel :
    
    def __init__(self,
                 x_train : pd.DataFrame,
                 y_train : np.array,
                 x_test : pd.DataFrame,
                 y_test : np.array,
                 model_args:dict):
        self.model_args = model_args 
        nltk.download('stopwords')
        french_stopwords = stopwords.words('french')
        self.count_vect = CountVectorizer(stop_words=french_stopwords)
        self.tfidf_transformer = TfidfTransformer()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def fit(self):
        X_train_counts = self.count_vect.fit_transform(self.x_train)
        X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)
        nb = naive_bayes.GaussianNB(**self.model_args)
        
        self.model =  nb.fit(X_train_tfidf.toarray(), self.y_train)
        self.y_pred_train  = self.model.predict(X_train_tfidf.toarray())
        self.accuracy_train = accuracy_score(self.y_train,self.y_pred_train)
            
    def predict(self) :
        X_new_counts = self.count_vect.transform(self.x_test)
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
        self.y_pred_test = self.model.predict(X_new_tfidf.toarray())
        self.accuracy_test = accuracy_score(self.y_test,self.y_pred_test)
        
        


class LogReg :
    
    def __init__(self,
                 x_train : pd.DataFrame,
                 y_train : np.array,
                 x_test : pd.DataFrame,
                 y_test : np.array,
                 model_args:dict):
        self.model_args = model_args 
        nltk.download('stopwords')
        french_stopwords = stopwords.words('french')
        self.count_vect = CountVectorizer(stop_words=french_stopwords)
        self.tfidf_transformer = TfidfTransformer()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def fit(self):
        X_train_counts = self.count_vect.fit_transform(self.x_train)
        X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)
        nb = LogisticRegression(**self.model_args)
        
        self.model =  nb.fit(X_train_tfidf.toarray(), self.y_train)
        self.y_pred_train  = self.model.predict(X_train_tfidf.toarray())
        self.accuracy_train = accuracy_score(self.y_train,self.y_pred_train)
            
    def predict(self) :
        X_new_counts = self.count_vect.transform(self.x_test)
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
        self.y_pred_test = self.model.predict(X_new_tfidf.toarray())
        self.accuracy_test = accuracy_score(self.y_test,self.y_pred_test)

class XGBModel:
    
    
    def __init__(self,x_train : pd.DataFrame,
                 y_train : np.array,
                 x_test : pd.DataFrame,
                 y_test : np.array,
                 model_args:dict):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.final_stopwords_list = stopwords.words('french')

        self.model_args = model_args 
        
    def fit(self):
        
        self.vectorizer = TfidfVectorizer(max_features=None,
                                          min_df = 0.2,
                        stop_words=self.final_stopwords_list,
                        use_idf=True,
                        ngram_range=(1,3))
        X_train = self.vectorizer.fit_transform(self.x_train)
        self.model = xgb.XGBClassifier(**self.model_args)
        self.model.fit(X_train, self.y_train)
        
        self.y_pred_train = self.model.predict(X_train)
        self.accuracy_train = accuracy_score(self.y_train,self.y_pred_train)
        
    def predict(self):
        X_test = self.vectorizer.transform(self.x_test)
        self.y_pred_test = self.model.predict(X_test)
        self.accuracy_test = accuracy_score(self.y_test,self.y_pred_test)
    
        
class MLPModel :
    
    def __init__(self,
                 x_train,
                 y_train ,
                 x_test ,
                 y_test ,
                 model_args:dict):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_args = model_args
        
        
    def fit(self):
        self.vocab_size = self.model_args["vocab_size"]
        self.embed_size = self.model_args["embed_size"]
        self.n_layers = self.model_args["n_layers"]
        self.hidden_size =self.model_args["hidden_size"]
        self.output_dim =1 
        self.dropout_rate = self.model_args["dropout_rate"]
        self.lr = self.model_args["lr"]
        self.max_document_length = self.model_args["max_document_length"]
        self.hidden_function = self.model_args["hidden_function"]
        self.model = Sequential()
        self.model.add(Input(shape=(self.max_document_length,)))  # input layer specifying the input shape
        self.model.add(Embedding(input_dim=self.vocab_size, output_dim=self.embed_size, input_length=self.max_document_length))
        self.model.add(Flatten())  # flattening the output of the embedding layer to fit into dense layers
        for _ in range(self.n_layers):
            self.model.add(Dense(self.hidden_size, activation=self.hidden_function))
            self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(self.output_dim, activation='sigmoid' if self.output_dim == 1 else 'softmax'))  # for binary classification use 'sigmoid', for multi-class use 'softmax'

        # Compile the self.
        self.model.compile(optimizer="adam",
                           loss='binary_crossentropy' if self.output_dim == 1 else 'categorical_crossentropy',
                           metrics=['accuracy'])
        self.num_epochs = self.model_args["num_epochs"]
        self.batch_size = self.model_args["batch_size"]
        early_stop = EarlyStopping(monitor='val_loss', patience=10)
        self.history = self.model.fit(self.x_train,
                            self.y_train,
                            epochs=self.num_epochs, 
                            validation_split = 0.2,
                            batch_size = self.batch_size,
                            callbacks =[early_stop],
                            verbose=2)
        
        self.train_loss = pd.DataFrame(self.history.history['loss'],columns = ['train_loss'])
        self.val_loss = pd.DataFrame(self.history.history['val_loss'],columns = ['val_loss'])
        self.hist_accuracy_train = pd.DataFrame(self.history.history['accuracy'],columns = ['train_accuracy'])
        self.hist_accuracy_val = pd.DataFrame(self.history.history['val_accuracy'],columns = ['val_accuracy'])
        self.accuracy = self.hist_accuracy_train.join(self.hist_accuracy_val)
        self.loss = self.train_loss.join(self.val_loss)                
    def predict(self):

        train_dense_results = self.model.evaluate(self.x_train, np.asarray(self.y_train), verbose=2, batch_size=256)
        test_dense_results = self.model.evaluate(self.x_test, self.y_test)
        self.accuracy_train = train_dense_results[1] *100
        self.accuracy_test = test_dense_results[1] *100
                        
            
class Model:
    
    
    
    def __init__(self,
                 data:DataModel,
                 model_name : str= "KNN",
                 model_args: dict = {}
                 ):
        
        
        self.data = data 
        self.model_name = model_name
        self.model_args = model_args
        
    
    def fit(self):
        
        if self.model_name == 'knn' :
            self.model_args = knn_args
            self.model = KnnModel(self.data.x_train,
                                  self.data.y_train,
                                  self.data.x_test,
                                  self.data.y_test,
                                  self.model_args)
            
        elif self.model_name == 'nb' :
            self.model_args = {}
            self.model = NaiveBayesModel(self.data.x_train,
                                  self.data.y_train,
                                  self.data.x_test,
                                  self.data.y_test,
                                  self.model_args)
            
        elif self.model_name == 'logreg' :
            self.model_args = {}
            self.model = LogReg(self.data.x_train,
                                  self.data.y_train,
                                  self.data.x_test,
                                  self.data.y_test,
                                  self.model_args)
        
        elif self.model_name == 'xgb' :
            self.model_args = xgb_args
            self.model = XGBModel(self.data.x_train,
                                  self.data.y_train,
                                  self.data.x_test,
                                  self.data.y_test,
                                  self.model_args)
        elif self.model_name == "mlp":
            self.model_args = mlp_args
            self.model = MLPModel(self.data.training_padded,
                        self.data.y_train,
                        self.data.testing_padded,
                        self.data.y_test,
                        self.model_args)
        self.model.fit()
        
    def predict(self):
        
        self.model.predict()
        
    
        
    def compute_results(self):
        
        self.results = {'accuracy_train':self.model.accuracy_train,
                        'accuracy_test':self.model.accuracy_test,
                        'model': self.model_name,
                        "custom": 1 if self.data.custom else 0,
                        "missing_names": self.data.use_enhanced }
        
        
        
        
        
        
        
        
    
        