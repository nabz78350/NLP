# Load, explore and plot data
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from params import FEATURES, PATH_DATA_TREATED,PATH_DATA
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





class DataClass :
    
    def __init__(self,use_prediction : bool = True,
                path_dir : str = PATH_DATA_TREATED,
                custom : bool = True,
                use_enhanced : bool = True

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
        
        if self.use_enhanced :
            print('modifying features')
            print(self.features)
            self.features.remove('firstname_lower')
            self.features.append('firstname_lower_enhanced')
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
                custom : bool = False,
                trunc_type : str = 'post',
                padding_type: str  = 'post',
                oov_tok : str = '<OOV>', 
                vocab_size : int = 500):
    
    
        self.data = data
        self.custom_test_index = custom_test_index
        self.max_len = max_len
        self.custom = custom
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
        
        
        
        
class Model(DataModel):
    
    
    def __init__(self,data: pd.DataFrame):
        super().__init__(data = data)