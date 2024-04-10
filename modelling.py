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
from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Dense,
    Embedding,
    Dropout,
    GlobalAveragePooling1D,
    Flatten,
    SpatialDropout1D,
    Bidirectional,
)
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
from utils import *
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
from tensorflow.keras.layers import (
    LSTM,
    GRU,
    Dense,
    Embedding,
    Dropout,
    GlobalAveragePooling1D,
    Flatten,
    SpatialDropout1D,
    Bidirectional,
)


##" for results
# import matplotlib.pyplot as plt
# from sklearn.metrics import plot_confusion_matrix
# from sklearn.metrics import confusion_matrix
import seaborn as sns


class DataClass:
    def __init__(
        self,
        use_prediction: bool = True,
        path_dir: str = PATH_DATA_TREATED,
        custom: bool = True,
        use_enhanced: str = "lstm",
        augmented_data: bool = True,
    ):
        self.use_prediction = use_prediction
        self.path_dir = path_dir
        self.custom = custom
        self.use_enhanced = use_enhanced
        self.augment_data = augmented_data
        self.import_data()

    def import_data(self):
        if self.use_prediction:
            path = os.path.join(self.path_dir, "data_prediction.pq")
        else:
            path = os.path.join(self.path_dir, "data_groundtruth.pq")
        self.data = pd.read_parquet(path)

        self.freq_name = pd.read_csv(
            os.path.join(PATH_DATA, "firstname_with_sex.csv"), sep=";"
        )
        self.freq_name["total"] = self.freq_name[["male", "female"]].sum(1)

        if self.use_prediction:
            self.enhanced_index = self.data[
                self.data["firstname_lower"] != self.data["firstname_lower_enhanced"]
            ].index.tolist()
        else:
            self.enhanced_index = []

    def merge_features(self, data, output_col: str = "X", is_train: bool = True):
        self.features = [
            "link",
            "lob",
            "employer",
            "occupation",
            "name_sex",
            "firstname_lower",
        ]
        df = data.copy()
        if self.use_enhanced == "enhanced":
            print("modifying features")
            print(self.features)
            self.features.remove("firstname_lower")
            self.features.append("firstname_lower_enhanced")
            print(self.features)
        elif self.use_enhanced == "lstm":
            print("modifying features")
            print(self.features)
            self.features.remove("firstname_lower")
            self.features.append("firstname_lower_lstm")
            print(self.features)
        elif self.use_enhanced == "fuzzy":
            print("modifying features")
            print(self.features)
            self.features.remove("firstname_lower")
            self.features.append("firstname_lower_fuzzy")
            print(self.features)
        else:
            pass
        df = df[self.features + ["target"]]
        if is_train:
            if self.augment_data:
                print("**************augmenting data set ***********")
                df = augment_data(df, n_samples_class=10000)
            else:
                print("**************augmenting data set diabled ***********")
        df["X"] = (
            df[self.features]
            .fillna("")
            .apply(lambda row: " ".join(row.values.astype(str)), axis=1)
        )
        return df[["X", "target"]].sample(frac=1)

        # return merged

    def create_dataset(self):
        if self.custom:
            test = self.data.loc[self.enhanced_index]
            train = self.data.drop(self.enhanced_index)

        else:
            self.data = self.data.sample(frac=1)
            frac = 0.33
            test = self.data.sample(frac=frac, random_state=42)
            train = self.data.drop(test.index)

        self.train = self.merge_features(data=train, is_train=True)
        self.train.columns = ["message", "label"]
        self.x_train = self.train["message"]
        self.y_train = self.train["label"].astype(int)

        self.test = self.merge_features(data=test, is_train=False)
        self.test.columns = ["message", "label"]
        self.x_test = self.test["message"]
        self.y_test = self.test["label"].astype(int)

    @staticmethod
    def create_unbalanced_dataset(
        data: pd.DataFrame, multiplier: int = 3, class_oversampled: str = "zeros"
    ):
        count_zeros = data["label"].value_counts()[0]
        count_ones = data["label"].value_counts()[1]
        valid = min(count_zeros, count_ones)

        zeros = data[data["label"] == 0].sample(n=valid)
        ones = data[data["label"] == 1].sample(n=valid)
        if class_oversampled == "zeros":
            zeros_df = [zeros for _ in range(multiplier - 1)]
            ones_df = ones
            zeros_df = pd.concat(zeros_df, axis=0)
        else:
            ones_df = [ones for _ in range(multiplier - 1)]
            zeros_df = zeros
            ones_df = pd.concat(ones_df, axis=0)
        unbalanced_data = pd.concat([zeros_df, ones_df], axis=0)
        unbalanced_data["label"].value_counts() / unbalanced_data.shape[0]
        return unbalanced_data


class DataModel:
    def __init__(
        self,
        data: DataClass,
        custom_test_index: list = [],
        max_len: int = 50,
        use_enhanced: str = "lstm",
        custom: bool = False,
        trunc_type: str = "post",
        padding_type: str = "post",
        oov_tok: str = "<OOV>",
        vocab_size: int = 500,
    ):
        self.data = data
        self.x_train = self.data.x_train
        self.y_train = self.data.y_train
        self.x_test = self.data.x_test
        self.y_test = self.data.y_test
        self.train = self.data.train
        self.test = self.data.test
        self.custom_test_index = custom_test_index
        self.max_len = max_len
        self.custom = custom
        self.use_enhanced = use_enhanced
        self.trunc_type = trunc_type
        self.padding_type = padding_type
        self.oov_tok = oov_tok
        self.vocab_size = vocab_size

    def create_tokenizer(self):
        self.tokenizer = Tokenizer(
            num_words=self.vocab_size, char_level=False, oov_token=self.oov_tok
        )

        self.tokenizer.fit_on_texts(self.x_train)

    def create_padding(self):
        self.create_tokenizer()

        self.training_sequences = self.tokenizer.texts_to_sequences(self.x_train)
        self.training_padded = pad_sequences(
            self.training_sequences,
            maxlen=self.max_len,
            padding=self.padding_type,
            truncating=self.trunc_type,
        )

        self.testing_sequences = self.tokenizer.texts_to_sequences(self.x_test)
        self.testing_padded = pad_sequences(
            self.testing_sequences,
            maxlen=self.max_len,
            padding=self.padding_type,
            truncating=self.trunc_type,
        )

        print("Shape of training tensor: ", self.training_padded.shape)
        print("Shape of testing tensor: ", self.testing_padded.shape)
