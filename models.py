import pandas as pd
import numpy as np
from modelling import *
from utils import *
from tqdm import tqdm

## for knn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

## for xgb
import nltk
from nltk.corpus import stopwords
import xgboost as xgb

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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

nltk.download("stopwords")
french_stopwords = stopwords.words("french")


## for bert
from bert_model import TextDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
import torch


class KnnModel:
    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: np.array,
        x_test: pd.DataFrame,
        y_test: np.array,
    ):
        self.cv_params = {
            "clf__n_neighbors": [3, 5, 7],
            "clf__weights": ["uniform", "distance"],
        }
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.pipeline = Pipeline(
            [
                ("vect", CountVectorizer(stop_words=french_stopwords)),
                ("tfidf", TfidfTransformer()),
                ("clf", KNeighborsClassifier()),
            ]
        )

    def fit(self):
        self.grid_search = GridSearchCV(
            self.pipeline, self.cv_params, scoring="accuracy", cv=5, verbose=2
        )
        self.grid_search.fit(self.x_train, self.y_train)
        self.model = self.grid_search.best_estimator_
        self.y_pred_train = self.model.predict(self.x_train)
        self.accuracy_train = accuracy_score(self.y_train, self.y_pred_train)

    def predict(self):
        self.y_pred_test = self.model.predict(self.x_test)
        self.accuracy_test = accuracy_score(self.y_test, self.y_pred_test) * 100
        self.table_results = pd.DataFrame(
            {"PRED": self.y_pred_test, "TRUE": self.y_test}
        )


class NaiveBayesModel:
    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: np.array,
        x_test: pd.DataFrame,
        y_test: np.array,
    ):
        self.cv_params = {
            "clf__alpha": [0.5, 1.0, 1.5],  # Smoothing parameter
            "clf__fit_prior": [
                True,
                False,
            ],  # Whether to learn class prior probabilities or assume uniform
        }
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # Define the pipeline
        self.pipeline = Pipeline(
            [
                ("vect", CountVectorizer(stop_words=french_stopwords)),
                ("tfidf", TfidfTransformer()),
                ("clf", naive_bayes.BernoulliNB()),
            ]
        )

    def fit(self):
        self.grid_search = GridSearchCV(
            self.pipeline, self.cv_params, scoring="accuracy", cv=5, verbose=2
        )
        self.grid_search.fit(self.x_train, self.y_train)
        self.model = self.grid_search.best_estimator_
        self.y_pred_train = self.model.predict(self.x_train)
        self.accuracy_train = accuracy_score(self.y_train, self.y_pred_train)

    def predict(self):
        # Predict on the test set
        self.y_pred_test = self.model.predict(self.x_test)
        self.accuracy_test = accuracy_score(self.y_test, self.y_pred_test) * 100
        self.table_results = pd.DataFrame(
            {"PRED": self.y_pred_test, "TRUE": self.y_test}
        )


class LogReg:
    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: np.array,
        x_test: pd.DataFrame,
        y_test: np.array,
    ):
        self.cv_params = {"clf__C": [0.1, 1, 10], "clf__penalty": ["l1", "l2"]}
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.pipeline = Pipeline(
            [
                ("vect", CountVectorizer(stop_words=french_stopwords)),
                ("tfidf", TfidfTransformer()),
                (
                    "clf",
                    LogisticRegression(solver="liblinear"),
                ),  # liblinear is a good choice for small datasets and binary classification
            ]
        )

    def fit(self):
        self.grid_search = GridSearchCV(
            self.pipeline, self.cv_params, scoring="accuracy", cv=5, verbose=2
        )
        self.grid_search.fit(self.x_train, self.y_train)
        self.model = self.grid_search.best_estimator_
        self.y_pred_train = self.model.predict(self.x_train)
        self.accuracy_train = accuracy_score(self.y_train, self.y_pred_train)

    def predict(self):
        self.y_pred_test = self.model.predict(self.x_test)
        self.accuracy_test = accuracy_score(self.y_test, self.y_pred_test) * 100
        self.table_results = pd.DataFrame(
            {"PRED": self.y_pred_test, "TRUE": self.y_test}
        )


class XGBModel:
    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: np.array,
        x_test: pd.DataFrame,
        y_test: np.array,
    ):
        self.cv_params = {
            "clf__learning_rate": [0.1, 1e-2, 1e-3],
            "clf__max_depth": [3, 5, 7, 10],
        }
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.pipeline = Pipeline(
            [
                ("vect", CountVectorizer(stop_words=french_stopwords)),
                ("tfidf", TfidfTransformer()),
                ("clf", xgb.XGBClassifier()),
            ]
        )

    def fit(self):
        self.grid_search = GridSearchCV(
            self.pipeline, self.cv_params, scoring="accuracy", cv=5, verbose=2
        )
        self.grid_search.fit(self.x_train, self.y_train)
        self.model = self.grid_search.best_estimator_
        self.y_pred_train = self.model.predict(self.x_train)
        self.accuracy_train = accuracy_score(self.y_train, self.y_pred_train)

    def predict(self):
        self.y_pred_test = self.model.predict(self.x_test)
        self.accuracy_test = accuracy_score(self.y_test, self.y_pred_test) * 100
        self.table_results = pd.DataFrame(
            {"PRED": self.y_pred_test, "TRUE": self.y_test}
        )


class MLPModel:
    def __init__(self, x_train, y_train, x_test, y_test, model_args: dict):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_args = model_args

    def fit(self):
        self.vocab_size = self.model_args["vocab_size"]
        self.embed_size = self.model_args["embed_size"]
        self.n_layers = self.model_args["n_layers"]
        self.hidden_size = self.model_args["hidden_size"]
        self.output_dim = 1
        self.dropout_rate = self.model_args["dropout_rate"]
        self.lr = self.model_args["lr"]
        self.max_document_length = self.model_args["max_document_length"]
        self.hidden_function = self.model_args["hidden_function"]
        self.model = Sequential()
        self.model.add(
            Input(shape=(self.max_document_length,))
        )  # input layer specifying the input shape
        self.model.add(
            Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embed_size,
                input_length=self.max_document_length,
            )
        )
        self.model.add(
            Flatten()
        )  # flattening the output of the embedding layer to fit into dense layers
        for _ in range(self.n_layers):
            self.model.add(Dense(self.hidden_size, activation=self.hidden_function))
            self.model.add(Dropout(self.dropout_rate))
        self.model.add(
            Dense(
                self.output_dim,
                activation="sigmoid" if self.output_dim == 1 else "softmax",
            )
        )  # for binary classification use 'sigmoid', for multi-class use 'softmax'

        # Compile the self.
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy"
            if self.output_dim == 1
            else "categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.num_epochs = self.model_args["num_epochs"]
        self.batch_size = self.model_args["batch_size"]
        early_stop = EarlyStopping(monitor="val_loss", patience=4)
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.num_epochs,
            validation_split=0.2,
            batch_size=self.batch_size,
            callbacks=[early_stop],
            verbose=2,
        )

        self.train_loss = pd.DataFrame(
            self.history.history["loss"], columns=["train_loss"]
        )
        self.val_loss = pd.DataFrame(
            self.history.history["val_loss"], columns=["val_loss"]
        )
        self.hist_accuracy_train = pd.DataFrame(
            self.history.history["accuracy"], columns=["train_accuracy"]
        )
        self.hist_accuracy_val = pd.DataFrame(
            self.history.history["val_accuracy"], columns=["val_accuracy"]
        )
        self.accuracy = self.hist_accuracy_train.join(self.hist_accuracy_val)
        self.loss = self.train_loss.join(self.val_loss)

    def predict(self):
        train_dense_results = self.model.evaluate(
            self.x_train, np.asarray(self.y_train), verbose=2, batch_size=256
        )
        test_dense_results = self.model.evaluate(self.x_test, self.y_test)
        self.accuracy_train = train_dense_results[1] * 100
        self.accuracy_test = test_dense_results[1] * 100
        self.y_pred = self.model.predict(self.x_test).flatten()
        self.table_results = pd.DataFrame([self.y_pred, self.y_test]).T
        self.table_results.columns = ["PRED", "TRUE"]
        self.table_results["PRED"] = self.table_results["PRED"].apply(
            lambda x: 1 if x > 0.5 else 0
        )
        self.table_results = self.table_results.astype(int)
        self.table_results


class BertModel:
    def __init__(self, data: DataModel):
        self.data = data
        self.test = TextDataset(data.test)
        self.test_loader = DataLoader(self.test, batch_size=self.data.test.shape[0])
        self.model_file_path = build_path(data.custom, data.use_enhanced)
        self.device = "cpu"
        self.load_model()

    def load_model(self):
        self.model_trained = BertForSequenceClassification.from_pretrained(
            self.model_file_path
        )
        self.model_trained.to(self.device)

    def fit(self):
        print(f"Bert model already fitted, results in {self.model_file_path}")
        return

    def predict(self):
        test_all_predictions = []
        test_all_true_labels = []
        self.model_trained.eval()
        epoch_loss_test = []
        test_all_predictions = []
        test_all_true_labels = []
        s = 0
        for data in self.test_loader:
            s += 1
            if s <= 1000:
                targets = data["targets"].to(self.device)
                mask = data["attention_mask"].to(self.device)
                ids = data["input_ids"].to(self.device)
                with torch.no_grad():
                    loss, logits = self.model_trained(
                        ids, token_type_ids=None, attention_mask=mask, labels=targets
                    ).to_tuple()

                    epoch_loss_test.append(loss.item())
                    cpu_logits = logits.cpu().detach().numpy()
                    test_all_predictions.extend(np.argmax(cpu_logits, axis=1).flatten())
                    test_all_true_labels.extend(targets.cpu().numpy())
            else:
                print("done")
                break
        self.accuracy_test = (
            accuracy_score(test_all_true_labels, test_all_predictions) * 100
        )
        self.accuracy_train = 1 * 100
        self.table_results = pd.DataFrame(
            {"PRED": test_all_predictions, "TRUE": test_all_true_labels}
        )


class Model:
    def __init__(self, data: DataModel, model_name: str = "KNN", model_args: dict = {}):
        self.data = data
        self.model_name = model_name
        self.model_args = model_args

    def fit(self):
        if self.model_name == "knn":
            self.model_args = knn_args
            self.model = KnnModel(
                self.data.x_train, self.data.y_train, self.data.x_test, self.data.y_test
            )

        elif self.model_name == "nb":
            self.model_args = {}
            self.model = NaiveBayesModel(
                self.data.x_train, self.data.y_train, self.data.x_test, self.data.y_test
            )

        elif self.model_name == "logreg":
            self.model_args = {}
            self.model = LogReg(
                self.data.x_train, self.data.y_train, self.data.x_test, self.data.y_test
            )

        elif self.model_name == "xgb":
            self.model_args = xgb_args
            self.model = XGBModel(
                self.data.x_train, self.data.y_train, self.data.x_test, self.data.y_test
            )
        elif self.model_name == "mlp":
            self.model_args = mlp_args
            self.model = MLPModel(
                self.data.training_padded,
                self.data.y_train,
                self.data.testing_padded,
                self.data.y_test,
                self.model_args,
            )
        elif self.model_name == "bert":
            self.model = BertModel(self.data)

        self.model.fit()

    def predict(self):
        self.model.predict()

    def compute_results(self):
        self.acc_man, self.acc_women = calculate_accuracy_by_gender(
            self.model.table_results
        )
        self.results = pd.DataFrame(
            [
                self.model.accuracy_train,
                self.model.accuracy_test,
                self.model_name,
                1 if self.data.custom else 0,
                self.data.use_enhanced,
                self.acc_man,
                self.acc_women,
            ]
        ).T

        self.results.columns = [
            "accuracy_train",
            "accuracy_test",
            "model",
            "custom",
            "missing_names",
            "acc_man",
            "acc_women",
        ]
