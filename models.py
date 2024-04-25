import pandas as pd
import numpy as np
import random

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

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

from tqdm import tqdm

from bert_model import TextDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
import torch

from modelling import (
    KnnModel,
    NaiveBayesModel,
    LogReg,
    XGBModel,
    BertModel,
    MLPModel,
    Benchmark,
)
from utils import check_or_create_directory

nltk.download("stopwords")
french_stopwords = stopwords.words("french")


class Benchmark:
    """
    Provides a baseline model for text classification tasks, using heuristic-based methods for prediction. This class is intended for quick performance assessments against more complex models.

    Attributes:
        x_train (pd.DataFrame): Training dataset containing text features.
        y_train (np.array): Training labels.
        x_test (pd.DataFrame): Testing dataset containing text features.
        y_test (np.array): Testing labels.
    """

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: np.array,
        x_test: pd.DataFrame,
        y_test: np.array,
    ):
        """
        Initializes the Benchmark class with training and testing data.

        Parameters:
            x_train (pd.DataFrame): Training dataset containing text features.
            y_train (np.array): Training labels.
            x_test (pd.DataFrame): Testing dataset containing text features.
            y_test (np.array): Testing labels.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def pred_benchmark(self, x):
        """
        Predicts labels based on keyword presence in the text.

        Parameters:
            x (str): Text input for which the label needs to be predicted.

        Returns:
            int: Predicted label (1 for 'homme', 0 for 'femme', or random for others).
        """
        if "homme" in x:
            return 1
        elif "femme" in x:
            return 0
        else:
            return round(random.random())

    def fit(self):
        """
        Applies the prediction logic to the training dataset and calculates training accuracy.
        """
        self.y_pred_train = self.x_train.apply(self.pred_benchmark)
        self.accuracy_train = 100 * accuracy_score(self.y_train, self.y_pred_train)

    def predict(self):
        """
        Applies the prediction logic to the testing dataset and calculates testing accuracy.
        """
        self.y_pred_test = self.x_test.apply(self.pred_benchmark)
        self.accuracy_test = accuracy_score(self.y_test, self.y_pred_test) * 100
        self.table_results = pd.DataFrame(
            {"PRED": self.y_pred_test, "TRUE": self.y_test}
        )


class NaiveBayesModel:
    """
    This class encapsulates a Naive Bayes model for classification tasks, utilizing a machine learning pipeline that integrates text vectorization,
    TF-IDF transformation, and Naive Bayes classification. It supports training and testing phases, including hyperparameter tuning through grid search to optimize model performance.

    Input to create the class:
    x_train: pd.DataFrame - Training feature dataset.
    y_train: np.array - Training label dataset.
    x_test: pd.DataFrame - Testing feature dataset.
    y_test: np.array - Testing label dataset.

    Purpose of the class:
    The NaiveBayesModel class is designed to provide a structured approach for applying the Naive Bayes algorithm to text classification problems.
    It aims to streamline the process of model training, hyperparameter optimization, and evaluation, ensuring efficient handling of text data for predictive modeling.

    Objectives:
    1. To set up a machine learning pipeline that prepares text data via vectorization and TF-IDF transformation for classification using a Naive Bayes classifier.
    2. To implement grid search for optimizing the hyperparameters of the Naive Bayes classifier, focusing on parameters like smoothing (alpha) and whether to fit class prior probabilities.
    3. To train the classifier using the provided training data and validate its performance using a cross-validation approach within the training set.
    4. To predict labels for new, unseen data (testing data) and evaluate the model's accuracy on this data, facilitating a clear assessment of how well the model generalizes.
    5. To produce a structured output comparing the predicted labels with the true labels, allowing for detailed performance analysis and reporting of the classification accuracy.
    """

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: np.array,
        x_test: pd.DataFrame,
        y_test: np.array,
    ):
        """
        Initializes the NaiveBayesModel class with training and testing data.

        Parameters:
            x_train (pd.DataFrame): The training feature dataset.
            y_train (np.array): The training label dataset.
            x_test (pd.DataFrame): The testing feature dataset.
            y_test (np.array): The testing label dataset.
        """
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
        """
        Trains the Naive Bayes classifier using grid search optimization.
        """
        self.grid_search = GridSearchCV(
            self.pipeline, self.cv_params, scoring="accuracy", cv=5, verbose=2
        )
        self.grid_search.fit(self.x_train, self.y_train)
        self.model = self.grid_search.best_estimator_
        self.y_pred_train = self.model.predict(self.x_train)
        self.accuracy_train = 100 * accuracy_score(self.y_train, self.y_pred_train)

    def predict(self):
        """
        Predicts labels for the testing dataset and calculates accuracy metrics.
        """
        # Predict on the test set
        self.y_pred_test = self.model.predict(self.x_test)
        self.accuracy_test = accuracy_score(self.y_test, self.y_pred_test) * 100
        self.table_results = pd.DataFrame(
            {"PRED": self.y_pred_test, "TRUE": self.y_test}
        )


class KnnModel:

    """
    This class encapsulates a k-nearest neighbors (KNN) model for classification tasks, employing a machine learning pipeline that includes text vectorization,
    TF-IDF transformation, and KNN classification. It is designed to handle the training and testing phases of the machine learning process, including hyperparameter tuning via grid search.

    Input to create the class:
    x_train: pd.DataFrame - Training feature dataset.
    y_train: np.array - Training label dataset.
    x_test: pd.DataFrame - Testing feature dataset.
    y_test: np.array - Testing label dataset.

    Purpose of the class:
    The goal of the KnnModel class is to provide an integrated environment for setting up, training, and evaluating a k-nearest neighbors classifier specifically tailored for text data. It aims to facilitate easy experimentation with different configurations of the KNN algorithm and preprocessing techniques to achieve optimal classification accuracy.

    Objectives:
    1. To construct a machine learning pipeline that preprocesses text data through vectorization and TF-IDF transformation before applying KNN classification.
    2. To utilize grid search to find the best hyperparameters for the KNN classifier within the pipeline, optimizing for accuracy.
    3. To train the KNN model using the specified training data and evaluate its performance on both training and testing datasets.
    4. To provide a straightforward interface for fitting the model to the training data and predicting labels for new, unseen data (testing data).
    5. To track and report the accuracy of the model on both training and testing datasets, allowing for a clear evaluation of the model's performance.
    """

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: np.array,
        x_test: pd.DataFrame,
        y_test: np.array,
    ):
        """
        Initializes the KnnModel class with training and testing data.

        Parameters:
            x_train (pd.DataFrame): The training feature dataset.
            y_train (np.array): The training label dataset.
            x_test (pd.DataFrame): The testing feature dataset.
            y_test (np.array): The testing label dataset.
        """
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
        """
        Trains the KNN classifier using grid search optimization.
        """
        self.grid_search = GridSearchCV(
            self.pipeline, self.cv_params, scoring="accuracy", cv=5, verbose=2
        )
        self.grid_search.fit(self.x_train, self.y_train)
        self.model = self.grid_search.best_estimator_
        self.y_pred_train = self.model.predict(self.x_train)
        self.accuracy_train = 100 * accuracy_score(self.y_train, self.y_pred_train)

    def predict(self):
        """
        Predicts labels for the testing dataset and calculates accuracy metrics.
        """
        self.y_pred_test = self.model.predict(self.x_test)
        self.accuracy_test = accuracy_score(self.y_test, self.y_pred_test) * 100
        self.table_results = pd.DataFrame(
            {"PRED": self.y_pred_test, "TRUE": self.y_test}
        )


class LogReg:
    """
    Implements logistic regression for binary classification tasks on textual data.

    Attributes:
        x_train (pd.DataFrame): Training feature dataset.
        y_train (np.array): Training label dataset.
        x_test (pd.DataFrame): Testing feature dataset.
        y_test (np.array): Testing label dataset.
    """

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: np.array,
        x_test: pd.DataFrame,
        y_test: np.array,
    ):
        """
        Initializes the LogReg class with training and testing data.

        Parameters:
            x_train (pd.DataFrame): The training feature dataset.
            y_train (np.array): The training label dataset.
            x_test (pd.DataFrame): The testing feature dataset.
            y_test (np.array): The testing label dataset.
        """
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
                ),
            ]
        )

    def fit(self):
        """
        Trains the logistic regression model using grid search optimization.
        """
        self.grid_search = GridSearchCV(
            self.pipeline, self.cv_params, scoring="accuracy", cv=5, verbose=2
        )
        self.grid_search.fit(self.x_train, self.y_train)
        self.model = self.grid_search.best_estimator_
        self.y_pred_train = self.model.predict(self.x_train)
        self.accuracy_train = 100 * accuracy_score(self.y_train, self.y_pred_train)

    def predict(self):
        """
        Predicts labels for the testing dataset and calculates accuracy metrics.
        """
        self.y_pred_test = self.model.predict(self.x_test)
        self.accuracy_test = accuracy_score(self.y_test, self.y_pred_test) * 100
        self.table_results = pd.DataFrame(
            {"PRED": self.y_pred_test, "TRUE": self.y_test}
        )


class XGBModel:
    """
    Implements an XGBoost classifier within a machine learning pipeline for text classification tasks.

    Attributes:
        x_train (pd.DataFrame): Training feature dataset.
        y_train (np.array): Training label dataset.
        x_test (pd.DataFrame): Testing feature dataset.
        y_test (np.array): Testing label dataset.
    """

    def __init__(
        self,
        x_train: pd.DataFrame,
        y_train: np.array,
        x_test: pd.DataFrame,
        y_test: np.array,
    ):
        """
        Initializes the XGBModel with training and testing data.

        Parameters:
            x_train (pd.DataFrame): The training feature dataset.
            y_train (np.array): The training label dataset.
            x_test (pd.DataFrame): The testing feature dataset.
            y_test (np.array): The testing label dataset.
        """
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
        """
        Trains the XGBoost model using grid search optimization.
        """
        self.grid_search = GridSearchCV(
            self.pipeline, self.cv_params, scoring="accuracy", cv=5, verbose=2
        )
        self.grid_search.fit(self.x_train, self.y_train)
        self.model = self.grid_search.best_estimator_
        self.y_pred_train = self.model.predict(self.x_train)
        self.accuracy_train = 100 * accuracy_score(self.y_train, self.y_pred_train)

    def predict(self):
        """
        Predicts labels for the testing dataset and calculates accuracy metrics.
        """
        self.y_pred_test = self.model.predict(self.x_test)
        self.accuracy_test = accuracy_score(self.y_test, self.y_pred_test) * 100
        self.table_results = pd.DataFrame(
            {"PRED": self.y_pred_test, "TRUE": self.y_test}
        )


class MLPModel:
    """
    Implements a multi-layer perceptron (MLP) for binary or multi-class classification of textual data using deep learning techniques.
    The class is designed to accommodate a variety of textual data complexities through a customizable neural network architecture.

    Attributes:
        x_train (array-like): Training features, typically pre-processed text data in numeric format.
        y_train (array): Training labels.
        x_test (array-like): Testing features similar in format to x_train.
        y_test (array): Testing labels.
        model_args (dict): Parameters for model configuration including vocabulary size, embedding size, number of layers, and others.
    """

    def __init__(self, x_train, y_train, x_test, y_test, model_args: dict):
        """
        Initializes the MLPModel with training and testing data along with model configuration arguments.

        Parameters:
            x_train (array-like): The training data features.
            y_train (array): The training data labels.
            x_test (array-like): The testing data features.
            y_test (array): The testing data labels.
            model_args (dict): A dictionary containing configurations for the neural network.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model_args = model_args

    def fit(self):
        """
        Constructs and trains the MLP using specified parameters in model_args. Configures layers, activation functions, and training conditions.
        """
        # Model architecture setup
        self.vocab_size = self.model_args["vocab_size"]
        self.embed_size = self.model_args["embed_size"]
        self.n_layers = self.model_args["n_layers"]
        self.hidden_size = self.model_args["hidden_size"]
        self.output_dim = 1  # default for binary classification
        self.dropout_rate = self.model_args["dropout_rate"]
        self.lr = self.model_args["lr"]
        self.max_document_length = self.model_args["max_document_length"]
        self.hidden_function = self.model_args["hidden_function"]

        # Model construction
        self.model = Sequential(
            [
                Input(shape=(self.max_document_length,)),
                Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.embed_size,
                    input_length=self.max_document_length,
                ),
                Flatten(),
            ]
        )

        for _ in range(self.n_layers):
            self.model.add(Dense(self.hidden_size, activation=self.hidden_function))
            self.model.add(Dropout(self.dropout_rate))

        self.model.add(
            Dense(
                self.output_dim,
                activation="sigmoid" if self.output_dim == 1 else "softmax",
            )
        )

        # Compile the model
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy"
            if self.output_dim == 1
            else "categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Training
        early_stop = EarlyStopping(monitor="val_loss", patience=20)
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.model_args["num_epochs"],
            validation_split=0.2,
            batch_size=self.model_args["batch_size"],
            callbacks=[early_stop],
            verbose=2,
        )

        # Saving training history for evaluation
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
        """
        Evaluates the trained model on the training and testing datasets and generates predictions.
        """
        # Evaluation
        train_dense_results = self.model.evaluate(
            self.x_train, np.asarray(self.y_train), verbose=2, batch_size=256
        )
        test_dense_results = self.model.evaluate(self.x_test, self.y_test)
        self.accuracy_train = train_dense_results[1] * 100
        self.accuracy_test = test_dense_results[1] * 100

        # Prediction
        self.y_pred = self.model.predict(self.x_test).flatten()
        self.table_results = pd.DataFrame([self.y_pred, self.y_test]).T
        self.table_results.columns = ["PRED", "TRUE"]
        self.table_results["PRED"] = self.table_results["PRED"].apply(
            lambda x: 1 if x > 0.5 else 0
        )
        self.table_results = self.table_results.astype(int)
        self.table_results


class BertModel:
    """
    Provides a framework for sequence classification using the BERT (Bidirectional Encoder Representations from Transformers) model.
    This class is designed to leverage pre-trained BERT models for accurate text classification, adapting to both simple and complex language contexts.

    Attributes:
        data (DataModel): An instance of DataModel containing dataset configurations and pre-processed data.
    """

    def __init__(self, data: DataModel):
        """
        Initializes the BertModel class with a specific DataModel instance.
        Sets up data loaders and model paths, and loads the BERT model onto the specified device.

        Parameters:
            data (DataModel): The data configuration and datasets for training and testing.
        """
        self.data = data
        self.test = TextDataset(data.test)
        self.test_loader = DataLoader(self.test, batch_size=self.data.test.shape[0])
        self.model_file_path = build_path(data.custom, data.use_enhanced)
        self.device = "cpu"
        self.load_model()

    def load_model(self):
        """
        Loads a pre-trained BERT model from a specified file path and allocates it to the defined device.
        """
        self.model_trained = BertForSequenceClassification.from_pretrained(
            self.model_file_path
        )
        self.model_trained.to(self.device)

    def fit(self):
        """
        Placeholder for the fit method, indicating the model is pre-trained and only needs adjustment or fine-tuning.
        Outputs the location of the model results.
        """
        print(f"Bert model already fitted, results in {self.model_file_path}")

    def predict(self):
        """
        Executes prediction on the test set using the pre-trained BERT model.
        Collects predictions and true labels to compute model accuracy.
        """
        test_all_predictions = []
        test_all_true_labels = []
        self.model_trained.eval()
        for i, data in enumerate(self.test_loader):
            if i >= 1000:
                print("Prediction limit reached")
                break
            targets = data["targets"].to(self.device)
            mask = data["attention_mask"].to(self.device)
            ids = data["input_ids"].to(self.device)
            with torch.no_grad():
                loss, logits = self.model_trained(
                    ids, token_type_ids=None, attention_mask=mask, labels=targets
                ).to_tuple()
                test_all_predictions.extend(
                    np.argmax(logits.cpu().detach().numpy(), axis=1).flatten()
                )
                test_all_true_labels.extend(targets.cpu().numpy())

        self.accuracy_test = (
            accuracy_score(test_all_true_labels, test_all_predictions) * 100
        )
        self.table_results = pd.DataFrame(
            {"PRED": test_all_predictions, "TRUE": test_all_true_labels}
        )


class Model:
    """
    A flexible model class that supports multiple machine learning models for text classification tasks.
    It manages the model lifecycle including initialization, training, prediction, and results handling.

    Attributes:
        data (DataModel): An instance of DataModel containing training and testing datasets.
        model_name (str): A string indicating the type of model to be used. Default is "KNN".
        model_args (dict): A dictionary containing model-specific arguments. Default is an empty dictionary.
    """

    def __init__(self, data: DataModel, model_name: str = "KNN", model_args: dict = {}):
        """
        Initializes the Model class with specified data, model name, and model arguments.
        It also builds the file path for saving model results based on model configuration.
        """
        self.data = data
        self.model_name = model_name
        self.model_args = model_args
        self.build_path()

    def build_path(self):
        """Constructs the file path for storing model results and ensures the directory exists."""
        self.model_path = os.path.join("results", self.model_name)
        check_or_create_directory(self.model_path)
        self.data_path = (
            f"{self.model_name}_{self.data.custom}_{self.data.use_enhanced}"
        )
        self.data_path = os.path.join(self.model_path, self.data_path)

    def fit(self):
        """
        Fits the specified model using the data provided. The method handles different models
        by switching through model names and initializing corresponding model classes with relevant arguments.
        """
        if self.model_name == "knn":
            self.model = KnnModel(
                self.data.x_train, self.data.y_train, self.data.x_test, self.data.y_test
            )
        elif self.model_name == "nb":
            self.model = NaiveBayesModel(
                self.data.x_train, self.data.y_train, self.data.x_test, self.data.y_test
            )
        elif self.model_name == "logreg":
            self.model = LogReg(
                self.data.x_train, self.data.y_train, self.data.x_test, self.data.y_test
            )
        elif self.model_name == "xgb":
            self.model = XGBModel(
                self.data.x_train, self.data.y_train, self.data.x_test, self.data.y_test
            )
        elif self.model_name == "mlp":
            self.model = MLPModel(
                self.data.training_padded,
                self.data.y_train,
                self.data.testing_padded,
                self.data.y_test,
                self.model_args,
            )
        elif self.model_name == "bert":
            self.model = BertModel(self.data)
        elif self.model_name == "benchmark":
            self.model = Benchmark(
                self.data.x_train, self.data.y_train, self.data.x_test, self.data.y_test
            )
        self.model.fit()

    def predict(self):
        """Executes the prediction method of the specified model."""
        self.model.predict()

    def compute_results(self):
        """
        Computes and stores the accuracy results for both training and testing datasets.
        Results are saved into an Excel file at the specified data path.
        """
        self.results = pd.DataFrame(
            [
                [
                    self.model.accuracy_train,
                    self.model.accuracy_test,
                    self.model_name,
                    1 if self.data.custom else 0,
                    self.data.use_enhanced,
                ]
            ],
            columns=[
                "accuracy_train",
                "accuracy_test",
                "model",
                "custom",
                "missing_names",
            ],
        ).T
        self.model.table_results.to_excel(self.data_path + ".xlsx")
        print("results computed and table results saved")
