import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import numpy as np
import nltk
import nltk
from nltk.corpus import stopwords
import string
from sklearn.metrics import f1_score, precision_score, recall_score

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
punctuation_list = string.punctuation


def clean_text(text):
    if text is not None:
        # Tokenize the text by words and filter stopwords
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # Remove punctuation and special characters, also filter out single characters
        cleaned_text = " ".join(
            word
            for word in filtered_words
            if word not in punctuation_list and len(word) > 1
        )

        # Remove any remaining punctuation from each word
        cleaned_text = "".join(
            char for char in cleaned_text if char not in punctuation_list
        )
    else:
        cleaned_text = None
    return cleaned_text


def plot_confusion_matrix_from_df(df):
    """
    This function plots a confusion matrix based on the PRED and TRUE columns of the dataframe.
    Assumes 0 is 'femme' and 1 is 'homme'.
    """

    true_labels = df["TRUE"]
    pred_labels = df["PRED"]

    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["femme", "homme"],
        yticklabels=["femme", "homme"],
    )
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()


def compute_metrics(df):
    precision = precision_score(df["TRUE"], df["PRED"])
    recall = recall_score(df["TRUE"], df["PRED"])
    f1 = f1_score(df["TRUE"], df["PRED"])
    return precision, recall, f1


def calculate_accuracy_by_gender(df):
    """
    This function calculates and returns the accuracy for each class ('femme' and 'homme') in percentage.
    """
    cm = confusion_matrix(df["TRUE"], df["PRED"], labels=[0, 1])
    acc_femme = cm[0, 0] / sum(cm[0]) * 100  # Accuracy for femme
    acc_homme = cm[1, 1] / sum(cm[1]) * 100  # Accuracy for homme
    return acc_femme, acc_homme


def check_or_create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")


def augment_data(data, target_column: str = "target", n_samples_class: int = 1000):
    target_class = data[target_column].unique().tolist()
    results = []
    for target in target_class:
        data_class_augmented = augment_class_data(data, target, n_samples_class)
        results.append(data_class_augmented)

    return pd.concat(results)


def augment_class_data(data, target_class: int = 0, n_samples: int = 1000):
    subclass_df = data[data["target"] == target_class]
    # unique_values_per_column = {col: subclass_df[col].dropna().unique().tolist() for col in subclass_df.columns if col}
    unique_values_per_column = {
        col: subclass_df[col].unique().tolist() for col in subclass_df.columns if col
    }

    def create_synthetic_sample(unique_values):
        synthetic_sample = []
        for column, values in unique_values.items():
            synthetic_sample.append(np.random.choice(values))
        out = pd.DataFrame(synthetic_sample).T
        out.columns = list(unique_values_per_column.keys())
        return out

    all = []
    for i in range(n_samples):
        all.append(create_synthetic_sample(unique_values_per_column))

    return pd.concat(all, axis=0)


def build_path(custom: bool = True, use_enhanced: str = None):
    root_path = "bert_models"
    custom_str = str(custom)
    use_enhanced_str = use_enhanced if use_enhanced is not None else "none"

    subdir_path = "bert_" + use_enhanced_str + "_" + custom_str + ".pth"
    dir_full_path = os.path.join(root_path, subdir_path)
    path_weights = os.path.join(dir_full_path, "model.safetensors")
    path_config = os.path.join(dir_full_path, "config.json")
    return dir_full_path
