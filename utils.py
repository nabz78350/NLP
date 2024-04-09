import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os 

def plot_confusion_matrix_from_df(df):
    """
    This function plots a confusion matrix based on the PRED and TRUE columns of the dataframe.
    Assumes 0 is 'femme' and 1 is 'homme'.
    """

    true_labels = df['TRUE']
    pred_labels = df['PRED']
    

    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['femme', 'homme'], yticklabels=['femme', 'homme'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def calculate_accuracy_by_gender(df):
    """
    This function calculates and returns the accuracy for each class ('femme' and 'homme') in percentage.
    """
    cm = confusion_matrix(df['TRUE'], df['PRED'], labels=[0, 1])
    acc_femme = cm[0, 0] / sum(cm[0]) * 100  # Accuracy for femme
    acc_homme = cm[1, 1] / sum(cm[1]) * 100  # Accuracy for homme
    return acc_femme, acc_homme


def check_or_create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directory created: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")