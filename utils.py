import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns


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