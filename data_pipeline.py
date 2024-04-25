from params import *
import pandas as pd
import numpy as np
import yaml
from params import *
import re
import os
from tqdm import tqdm
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import difflib
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from fuzzywuzzy import process


def import_transcriptions():
    """
    Imports transcription data with associated sex information from a CSV file.
    Output: pd.DataFrame - Returns a dataframe containing the transcriptions.
    """
    data = pd.read_csv(os.path.join(PATH_DATA, "transcriptions_with_sex.csv"))
    return data


def read_yaml(file_path):
    """
    Reads and parses a YAML file into a Python object.
    Arguments:
    file_path: str - The path to the YAML file.
    Output: dict - Returns the parsed YAML file content as a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def explode_column(data, column, tags):
    """
    Extracts and separates components of a column into separate columns based on specified tags.
    Arguments:
    data: pd.DataFrame - Dataframe containing the column to explode.
    column: str - The column containing the tagged text to explode.
    tags: list - List of tags used to split the column's content.
    Output: pd.DataFrame - Returns the modified dataframe with new columns for each tag.
    """
    output = data.copy()
    pattern = lambda tag: rf"{tag}: ([^:]+)(?= \w+:|$)"
    for tag in tags:
        output[tag] = output[column].apply(
            lambda x: re.search(pattern(tag), x).group(1)
            if re.search(pattern(tag), x)
            else np.nan
        )
    return output


def rename_predictions(data, mapping):
    """
    Renames columns in a dataframe based on a provided mapping.
    Arguments:
    data: pd.DataFrame - The dataframe to modify.
    mapping: dict - Dictionary mapping old column names to new names.
    Output: pd.DataFrame - Returns the renamed dataframe.
    """
    output = data.copy()
    output = output.rename(columns=mapping)
    return output


def encode_freq(x):
    """
    Encodes the frequency of names based on given thresholds.
    Arguments:
    x: float - The frequency value to encode.
    Output: str - Returns the encoded frequency category as a string.
    """
    if x > 0.5:
        return "prenom homme"
    else:
        return "prenom femme"


def import_freq_name():
    """
    Imports and processes a CSV containing first names and their frequencies by sex.
    Output: pd.DataFrame - Returns a dataframe enriched with frequency calculations and name sex categorization.
    """
    name_freq = pd.read_csv(os.path.join(PATH_DATA, "firstname_with_sex.csv"), sep=";")
    name_freq["total"] = name_freq[["male", "female"]].sum(1)
    name_freq["freq_male"] = name_freq["male"] / name_freq["total"]
    name_freq["freq_female"] = name_freq["female"] / name_freq["total"]
    name_freq["name_sex"] = name_freq["freq_male"].apply(lambda x: encode_freq(x))
    return name_freq


def modify_firstname(data):
    """
    Standardizes the 'firstname' field in a dataframe.
    Arguments:
    data: pd.DataFrame - Dataframe containing the 'firstname' column to be modified.
    Output: pd.DataFrame - Returns the dataframe with modified 'firstname' fields
    """
    data["firstname"] = data["firstname"].apply(
        lambda x: str(x).split(" ")[0] if x else None
    )
    data["firstname"] = data["firstname"].apply(
        lambda x: str(x).split("-")[0] if x else None
    )
    data["firstname"] = data["firstname"].apply(
        lambda x: str(x).replace("ï", "i") if x else None
    )
    data["firstname_lower"] = data["firstname"].apply(
        lambda x: str.lower(x) if x else None
    )
    data["firstname_lower"] = data["firstname_lower"].apply(
        lambda x: x.replace(" ", "") if x else None
    )
    return data


def merge_with_namefreq(data, name_freq):
    """
    Merges a dataframe with a frequency data frame on the 'firstname' key.
    Arguments:
    data: pd.DataFrame - The main dataframe.
    name_freq: pd.DataFrame - The dataframe containing frequency data.
    Output: pd.DataFrame - Returns the merged dataframe.
    """
    merged = data.merge(
        name_freq.set_index("firstname"),
        left_on="firstname_lower",
        right_index=True,
        how="left",
    )
    return merged


def get_list_valid_names(name_freq):
    """
    Retrieves a list of valid names based on specified frequency and threshold conditions.
    Arguments:
    name_freq: pd.DataFrame - Dataframe containing name frequencies.
    Output: list - Returns a list of valid first names.
    """
    valid_names = name_freq[
        ((name_freq["freq_female"] - 0.5).abs() > 0.25) & (name_freq["total"] > 1000)
    ]["firstname"].index.tolist()
    return valid_names


def drop_ambiguous(data):
    data = data[~(data["sex"] == "ambigu")]
    return data


def encode_label(x):
    if x:
        if x == "homme":
            return 1
        if x == "femme":
            return 0
    else:
        return None


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        # Create the directory
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' was created.")
    else:
        print(f"Directory '{dir_path}' already exists.")


def complete_missing_names(data: pd.DataFrame, name_freq: pd.DataFrame):
    data["firstname_lower_enhanced"] = data["firstname_lower"]
    missing_index = data[data["freq_male"].isna()].index
    missing_names = data.loc[missing_index, "firstname_lower"].values.tolist()

    print("**************** COMPLETING MISSING NAMES with DiffLib **************")
    for i in range(len(missing_index)):
        index = missing_index[i]
        name = missing_names[i]
        # similarity = reco.get_recommendations(name).sort_values(by = "similarity",ascending=False)
        new_name = difflib.get_close_matches(
            name, name_freq[name_freq["total"] > 250]["firstname"].unique(), n=2
        )
        if new_name:
            name_sex = name_freq[name_freq["firstname"] == new_name[0]][
                "name_sex"
            ].iloc[0]
            data.loc[index, "firstname_lower_enhanced"] = name_sex + " " + new_name[0]
    print(data.loc[missing_index][["firstname_lower", "firstname_lower_enhanced"]])

    return data


def get_predictions_network(names):
    model = load_model("model_misspelling/my_model.h5")
    with open("model_misspelling/label_encoder.pickle", "rb") as handle:
        loaded_label_encoder = pickle.load(handle)

    with open("model_misspelling/tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    new_texts = names
    new_texts_seq = tokenizer.texts_to_sequences(new_texts)
    new_texts_padded = pad_sequences(new_texts_seq, padding="post", maxlen=8)
    predictions = model.predict(new_texts_padded)
    predicted_labels = predictions.argmax(axis=-1)
    predicted_class_names = loaded_label_encoder.inverse_transform(predicted_labels)
    predicted_class_names
    return predicted_class_names


def complete_missing_names_lstm(data: pd.DataFrame, name_freq: pd.DataFrame):
    data["firstname_lower_lstm"] = data["firstname_lower"]
    missing_index = data[data["freq_male"].isna()].index
    missing_names = data.loc[missing_index, "firstname_lower"].values.tolist()
    predictions = get_predictions_network(missing_names)
    print("**************** COMPLETING MISSING NAMES with LSTM **************")
    for i in range(len(missing_index)):
        index = missing_index[i]
        new_name = predictions[i]
        if new_name:
            name_sex = name_freq[name_freq["firstname"] == new_name]["name_sex"].iloc[0]
            data.loc[index, "firstname_lower_lstm"] = name_sex + " " + new_name
    print(data.loc[missing_index][["firstname_lower", "firstname_lower_lstm"]])

    return data


def complete_missing_names_fuzzy(data: pd.DataFrame, name_freq: pd.DataFrame):
    data["firstname_lower_fuzzy"] = data["firstname_lower"]
    missing_index = data[data["freq_male"].isna()].index
    missing_names = data.loc[missing_index, "firstname_lower"].values.tolist()
    choices = name_freq[name_freq["total"] > 1000]["firstname"].unique()
    print("**************** COMPLETING MISSING NAMES with FuzzyWizzy **************")
    for i in range(len(missing_index)):
        index = missing_index[i]
        name = missing_names[i]
        new_name = process.extract(name, choices=choices, limit=1)[0][0]
        if new_name:
            name_sex = name_freq[name_freq["firstname"] == new_name]["name_sex"].iloc[0]
            data.loc[index, "firstname_lower_fuzzy"] = name_sex + " " + new_name
    print(data.loc[missing_index][["firstname_lower", "firstname_lower_fuzzy"]])

    return data


def pipeline():
    data = import_transcriptions()
    data = drop_ambiguous(data)

    data_groundtruth = explode_column(data, "groundtruth", tags_groundtruth)
    data_prediction = explode_column(data, "prediction", tags_prediction)
    data_prediction = rename_predictions(data_prediction, mapping)

    data_prediction = modify_firstname(data_prediction)
    data_groundtruth = modify_firstname(data_groundtruth)
    data_prediction["target"] = data_prediction["sex"].apply(encode_label)
    data_groundtruth["target"] = data_groundtruth["sex"].apply(encode_label)

    name_freq = import_freq_name()
    data_prediction = merge_with_namefreq(data_prediction, name_freq)
    data_groundtruth = merge_with_namefreq(data_groundtruth, name_freq)
    data_prediction = complete_missing_names(data_prediction, name_freq)
    data_prediction = complete_missing_names_lstm(data_prediction, name_freq)
    data_prediction = complete_missing_names_fuzzy(data_prediction, name_freq)

    make_dir(PATH_DATA_TREATED)
    data_prediction.to_parquet(os.path.join(PATH_DATA_TREATED, "data_prediction.pq"))
    data_groundtruth.to_parquet(os.path.join(PATH_DATA_TREATED, "data_groundtruth.pq"))
