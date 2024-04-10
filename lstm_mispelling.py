import pandas as pd
import numpy as np
import os
import random
from modelling import *
from utils import *
from tqdm import tqdm
import string
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from sklearn.preprocessing import LabelEncoder


def alter_word(word):
    operations = []
    if len(word) < 3:
        possible_ops = ["replace_one", "delete_one"]
    else:
        possible_ops = ["replace_two", "replace_one", "delete_one", "delete_two"]

    for op in possible_ops:
        if op == "replace_one":
            pos = random.randint(0, len(word) - 1)
            new_letter = random.choice(string.ascii_letters)
            new_word = word[:pos] + new_letter + word[pos + 1 :]
            new_word = str.lower(new_word)
            operations.append(new_word)

        elif op == "replace_two":
            pos1, pos2 = random.sample(range(len(word)), 2)
            new_letter1, new_letter2 = random.sample(string.ascii_letters, 2)
            new_word = list(word)
            new_word[pos1], new_word[pos2] = new_letter1, new_letter2

            operations.append(str.lower("".join(new_word)))

        elif op == "delete_one":
            pos = random.randint(0, len(word) - 1)
            new_word = word[:pos] + word[pos + 1 :]
            new_word = str.lower(new_word)
            operations.append(new_word)

        elif op == "delete_two":
            pos1, pos2 = random.sample(range(len(word)), 2)
            new_word = "".join(
                letter for i, letter in enumerate(word) if i not in [pos1, pos2]
            )
            new_word = str.lower(new_word)
            operations.append(new_word)

    return operations


def alter_dataframe(data_class: DataClass):
    data_model_translate = data_class.freq_name[data_class.freq_name["total"] > 1000][
        ["firstname"]
    ]
    data_model_translate["true"] = data_model_translate[["firstname"]]
    data = []
    for ope in tqdm(range(1)):
        for index in range(data_model_translate.shape[0]):
            name, true = data_model_translate.iloc[index]
            altered_name = alter_word(name)
            out = pd.DataFrame([altered_name, [true] * len(altered_name)]).T
            data.append(out)

    data_model = pd.concat(data, axis=0)
    data_model.columns = ["X", "Y"]
    return data_model


def train_model(data_model: pd.DataFrame):
    label_encoder = LabelEncoder()
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>", char_level=True)
    tokenizer.fit_on_texts(data_model["X"])
    X_seq = tokenizer.texts_to_sequences(data_model["X"])
    X_padded = pad_sequences(X_seq, padding="post", maxlen=8)

    Y_encoded = label_encoder.fit_transform(data_model["Y"])
    Y_categorical = to_categorical(Y_encoded)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_padded, Y_categorical, test_size=0.1, random_state=42
    )

    model = Sequential(
        [
            Embedding(1000, 16, input_length=8),
            Bidirectional(LSTM(200, return_sequences=True)),
            Bidirectional(LSTM(200)),
            Dense(200, activation="relu"),
            Dense(Y_categorical.shape[1], activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)

    model.summary()

    output_dir = "model_misspelling"
    check_or_create_directory(output_dir)
    with open(os.path.join(output_dir, "label_encoder.pickle"), "wb") as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_dir, "tokenizer.pickle"), "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    history = model.fit(
        X_train,
        Y_train,
        epochs=20,
        validation_data=(X_test, Y_test),
        validation_split=0.25,
        callbacks=[early_stopping],
        verbose=2,
    )

    model.save(os.path.join(output_dir, "my_model.h5"))


def main():
    data_class = DataClass(use_prediction=True, use_enhanced="no", custom=True)
    data_model = alter_dataframe(data_class)
    train_model(data_model)


if __name__ == "__main__":
    main()
