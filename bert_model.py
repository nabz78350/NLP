import pandas as pd
from modelling import *
from utils import *

import os
from sklearn.model_selection import train_test_split
import warnings
import torch
from torch.utils.data import Dataset
import time
import datetime
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch import cuda
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.optim import AdamW
from transformers import AutoTokenizer

torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

BATCH_SIZE = 16
EPOCHS = 5
device = "cuda" if cuda.is_available() else "cpu"


class TextDataset(Dataset):
    def __init__(self, dataframe, max_length=180, tokenizer_name="bert-base-uncased"):
        self.texts = dataframe["message"].values
        self.targets = dataframe["label"].values
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        return {
            "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.as_tensor(attention_mask, dtype=torch.long),
            "targets": torch.as_tensor(target, dtype=torch.long),
            "text": text,
        }


def build_model():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )

    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-8)

    return model, optimizer


def split_data(modelling_data):
    train_dataset = TextDataset(modelling_data.train)
    test_dataset = TextDataset(modelling_data.test)
    torch.manual_seed(1702)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)

    return train_loader, test_loader


def train_model(train_loader, output_dir: str = "bert_model"):
    model, optimizer = build_model()
    training_stats = []
    epoch_loss_train = []
    total_t0 = time.time()

    # TRAINING
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        print("")
        print("================ Epoch {:} / {:} ================".format(epoch, EPOCHS))
        train_all_predictions = []
        train_all_true_labels = []
        for step, data in enumerate(train_loader):
            print(step)
            if step % 2 == 0 and not step == 0:
                elapsed = int(round(time.time() - t0))
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print(
                    "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                        step, len(train_loader), elapsed
                    )
                )

            targets = data["targets"].to(device)
            mask = data["attention_mask"].to(device)
            ids = data["input_ids"].to(device)

            model.zero_grad()

            loss, logits = model(
                ids, token_type_ids=None, attention_mask=mask, labels=targets
            ).to_tuple()
            epoch_loss_train.append(loss.item())

            cpu_logits = logits.cpu().detach().numpy()
            train_all_predictions.extend(np.argmax(cpu_logits, axis=1).flatten())
            train_all_true_labels.extend(targets.cpu().numpy())

            loss.backward()
            optimizer.step()
        train_accuracy = accuracy_score(train_all_true_labels, train_all_predictions)
        print(train_accuracy)
        training_stats.append(
            {
                "epoch": epoch,
                "Training Loss": np.mean(epoch_loss_train),
                "Training Accuracy": train_accuracy,
            }
        )

    save_model(model, output_dir)


def save_model(model, output_dir):
    check_or_create_directory(output_dir)
    model_file_path = os.path.join(output_dir, "socface_model.pth")
    model.save_pretrained(model_file_path)


def main():
    for enhanced in ["none", "enhanced", "lstm", "fuzzy"]:
        for custom in [True, False]:
            data_class = DataClass(
                use_prediction=True,
                use_enhanced=enhanced,
                custom=custom,
                augmented_data=False,
                augmented_size=250,
            )
            data_class.create_dataset()
            modelling_data = DataModel(
                data=data_class,
                use_enhanced=data_class.use_enhanced,
                custom_test_index=data_class.enhanced_index,
                custom=data_class.custom,
            )
            modelling_data.create_padding()
            train_loader, test_loader = split_data(modelling_data)
            output_dir = (
                "bert_model_" + data_class.use_enhanced + "_" + str(data_class.custom)
            )
            train_model(train_loader, output_dir)


if __name__ == "__main__":
    main()
