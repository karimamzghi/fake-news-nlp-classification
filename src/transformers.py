import torch
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased"


def combine_title_and_text(data):
    data = data.copy()

    data["combined_text"] = (
        data["title"].fillna("").astype(str)
        + " "
        + data["text"].fillna("").astype(str)
    ).str.strip()

    return data


def load_distilbert_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_texts(texts, tokenizer, max_length=256):
    return tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length
    )


class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.tolist()

    def __getitem__(self, index):
        item = {
            key: torch.tensor(values[index])
            for key, values in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item

    def __len__(self):
        return len(self.labels)


def create_distilbert_dataset(data, labels, tokenizer, max_length=256):
    data = combine_title_and_text(data)

    encodings = tokenize_texts(
        data["combined_text"],
        tokenizer,
        max_length=max_length
    )

    return FakeNewsDataset(encodings, labels)
