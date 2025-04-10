import torch
from torch.utils.data import Dataset


class CausalLMDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        """
        Args:
            file_path (str): Path to the text file.
            tokenizer (AutoTokenizer): Hugging Face tokenizer.
            max_length (int): Maximum tokenized sequence length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()
        text = text.encode("latin-1").decode("utf-8", errors="ignore")
        # Tokenize the entire text corpus
        tokenized_text = tokenizer(
            text,
            # return_tensors="pt",
            truncation=True,
            # padding=True,
            max_length=max_length,  # Ensure it fits model constraints
        )["input_ids"]

        # Split into chunks of max_length
        self.samples = [
            tokenized_text[i : i + max_length]
            for i in range(0, len(tokenized_text), max_length)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids = self.samples[idx]
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

        # Labels are the same as input_ids, but padding tokens are replaced with -100 (ignored in loss)
        labels = input_ids.copy()
        labels = [
            -100 if token == self.tokenizer.pad_token_id else token for token in labels
        ]

        return_dict = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        return return_dict
