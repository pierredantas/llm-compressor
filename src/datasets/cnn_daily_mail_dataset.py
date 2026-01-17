from torch.utils.data import Dataset
from datasets import load_dataset
import torch


class CNNDailyMailDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_length=512):
        """
        Initializes the dataset.
        :param split: The dataset split to be used ('train', 'test', 'validation').
        :param max_length: The maximum length of input sequences.
        """
        self.dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Returns the size of the dataset."""
        return 10 # len(self.dataset)

    def __getitem__(self, idx):
        """Gets an item from the dataset."""
        article = self.dataset[idx]["article"]
        summary = self.dataset[idx]["highlights"]

        # Tokenize the input text (article) and the output (summary)
        input_text = "summarize: " + article
        target_text = summary

        # Tokenization of article and summary
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=self.max_length, truncation=True,
                                padding="max_length")
        targets = self.tokenizer(target_text, return_tensors="pt", max_length=self.max_length, truncation=True,
                                 padding="max_length")

        # Returns the inputs and targets (summary) as tensors
        return {
            'input_ids': inputs['input_ids'].squeeze(),  # Remove extra dimensions
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()  # T5 model uses 'labels' as output
        }
