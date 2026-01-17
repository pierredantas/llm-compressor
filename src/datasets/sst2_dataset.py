from torch.utils.data import Dataset
from datasets import load_dataset

class Sst2Dataset(Dataset):
    def __init__(self, tokenizer, split: str = "validation", max_length: int = 128):
        self.dataset = load_dataset("glue", "sst2", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        text = item["sentence"]
        label = int(item["label"])

        # NÃO use return_tensors aqui; deixe o DataCollatorWithPadding padronizar
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": label,   # int; o collator converte p/ tensor
            # NÃO retorne "text": text  -> isso causa o erro
        }
