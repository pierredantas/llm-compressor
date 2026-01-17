from typing import Dict, Any
from torch.utils.data import Dataset
from datasets import load_dataset

class MrpcDataset(Dataset):
    def __init__(self, tokenizer, split: str = "validation", max_length: int = 128):
        self.dataset = load_dataset("glue", "mrpc", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        s1, s2 = item["sentence1"], item["sentence2"]
        enc = self.tokenizer(
            s1, s2,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        out: Dict[str, Any] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": int(item["label"]),  # 0 = not paraphrase, 1 = paraphrase
        }
        if "token_type_ids" in enc:
            out["token_type_ids"] = enc["token_type_ids"]
        return out
