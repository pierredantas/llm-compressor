import evaluate
from bert_score import score
from sklearn.metrics import accuracy_score, f1_score
# from summac.model_summac import SummaCConv


def compute_rouge(predictions: list, references: list) -> dict:
    """
    Computes ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for text summarization evaluation.
    """
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)

    return {
        "ROUGE-1": results["rouge1"],
        "ROUGE-2": results["rouge2"],
        "ROUGE-L": results["rougeL"],
    }


def compute_bleu(predictions: list, references: list) -> float:
    """
    Computes BLEU score for text summarization evaluation.
    """
    bleu = evaluate.load("bleu")
    tokenized_preds = predictions  # BLEU expects raw text
    tokenized_refs = [[ref] for ref in references]  # Needs nested structure

    results = bleu.compute(predictions=tokenized_preds, references=tokenized_refs)
    return results["bleu"]


def compute_meteor(predictions: list, references: list) -> float:
    """
    Computes METEOR score for text summarization evaluation.
    """
    meteor = evaluate.load("meteor")
    results = meteor.compute(predictions=predictions, references=references)
    return results["meteor"]


def compute_bertscore(predictions: list, references: list, model_type="bert-base-uncased") -> float:
    """
    Computes BERTScore for text summarization evaluation.
    """
    P, R, F1 = score(predictions, references, lang="en", model_type=model_type)
    return F1.mean().item()


# def compute_summac(predictions: list, references: list) -> float:
#     """
#     Computes SummaC score (consistency check) for text summarization evaluation.
#     """
#     model = SummaCConv(granularity="sentence", model_path="vitc")
#     scores = model.score(references, predictions)
#     return sum(scores["scores"]) / len(scores["scores"])


def evaluate_summaries(predictions: list, references: list):
    """
    Runs all evaluation metrics on a set of generated summaries.
    """
    print("🔹 Evaluating Summaries...")

    results = {
        "ROUGE": compute_rouge(predictions, references),
        "BLEU": compute_bleu(predictions, references),
        "METEOR": compute_meteor(predictions, references),
        "BERTScore": compute_bertscore(predictions, references),
        # "SummaC": compute_summac(predictions, references),
    }

    # Print results
    for metric, value in results.items():
        print(f"✅ {metric}: {value}")

    return results


def compute_accuracy(preds, labels) -> float:
    return float(accuracy_score(labels, preds))

def compute_f1_binary(preds, labels) -> float:
    # SST-2 é binário: 0 = negative, 1 = positive
    return float(f1_score(labels, preds, average="binary"))

def evaluate_glue_task(task: str, preds, labels) -> dict:
    acc = compute_accuracy(preds, labels)
    out = {"accuracy": acc}

    # MRPC/QQP: tradicionalmente reporta F1 também
    if task in {"mrpc", "qqp"}:
        out["f1"] = compute_f1_binary(preds, labels)

    # SST-2: normalmente só accuracy; se quiser, pode incluir F1 como extra
    if task == "sst2":
        out["f1"] = compute_f1_binary(preds, labels)

    print("✅ " + " | ".join([f"{k}: {v:.4f}" for k, v in out.items()]))
    return out

# Example Usage
if __name__ == "__main__":
    generated_summaries = ["The cat is on the mat."]
    reference_summaries = ["A cat sat on the mat."]

    scores = evaluate_summaries(generated_summaries, reference_summaries)
