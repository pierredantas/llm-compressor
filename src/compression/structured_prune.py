from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from transformers.modeling_utils import prune_linear_layer


@dataclass
class HeadsPruneReport:
    per_layer_pruned: Dict[int, List[int]]
    total_pruned: int


@dataclass
class MLPPruneReport:
    per_layer_kept: Dict[int, int]
    per_layer_pruned: Dict[int, int]
    total_pruned: int


def _get_bert_encoder_layers(model: nn.Module) -> List[nn.Module]:
    """
    Suporta: BertForSequenceClassification (model.bert.encoder.layer),
             RobertaForSequenceClassification (model.roberta.encoder.layer), etc.
    """
    if hasattr(model, "bert"):
        return list(model.bert.encoder.layer)
    if hasattr(model, "roberta"):
        return list(model.roberta.encoder.layer)
    raise ValueError("Modelo não parece ser BERT/RoBERTa-like (faltou .bert ou .roberta).")


def structured_prune_attention_heads(
    model: nn.Module,
    amount: float = 0.2,
    strategy: str = "l1",  # "l1" ou "random"
    seed: int = 42,
) -> Tuple[nn.Module, HeadsPruneReport]:
    """
    Structured pruning de attention heads usando API do Transformers.

    amount: fração de heads removidas por camada (ex.: 0.2 remove 20% de heads em cada layer).
    strategy:
      - "random": remove heads aleatórias
      - "l1": remove heads com menor norma L1 (aprox. por QKV/out proj)
    """
    layers = _get_bert_encoder_layers(model)
    torch.manual_seed(seed)

    per_layer: Dict[int, List[int]] = {}
    total_pruned = 0

    for li, layer in enumerate(layers):
        attn = layer.attention.self
        n_heads = int(attn.num_attention_heads)
        if n_heads <= 1:
            per_layer[li] = []
            continue

        n_prune = int(round(n_heads * amount))
        n_prune = max(0, min(n_prune, n_heads - 1))  # deixa pelo menos 1 head
        if n_prune == 0:
            per_layer[li] = []
            continue

        if strategy == "random":
            heads = torch.randperm(n_heads)[:n_prune].tolist()

        elif strategy == "l1":
            # Heurística: score por head usando pesos de query/key/value (e output.dense)
            # Shapes típicos: (all_head_size, hidden) para Q/K/V em BERT
            head_size = int(attn.attention_head_size)
            # projeções
            Wq = attn.query.weight.detach()
            Wk = attn.key.weight.detach()
            Wv = attn.value.weight.detach()

            # Algumas variantes usam output em layer.attention.output.dense
            Wout = layer.attention.output.dense.weight.detach()

            scores = torch.zeros(n_heads, dtype=torch.float32)
            for h in range(n_heads):
                a, b = h * head_size, (h + 1) * head_size
                # soma L1 dos blocos do head
                scores[h] = (
                    Wq[a:b].abs().sum()
                    + Wk[a:b].abs().sum()
                    + Wv[a:b].abs().sum()
                    + Wout[:, a:b].abs().sum()
                )

            # prune os menores scores
            heads = torch.argsort(scores)[:n_prune].tolist()

        else:
            raise ValueError("strategy deve ser 'random' ou 'l1'")

        # Transformers API: prune_heads recebe set/list de heads
        layer.attention.prune_heads(set(heads))

        per_layer[li] = heads
        total_pruned += len(heads)

    return model, HeadsPruneReport(per_layer_pruned=per_layer, total_pruned=total_pruned)


def structured_prune_mlp_neurons(
    model: nn.Module,
    amount: float = 0.3,
    strategy: str = "l1",  # "l1" ou "random"
    seed: int = 42,
) -> Tuple[nn.Module, MLPPruneReport]:
    """
    Structured pruning do FFN (MLP) removendo neurônios do intermediate.
    Em BERT: intermediate.dense: hidden -> intermediate
             output.dense: intermediate -> hidden

    amount: fração de neurônios removidos por camada (ex.: 0.3 remove 30% do intermediate).
    """
    layers = _get_bert_encoder_layers(model)
    torch.manual_seed(seed)

    kept: Dict[int, int] = {}
    pruned: Dict[int, int] = {}
    total_pruned = 0

    for li, layer in enumerate(layers):
        inter: nn.Linear = layer.intermediate.dense
        out: nn.Linear = layer.output.dense

        inter_size = inter.out_features
        if inter_size <= 1:
            kept[li] = inter_size
            pruned[li] = 0
            continue

        n_prune = int(round(inter_size * amount))
        n_prune = max(0, min(n_prune, inter_size - 1))
        if n_prune == 0:
            kept[li] = inter_size
            pruned[li] = 0
            continue

        if strategy == "random":
            prune_idx = torch.randperm(inter_size)[:n_prune]
        elif strategy == "l1":
            # score por neurônio do intermediate:
            # - rows de inter.weight correspondem a neurônios do intermediate
            # - cols de out.weight correspondem às mesmas unidades (entrada do out)
            W_inter = inter.weight.detach()          # [inter, hidden]
            b_inter = inter.bias.detach() if inter.bias is not None else None
            W_out = out.weight.detach()              # [hidden, inter]

            scores = W_inter.abs().sum(dim=1) + W_out.abs().sum(dim=0)
            if b_inter is not None:
                scores = scores + b_inter.abs()

            prune_idx = torch.argsort(scores)[:n_prune]
        else:
            raise ValueError("strategy deve ser 'random' ou 'l1'")

        # índices que vamos manter
        mask = torch.ones(inter_size, dtype=torch.bool)
        mask[prune_idx] = False
        keep_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

        # 1) prune intermediate.dense na dimensão de saída (dim=0 -> remove neurônios)
        layer.intermediate.dense = prune_linear_layer(inter, keep_idx, dim=0)

        # 2) prune output.dense na dimensão de entrada (dim=1 -> remove mesmas unidades)
        layer.output.dense = prune_linear_layer(out, keep_idx, dim=1)

        kept[li] = int(keep_idx.numel())
        pruned[li] = int(n_prune)
        total_pruned += int(n_prune)

    return model, MLPPruneReport(per_layer_kept=kept, per_layer_pruned=pruned, total_pruned=total_pruned)


def pretty_print_heads_report(report: HeadsPruneReport) -> None:
    print(f"=> Total heads pruned: {report.total_pruned}")
    for li in sorted(report.per_layer_pruned.keys()):
        heads = report.per_layer_pruned[li]
        if heads:
            print(f" - Layer {li}: pruned heads = {heads}")


def pretty_print_mlp_report(report: MLPPruneReport) -> None:
    print(f"=> Total MLP neurons pruned: {report.total_pruned}")
    for li in sorted(report.per_layer_kept.keys()):
        print(f" - Layer {li}: kept={report.per_layer_kept[li]} pruned={report.per_layer_pruned[li]}")
