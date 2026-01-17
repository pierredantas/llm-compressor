# src/pruning/prune.py
from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


LinearTarget = Tuple[nn.Module, str]


def _linear_weight_targets(model: nn.Module) -> List[LinearTarget]:
    """Coleta todos os (módulo Linear, 'weight') do modelo."""
    targets: List[LinearTarget] = []
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            targets.append((mod, "weight"))
    if not targets:
        raise ValueError("Nenhuma camada nn.Linear encontrada para pruning.")
    return targets


def prune_model_magnitude(
    model: nn.Module,
    amount: float = 0.30,
    remove_pruned: bool = True,
) -> tuple[nn.Module, Dict[str, float]]:
    """
    Pruning global não-estruturado por magnitude (L1) nos pesos de TODAS as nn.Linear.
    - amount: fração de pesos a zerar globalmente (ex.: 0.3 => 30%)
    - remove_pruned: se True, remove a parametrização de pruning (torna permanente)
    Retorna (modelo_prunado, relatorio_sparsity)
    """
    model.eval()

    targets = _linear_weight_targets(model)
    # pruning global por magnitude L1 dos PESOS das Linear (viés não é prunado)
    prune.global_unstructured(
        targets,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Relatório de sparsity por camada + global
    report: Dict[str, float] = {}
    total_params, total_zeros = 0, 0
    idx = 0
    for m, _ in targets:
        # se ainda não removeu, há weight_mask
        if hasattr(m, "weight_mask"):
            mask = m.weight_mask
            zeros = mask.numel() - mask.count_nonzero()
            numel = mask.numel()
        else:
            w = m.weight
            zeros = (w == 0).sum()
            numel = w.numel()

        layer_name = f"Linear[{idx}]"
        sparsity = float(zeros) / float(numel)
        report[layer_name] = sparsity

        total_params += int(numel)
        total_zeros += int(zeros)
        idx += 1

    report["global_sparsity"] = total_zeros / max(1, total_params)

    if remove_pruned:
        # torna o prune permanente (remove weight_orig/weight_mask)
        for m, _ in targets:
            prune.remove(m, "weight")

    return model, report


def pretty_print_prune_report(report: Dict[str, float]) -> None:
    """Imprime sparsity por camada e global."""
    for k, v in report.items():
        if k == "global_sparsity":
            continue
        print(f" - {k}: {v:.2%} zeros")
    print(f" => Global sparsity: {report.get('global_sparsity', 0.0):.2%}")


def prune_pair_magnitude(
    original: nn.Module,
    denoised: nn.Module,
    amount: float = 0.30,
    remove_pruned: bool = True,
) -> tuple[nn.Module, nn.Module, Dict[str, float], Dict[str, float]]:
    """
    Utilitário caso você tenha as duas variantes (original e denoised) e
    queira aplicar exatamente o mesmo esquema de pruning em ambas.
    """
    original_p, rep_o = prune_model_magnitude(original, amount=amount, remove_pruned=remove_pruned)
    denoised_p, rep_d = prune_model_magnitude(denoised, amount=amount, remove_pruned=remove_pruned)
    return original_p, denoised_p, rep_o, rep_d
