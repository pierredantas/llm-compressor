from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from src.compression.utils import should_compress_linear



@dataclass
class LayerLowRankReport:
    name: str
    orig_shape: Tuple[int, int]
    k: int
    orig_params: int
    new_params: int


@dataclass
class RMTLowRankReport:
    layers: List[LayerLowRankReport]
    total_orig_params: int
    total_new_params: int

    @property
    def compression_ratio_params(self) -> float:
        return self.total_orig_params / max(1, self.total_new_params)


class LowRankLinear(nn.Module):
    """
    Aproxima uma Linear W (out x in) como W ≈ A @ B, onde:
      A: (out x k)
      B: (k x in)

    forward: x -> (x @ B.T) @ A.T + bias
    """
    def __init__(self, in_features: int, out_features: int, k: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        self.A = nn.Parameter(torch.empty(out_features, k))
        self.B = nn.Parameter(torch.empty(k, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in)
        y = x.matmul(self.B.t())          # (..., k)
        y = y.matmul(self.A.t())          # (..., out)
        if self.bias is not None:
            y = y + self.bias
        return y


def _iter_named_linear(model: nn.Module) -> Iterator[Tuple[str, nn.Linear]]:
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            yield name, mod


def _get_parent(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    """
    Retorna (parent_module, child_attr_name) para poder fazer setattr e substituir o módulo.
    """
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _should_apply(name: str, apply_to: str) -> bool:
    if apply_to == "all_linear":
        return True
    if apply_to == "bert_linear_only":
        return name.startswith("bert.")
    if apply_to == "exclude_classifier":
        # geralmente classifier é "classifier" ou "score" dependendo do modelo
        if name.endswith("classifier") or ".classifier." in name or name == "classifier":
            return False
        if name.endswith("score") or ".score." in name or name == "score":
            return False
        return True
    return True


# ---------------------------
# Parte RMT (estilo notebook)
# ---------------------------

def _column_standardize(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # W: (m, n)
    mu = W.mean(dim=0)                  # (n,)
    sd = W.std(dim=0, unbiased=False)   # (n,)
    safe_sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    WMP = (W - mu.unsqueeze(0)) / safe_sd.unsqueeze(0)
    return WMP, mu, safe_sd


def _mp_support(m: int, n: int, phi: float = 0.15, ca: float = 1.5) -> Tuple[float, float]:
    # garante q <= 1 (se m>n, você poderia transpor; aqui só ajusta pelo min/max)
    p = min(m, n)
    q = min(m, n) / max(m, n)

    lam_minus = (1.0 - q**0.5) ** 2
    lam_plus  = (1.0 + q**0.5) ** 2

    # trimming margin (heurístico do notebook)
    beta = 1.0
    tw_scaled_term = ca * (p ** (-2.0 / 3.0)) * ((1.0 + beta**0.5) ** (4.0 / 3.0))
    delta_beta = max(phi * (lam_plus - lam_minus), tw_scaled_term)

    L = lam_minus + delta_beta
    U = lam_plus - delta_beta
    # garante intervalo válido
    if not (L < U):
        L, U = lam_minus, lam_plus
    return float(L), float(U)


def _shrink_svd(W: torch.Tensor, L: float, U: float, method: str, tau: float = 0.1) -> torch.Tensor:
    """
    Aplica shrinkage nos valores singulares de W.
    Observação: L,U são limites para autovalores do covariance -> usamos sqrt(L), sqrt(U) para singular values.
    """
    # W: (m, n)
    U_svd, s, Vh = torch.linalg.svd(W, full_matrices=False)

    s_shrunk = s.clone()
    sqrt_L = (L ** 0.5)
    sqrt_U = (U ** 0.5)

    if method == "bulk_shrink":
        mask = (s >= sqrt_L) & (s <= sqrt_U)
        if mask.any():
            mu = s[mask].mean()
            s_shrunk[mask] = mu

    elif method == "hard_threshold":
        s_shrunk = torch.clamp(s_shrunk, min=sqrt_L, max=sqrt_U)

    elif method == "soft_threshold":
        s_shrunk = torch.clamp(s_shrunk - tau, min=0.0)

    elif method == "gavish_donoho":
        m, n = W.shape
        beta = min(m, n) / max(m, n)
        omega = (4.0 / (3.0 ** 0.5)) * (beta ** 0.5)
        thresh = omega * s.median()
        s_shrunk = torch.where(s >= thresh, s, torch.zeros_like(s))

    else:
        raise ValueError(f"Unknown shrinkage method: {method}")

    # reconstrói
    return (U_svd * s_shrunk.unsqueeze(0)) @ Vh


def _choose_k_from_singulars(s: torch.Tensor, mode: str, energy: float, ratio: float, k_fixed: int) -> int:
    r = int(s.numel())
    if r <= 1:
        return 1

    if mode == "fixed":
        return max(1, min(k_fixed, r))

    if mode == "ratio":
        k = int(round(r * ratio))
        return max(1, min(k, r))

    # mode == "energy"
    # energia acumulada: sum(s^2)
    e = (s ** 2)
    total = e.sum().clamp_min(1e-12)
    cdf = torch.cumsum(e, dim=0) / total
    k = int(torch.searchsorted(cdf, torch.tensor(energy, device=s.device)).item()) + 1
    return max(1, min(k, r))


def compress_model_rmt_lowrank(
    model: nn.Module,
    *,
    denoise: bool = True,
    shrink_method: str = "gavish_donoho",
    k_mode: str = "energy",
    energy: float = 0.99,
    rank_ratio: float = 0.25,
    k_fixed: int = 64,
    apply_to: str = "exclude_classifier",
    layers: list[int] | None = None,
    k_min: int | None = None,
    k_max: int | None = None,
) -> Tuple[nn.Module, RMTLowRankReport]:
    """
    Para cada nn.Linear selecionada:
      - (opcional) padroniza por coluna + shrinkage via SVD (estilo notebook)
      - faz SVD e aproxima por rank-k
      - substitui a Linear por LowRankLinear (2 matrizes menores)
    """
    model = model.to("cpu").eval()
    torch.set_grad_enabled(False)

    reports: List[LayerLowRankReport] = []
    total_orig, total_new = 0, 0

    for name, lin in list(_iter_named_linear(model)):
        if not should_compress_linear(name, apply_to, layers=layers):
            continue

        W = lin.weight.detach().to(torch.float32)  # (out, in)
        b = lin.bias.detach().to(torch.float32) if lin.bias is not None else None
        out_f, in_f = W.shape

        # 1) (opcional) denoise no estilo WMP
        if denoise:
            WMP, mu, sd = _column_standardize(W)
            L, U = _mp_support(out_f, in_f)
            WMP_hat = _shrink_svd(WMP, L, U, method=shrink_method)
            W_hat = (WMP_hat * sd.unsqueeze(0)) + mu.unsqueeze(0)
        else:
            W_hat = W

        # 2) low-rank factorization
        U_svd, s, Vh = torch.linalg.svd(W_hat, full_matrices=False)
        k = _choose_k_from_singulars(s, mode=k_mode, energy=energy, ratio=rank_ratio, k_fixed=k_fixed)

        if k_min is not None:
            k = max(k, int(k_min))
        if k_max is not None:
            k = min(k, int(k_max))

        U_k = U_svd[:, :k]
        s_k = s[:k]
        V_k = Vh[:k, :]

        # W ≈ (U_k * s_k) @ V_k
        A = (U_k * s_k.unsqueeze(0))        # (out, k)
        B = V_k                              # (k, in)

        # 3) troca o módulo
        lr = LowRankLinear(in_features=in_f, out_features=out_f, k=k, bias=(b is not None))
        lr.A.data.copy_(A)
        lr.B.data.copy_(B)
        if b is not None:
            lr.bias.data.copy_(b)

        parent, attr = _get_parent(model, name)
        setattr(parent, attr, lr)

        # report
        orig_params = out_f * in_f + (out_f if b is not None else 0)
        new_params = out_f * k + k * in_f + (out_f if b is not None else 0)

        total_orig += orig_params
        total_new += new_params

        reports.append(LayerLowRankReport(
            name=name,
            orig_shape=(out_f, in_f),
            k=k,
            orig_params=orig_params,
            new_params=new_params,
        ))

    return model, RMTLowRankReport(
        layers=reports,
        total_orig_params=total_orig,
        total_new_params=total_new,
    )
