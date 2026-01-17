import io
import re
from pathlib import Path
import os
import torch
import torch.nn as nn


def sizeof_state_dict_bytes(model: nn.Module) -> int:
    """
    Mede o tamanho (bytes) do state_dict quando salvo.
    - Só move para CPU se o item for Tensor.
    - Mantém itens não-tensor (ex.: torch.dtype, packed params) como estão.
    """
    cpu_state = {}
    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor):
            cpu_state[k] = v.detach().cpu()
        else:
            cpu_state[k] = v  # dtype, ints, objetos serializáveis do quantized
    buf = io.BytesIO()
    try:
        torch.save(cpu_state, buf)
        return buf.tell()
    except Exception:
        # fallback robusto: serializa o modelo inteiro (pode ficar maior, mas não quebra)
        buf = io.BytesIO()
        torch.save(model, buf)
        return buf.tell()


def file_size_bytes(path: str | Path) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return p.stat().st_size


def human_readable(nbytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(nbytes)
    for u in units:
        if size < 1024 or u == units[-1]:
            return f"{size:.2f} {u}"
        size /= 1024.0


def _layer_index_from_name(name: str) -> int | None:
    m = re.search(r"\.encoder\.layer\.(\d+)\.", name)
    return int(m.group(1)) if m else None

def should_compress_linear(
    name: str,
    apply_to: str,
    layers: list[int] | None = None,
) -> bool:
    # filtra por camada, se fornecido
    li = _layer_index_from_name(name)
    if layers is not None and li is not None and li not in layers:
        return False

    if apply_to == "all_linear":
        return True

    if apply_to == "bert_linear_only":
        return name.startswith("bert.") and (".dense" in name or ".query" in name or ".key" in name or ".value" in name)

    if apply_to == "exclude_classifier":
        return not name.startswith("classifier") and not name.endswith(".classifier") and "classifier" not in name

    # ✅ novos
    if apply_to == "ffn_only":
        return (
            ".intermediate.dense" in name
            or ".output.dense" in name
        ) and "attention" not in name  # evita confundir output.dense do attention

    if apply_to == "ffn_plus_attn_output":
        return (
            ".intermediate.dense" in name
            or (".output.dense" in name and "attention" not in name)
            or ".attention.output.dense" in name
        )

    if apply_to == "qkv_only":
        return ".attention.self.query" in name or ".attention.self.key" in name or ".attention.self.value" in name

    raise ValueError(f"apply_to inválido: {apply_to}")

