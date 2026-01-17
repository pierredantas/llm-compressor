# src/quantization/quantize.py
from __future__ import annotations
from typing import Dict, Literal, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn


def quantize_dynamic_torch(model: nn.Module) -> nn.Module:
    """
    Quantiza dinamicamente camadas Linear do BERT para INT8 (CPU).
    Ativa ganhos de latência e redução de memória em CPU.

    Observações:
    - Mantém embeddings e LayerNorm em float (correto para Transformers).
    - Precisa rodar em CPU.
    """
    model = model.to("cpu").eval()
    qmodel = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},            # alvo principal p/ BERT
        dtype=torch.qint8       # INT8
    )
    return qmodel


def _export_onnx_fp32(
    model: nn.Module,
    onnx_fp32_path: Path,
    opset: int = 17,
    seq_len: int = 8,
) -> Tuple[str, Tuple[str, ...]]:
    """
    Exporta o modelo de classificação (BERT) para ONNX (FP32).
    Define e nomeia entradas padrão de BERT:
      input_ids, attention_mask, token_type_ids (opcional no forward).
    """
    model = model.to("cpu").eval()

    # dummies
    bsz = 2
    input_ids = torch.ones((bsz, seq_len), dtype=torch.long)
    attention_mask = torch.ones((bsz, seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((bsz, seq_len), dtype=torch.long)

    # ordem dos args segue assinatura do forward do BERT
    args = (input_ids, attention_mask, token_type_ids)

    input_names = ("input_ids", "attention_mask", "token_type_ids")
    output_names = ("logits",)

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "token_type_ids": {0: "batch", 1: "seq"},
        "logits": {0: "batch"},
    }

    torch.onnx.export(
        model,
        args,
        f=str(onnx_fp32_path),
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    return str(onnx_fp32_path), input_names


def export_and_quantize_onnx_dynamic(
    model: nn.Module,
    save_dir: str | Path = "./model",
    opset: int = 17,
    seq_len_for_export: int = 128,
onnxruntime=None) -> Dict[str, str]:
    """
    Exporta p/ ONNX (FP32) e aplica quantização dinâmica INT8 via onnxruntime.
    Retorna caminhos dos modelos ONNX (fp32 e int8).
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as e:
        raise RuntimeError(
            "Precisa instalar onnx e onnxruntime para quantização ONNX: "
            "pip install onnx onnxruntime"
        ) from e

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    onnx_fp32_path = save_dir / "model_fp32.onnx"
    onnx_int8_path = save_dir / "model_int8.onnx"

    _export_onnx_fp32(model, onnx_fp32_path, opset=opset, seq_len=seq_len_for_export)

    # Quantização dinâmica (peso INT8; ativações em tempo de execução)
    quantize_dynamic(
        model_input=str(onnx_fp32_path),
        model_output=str(onnx_int8_path),
        weight_type=QuantType.QInt8,
        per_channel=False,     # per-tensor é o mais compatível
        optimize_model=False,  # pode ligar depois se quiser
    )

    return {
        "onnx_fp32": str(onnx_fp32_path),
        "onnx_int8": str(onnx_int8_path),
    }


def onnx_logits(
    ort_session,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: Optional[torch.Tensor] = None,
):
    """
    Executa inferência no ONNX Runtime e retorna logits (numpy).
    Tensores devem estar em CPU.
    """
    import numpy as np

    inputs: Dict[str, "np.ndarray"] = {
        "input_ids": input_ids.cpu().numpy().astype("int64"),
        "attention_mask": attention_mask.cpu().numpy().astype("int64"),
    }
    if token_type_ids is not None:
        inputs["token_type_ids"] = token_type_ids.cpu().numpy().astype("int64")

    outputs = ort_session.run(None, inputs)
    # pela nossa exportação, outputs[0] = "logits"
    return outputs[0]
