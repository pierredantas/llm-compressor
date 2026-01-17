import argparse
import time

import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from src.config.config import Config
from src.datasets import Sst2Dataset, MrpcDataset, QqpDataset
from src.metrics.metrics import evaluate_glue_task

from src.compression.utils import sizeof_state_dict_bytes, file_size_bytes, human_readable
from src.compression.quantization import (
    quantize_dynamic_torch,
    export_and_quantize_onnx_dynamic,
    onnx_logits,
)
from src.compression.structured_prune import (
    structured_prune_attention_heads,
    structured_prune_mlp_neurons,
    pretty_print_heads_report,
    pretty_print_mlp_report,
)
from src.compression.rmt_lowrank import compress_model_rmt_lowrank



def load_config(file_path: str) -> Config:
    with open(file_path, "r") as f:
        return Config(**yaml.safe_load(f))

def build_dataset(task: str, tokenizer, split: str, max_len: int):
    if task == "sst2":
        return Sst2Dataset(tokenizer, split=split, max_length=max_len)
    if task == "mrpc":
        return MrpcDataset(tokenizer, split=split, max_length=max_len)
    if task == "qqp":
        return QqpDataset(tokenizer, split=split, max_length=max_len)
    raise ValueError(f"Tarefa desconhecida: {task}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Caminho para o arquivo de configuração YAML")
    args = parser.parse_args()

    config = load_config(args.config)

    # A ordem recomendada: PRUNE -> QUANTIZE
    # device inicial só considera quantização se ela estiver habilitada (abaixo, após prune)
    device = torch.device("cuda" if torch.cuda.is_available() and config.device.cuda else "cpu")
    print(f"Using device: {device}")

    repo_id = config.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, num_labels=2).to(device).eval()

    base_bytes = sizeof_state_dict_bytes(model)
    print(f"[SIZE] Base (FP32 state_dict): {human_readable(base_bytes)}")

    # # ===== PRUNING (se habilitado) =====
    # if config.prune and config.prune.magnitude and config.prune.magnitude.enable:
    #     args_p = config.prune.magnitude.args or {}
    #     amount = float(args_p.get("prune_amount", 0.30))
    #     remove_pruned = bool(args_p.get("remove_pruned", True))
    #
    #     print(f"-> Applying global unstructured magnitude pruning on Linear weights ({amount:.0%}).")
    #     model, prune_report = prune_model_magnitude(model, amount=amount, remove_pruned=remove_pruned)
    #     pretty_print_prune_report(prune_report)
    #
    #     pruned_bytes = sizeof_state_dict_bytes(model)
    #     print(f"[SIZE] After pruning (FP32 state_dict, zeros included): {human_readable(pruned_bytes)}")
    #     if pruned_bytes == base_bytes:
    #         print("  (como esperado: pruning não-estruturado não reduz o tamanho do arquivo em PyTorch)")

    # ===== RMT + LOW-RANK (se habilitado) =====
    if config.rmt_lowrank and config.rmt_lowrank.enable:
        print("-> Applying RMT denoise + Low-Rank compression on Linear layers (CPU).")

        model, rep_lr = compress_model_rmt_lowrank(
            model,
            denoise=bool(config.rmt_lowrank.denoise),
            shrink_method=str(config.rmt_lowrank.shrink_method),
            k_mode=str(config.rmt_lowrank.k_mode),
            energy=float(config.rmt_lowrank.energy),
            rank_ratio=float(config.rmt_lowrank.ratio or 0.25),
            k_fixed=int(config.rmt_lowrank.k or 64),
            apply_to=str(config.rmt_lowrank.apply_to),
            layers=(config.rmt_lowrank.layers or None),
            k_min=config.rmt_lowrank.k_min,
            k_max=config.rmt_lowrank.k_max,
        )

        lr_bytes = sizeof_state_dict_bytes(model)
        print(f"[SIZE] After RMT+LowRank (state_dict): {human_readable(lr_bytes)}")
        print(f"[RANK] Layers compressed: {len(rep_lr.layers)} | "
              f"param ratio ≈ {rep_lr.compression_ratio_params:.2f}x")

        # (opcional) mostrar 3 camadas
        for e in rep_lr.layers[:3]:
            print(f"  - {e.name}: W={e.orig_shape} k={e.k} "
                  f"params {e.orig_params}->{e.new_params}")


    # ===== STRUCTURED PRUNING: HEADS =====
    if config.prune and config.prune.attention_heads and config.prune.attention_heads.enable:
        args_h = config.prune.attention_heads.args or {}
        amount = float(args_h.get("prune_amount", 0.2))
        strategy = str(args_h.get("strategy", "l1"))
        seed = int(args_h.get("seed", 42))

        print(f"-> Structured pruning attention heads: amount={amount:.0%}, strategy={strategy}")
        model, rep_h = structured_prune_attention_heads(model, amount=amount, strategy=strategy, seed=seed)
        pretty_print_heads_report(rep_h)

        # (opcional) medir tamanho aqui de novo
        pruned_struct_bytes = sizeof_state_dict_bytes(model)
        print(f"[SIZE] After attention head structured pruning: {human_readable(pruned_struct_bytes)}")

    # ===== STRUCTURED PRUNING: MLP NEURONS =====
    if config.prune and config.prune.mlp_neurons and config.prune.mlp_neurons.enable:
        args_m = config.prune.mlp_neurons.args or {}
        amount = float(args_m.get("prune_amount", 0.3))
        strategy = str(args_m.get("strategy", "l1"))
        seed = int(args_m.get("seed", 42))

        print(f"-> Structured pruning MLP neurons: amount={amount:.0%}, strategy={strategy}")
        model, rep_m = structured_prune_mlp_neurons(model, amount=amount, strategy=strategy, seed=seed)
        pretty_print_mlp_report(rep_m)

        # (opcional) medir tamanho aqui de novo
        pruned_struct_bytes = sizeof_state_dict_bytes(model)
        print(f"[SIZE] After mlp structured pruning: {human_readable(pruned_struct_bytes)}")

    # ===== QUANTIZAÇÃO (se habilitada) =====
    force_cpu = bool(config.quantization and config.quantization.enable)
    use_onnx = False
    ort_session = None

    if config.quantization and config.quantization.enable:
        method = (config.quantization.method or "dynamic").lower()
        if method == "dynamic":
            print("-> Applying PyTorch dynamic INT8 quantization (CPU).")
            model = quantize_dynamic_torch(model)  # move para CPU
            device = torch.device("cpu")

            q_bytes = sizeof_state_dict_bytes(model)
            print(f"[SIZE] After PyTorch dynamic INT8: {human_readable(q_bytes)}")
        elif method == "onnx":
            print("-> Exporting to ONNX and applying dynamic INT8 quantization.")
            paths = export_and_quantize_onnx_dynamic(
                model,
                save_dir=config.trainer.save_model_path,
                opset=17,
                seq_len_for_export=config.data.max_input_length,
            )
            import onnxruntime as ort

            ort_session = ort.InferenceSession(paths["onnx_int8"], providers=["CPUExecutionProvider"])
            use_onnx = True
            device = torch.device("cpu")

            onnx_fp32_size = file_size_bytes(paths["onnx_fp32"])
            onnx_int8_size = file_size_bytes(paths["onnx_int8"])
            print(f"[SIZE] ONNX FP32: {human_readable(onnx_fp32_size)}")
            print(f"[SIZE] ONNX INT8 (dynamic): {human_readable(onnx_int8_size)}")
            red = (1.0 - (onnx_int8_size / max(1, onnx_fp32_size))) * 100.0 if onnx_fp32_size else 0.0
            print(f"  (redução ONNX ≈ {red:.1f}%)")
        else:
            print(f"[WARN] Método de quantização desconhecido: {method}. Pulando quantização.")

    dataset = build_dataset(
        task=config.data.task,
        tokenizer=tokenizer,
        split=config.data.split,
        max_len=config.data.max_input_length,
    )
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_preds, all_labels = [], []
    if use_onnx:
        # ===== INFERÊNCIA VIA ONNX RUNTIME =====
        for batch in tqdm(loader, desc=f"Evaluating {config.data.task.upper()} [ONNX-INT8]"):
            # batch já está em CPU; só coletamos labels
            labels = batch["labels"]
            logits = onnx_logits(
                ort_session,
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("token_type_ids"),
            )
            preds = logits.argmax(axis=-1)  # numpy
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().tolist())
    else:
        WARMUP_STEPS = 10
        MEASURE_STEPS = 50  # ou len(loader) se quiser medir tudo

        all_preds, all_labels = [], []

        # -------- timing helpers --------
        is_cuda = (device.type == "cuda")


        def sync():
            if is_cuda:
                torch.cuda.synchronize()


        warmup_done = 0
        measured = 0
        total_forward_time = 0.0
        total_samples = 0

        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Evaluating {config.data.task.upper()}"):
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                if "token_type_ids" in batch:
                    inputs["token_type_ids"] = batch["token_type_ids"].to(device)

                labels = batch["labels"].to(device)

                # ---- warmup ----
                do_warmup = warmup_done < WARMUP_STEPS
                if do_warmup:
                    warmup_done += 1

                if measured < MEASURE_STEPS and not do_warmup:
                    sync()
                    t0 = time.perf_counter()
                    logits = model(**inputs).logits
                    sync()
                    t1 = time.perf_counter()
                    total_forward_time += (t1 - t0)
                    total_samples += int(labels.shape[0])
                    measured += 1
                else:
                    logits = model(**inputs).logits

                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        # ---- report timing ----
        if measured > 0:
            avg_ms_per_batch = (total_forward_time / measured) * 1000.0
            samples_per_s = total_samples / max(1e-12, total_forward_time)
            print(f"[TIME] avg forward: {avg_ms_per_batch:.2f} ms/batch | throughput: {samples_per_s:.2f} samples/s")

    scores = evaluate_glue_task(config.data.task, all_preds, all_labels)
    print(scores)
