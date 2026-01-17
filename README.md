# LLM Compressor — BERT Compression Experiments (GLUE)

This repository contains an experimentation pipeline to **compress Transformer models** and measure the **impact on quality and efficiency**.

We evaluate multiple compression strategies on **BERT fine-tuned GLUE checkpoints** (MRPC, SST-2, QQP), comparing:

- **Quality metrics:** Accuracy and F1
- **Efficiency metrics:** average forward time (ms/batch) and throughput (samples/s)
- **Footprint:** serialized model size (MB)

The goal is to understand the trade-offs between **model size / latency** and **task performance**.

---

## Benchmarks & Results

### MRPC (GLUE)
| Variant | Acc ↑ | F1 ↑ | Avg Forward (ms/batch) ↓ | Throughput (samples/s) ↑ | Size (MB) ↓ |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.877 | 0.913 | 672.530 | 35.690 | 417.71 |
| Quantized | 0.838 | 0.878 | 305.790 | 78.480 | 173.07 |
| Pruned (Heads) - 25% | 0.806 | 0.865 | 620.820 | 38.660 | 390.68 |
| Pruned (MLP) - 15% | 0.694 | 0.732 | 609.420 | 39.720 | 385.27 |
| Pruned (Heads 15% + MLP 10%) | 0.380 | 0.181 | 588.710 | 40.770 | 378.09 |
| RMT LowRank | 0.855 | 0.892 | 578.990 | 41.450 | 383.96 |
| Baseline (GPU) | 0.877 | 0.913 | 108.830 | 220.530 | 417.71 |
| Pruned (Heads) - 25% (GPU) | 0.803 | 0.865 | 98.860 | 242.780 | 390.68 |
| Pruned (MLP) - 15% (GPU) | 0.693 | 0.732 | 100.550 | 238.680 | 385.27 |
| Pruned (Heads 15% + MLP 10%) (GPU) | 0.380 | 0.181 | 96.870 | 247.760 | 378.09 |

### SST-2 (GLUE)
| Variant | Acc ↑ | F1 ↑ | Avg Forward (ms/batch) ↓ | Throughput (samples/s) ↑ | Size (MB) ↓ |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.924 | 0.926 | 585.470 | 52.380 | 417.71 |
| Quantized | 0.911 | 0.913 | 245.610 | 124.860 | 173.07 |
| Pruned (Heads) - 25% | 0.911 | 0.913 | 547.040 | 56.060 | 390.68 |
| Pruned (MLP) - 15% | 0.885 | 0.887 | 497.560 | 61.630 | 385.27 |
| Pruned (Heads 15% + MLP 10%) | 0.891 | 0.893 | 484.120 | 63.350 | 378.09 |
| RMT LowRank | 0.906 | 0.911 | 488.950 | 62.720 | 383.96 |
| Baseline (GPU) | 0.924 | 0.926 | 85.730 | 357.720 | 417.71 |
| Pruned (Heads) - 25% (GPU) | 0.910 | 0.913 | 79.890 | 383.880 | 390.68 |
| Pruned (MLP) - 15% (GPU) | 0.885 | 0.887 | 80.040 | 383.160 | 385.27 |
| Pruned (Heads 15% + MLP 10%) (GPU) | 0.891 | 0.893 | 77.810 | 394.120 | 378.09 |

### QQP (GLUE)
| Variant | Acc ↑ | F1 ↑ | Avg Forward (ms/batch) ↓ | Throughput (samples/s) ↑ | Size (MB) ↓ |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.909 | 0.878 | 815.500 | 39.240 | 417.71 |
| Quantized | 0.895 | 0.863 | 325.740 | 98.240 | 173.07 |
| Pruned (Heads) - 25% | 0.874 | 0.845 | 769.530 | 41.580 | 390.68 |
| Pruned (MLP) - 15% | 0.851 | 0.778 | 778.400 | 41.110 | 385.27 |
| Pruned (Heads 15% + MLP 10%) | 0.779 | 0.620 | 675.420 | 47.380 | 378.09 |
| RMT LowRank | 0.892 | 0.853 | 733.840 | 43.640 | 383.96 |
| Baseline (GPU) | 0.909 | 0.878 | 128.360 | 249.300 | 417.71 |
| Pruned (Heads) - 25% (GPU) | 0.873 | 0.845 | 120.000 | 266.670 | 390.68 |
| Pruned (MLP) - 15% (GPU) | 0.851 | 0.778 | 119.310 | 268.220 | 385.27 |
| Pruned (Heads 15% + MLP 10%) (GPU) | 0.779 | 0.620 | 116.620 | 274.410 | 378.09 |

---

## What is being tested?

### 1) Baseline
Loads a Hugging Face model (e.g., `textattack/bert-base-uncased-MRPC`) and evaluates it on the selected GLUE validation split.

### 2) Dynamic INT8 Quantization (CPU)
Uses **PyTorch dynamic quantization** targeting `nn.Linear` layers:
- **Goal:** reduce model size and improve CPU latency/throughput.
- **Note:** dynamic quantization runs on **CPU** (the script automatically switches device to CPU when enabled).

Implementation: `src/compression/quantization.py::quantize_dynamic_torch`

### 3) Structured Pruning (Transformer-aware)
This repo implements **structured pruning** (removal that changes the architecture), which can translate into real compute savings:

**A. Attention head pruning**
- Removes a fraction of attention heads per layer.
- `strategy: "l1"` prunes the lowest L1-norm heads (heuristic over Q/K/V + output projection).
- Parameters: `prune_amount`, `strategy`, `seed`

Implementation: `src/compression/structured_prune.py::structured_prune_attention_heads`

**B. FFN/MLP neuron pruning**
- Removes a fraction of intermediate FFN neurons per layer.
- `strategy: "l1"` prunes the lowest L1-norm neurons (intermediate + output projections).
- Parameters: `prune_amount`, `strategy`, `seed`

Implementation: `src/compression/structured_prune.py::structured_prune_mlp_neurons`

### 4) RMT + Low-Rank Compression (Linear layers)
Applies **Random Matrix Theory (RMT)-style denoising** + **low-rank factorization** to selected Linear layers:
- Optional denoise step (column standardization + singular value shrinkage)
- Rank selection via:
  - `k_mode: "energy"` with `energy` threshold (e.g., 0.995 keeps 99.5% spectral energy)
  - Optional `k_min` to prevent ranks that are too small
- Layer filtering via `apply_to`:
  - `ffn_only` focuses on FFN layers (`intermediate.dense` and `output.dense`)
- Can restrict to specific encoder layers via `layers: [7,8,9,10,11]`

Implementation: `src/compression/rmt_lowrank.py::compress_model_rmt_lowrank`

---

## How the pipeline works (high level)

`main.py` runs the following steps (depending on your YAML config):

1. Load config (`configs/*.yaml`)
2. Load tokenizer + HF model
3. (Optional) Apply **RMT LowRank**
4. (Optional) Apply **structured pruning** (heads, then MLP)
5. (Optional) Apply **quantization** (PyTorch dynamic INT8 or ONNX dynamic INT8)
6. Build GLUE dataset + DataLoader
7. Run evaluation:
   - Collect predictions/labels
   - Compute metrics (Accuracy/F1)
   - Measure performance with warmup + timed forward passes

Timing parameters in `main.py`:
- `WARMUP_STEPS = 10`
- `MEASURE_STEPS = 50`

---

## Project Structure

```

├── configs/                # Experiment configs (per task)
├── src/
│   ├── compression/        # Quantization, pruning, low-rank methods
│   ├── datasets/           # GLUE dataset wrappers
│   ├── metrics/            # Accuracy/F1 + optional text metrics
│   └── config/             # Pydantic config schema
└── main.py                 # Main experimentation entrypoint

````

---

## Setup (Conda)

> Recommended: Python 3.11 (matches the existing README and is compatible with the pinned libs).

```bash
conda create -n llm_compressor python=3.11 -y
conda activate llm_compressor

pip install -r requirements.txt
````


---

## Running Experiments

### 1) Baseline evaluation

Pick a task config and run:

```bash
python main.py --config configs/config_bert_mrpc.yaml
python main.py --config configs/config_bert_sst2.yaml
python main.py --config configs/config_bert_qqp.yaml
```

### 2) Quantization experiments

Edit the YAML:

```yaml
quantization:
  enable: True
  method: "dynamic"   # or "onnx"
```

Run:

```bash
python main.py --config configs/config_bert_mrpc.yaml
```

### 3) Structured pruning experiments (heads / MLP)

Enable in YAML:

```yaml
prune:
  attention_heads:
    enable: True
    args:
      prune_amount: 0.25
      strategy: "l1"
      seed: 42
  mlp_neurons:
    enable: False
```

Or enable MLP pruning:

```yaml
prune:
  mlp_neurons:
    enable: True
    args:
      prune_amount: 0.15
      strategy: "l1"
      seed: 42
```

Run:

```bash
python main.py --config configs/config_bert_sst2.yaml
```

### 4) RMT LowRank experiments

Example (your current configs):

```yaml
rmt_lowrank:
  enable: true
  denoise: true
  shrink_method: "gavish_donoho"
  k_mode: "energy"
  energy: 0.995
  apply_to: "ffn_only"
  k_min: 384
  layers: [7, 8, 9, 10, 11]
```

Run:

```bash
python main.py --config configs/config_bert_qqp.yaml
```

---

## Configuration Parameters (brief)

### `model`

* `model_name`: Hugging Face repo id (e.g., `textattack/bert-base-uncased-MRPC`)

### `data`

* `batch_size`: batch size used in the DataLoader
* `max_input_length`: tokenizer max length (truncation)
* `task`: `mrpc`, `sst2`, `qqp`
* `split`: dataset split (`train`, `validation`, `test`)

### `device`

* `cuda`: if `true` and CUDA is available, runs on GPU (unless quantization forces CPU)

### `prune.attention_heads`

* `prune_amount`: fraction of heads removed per layer (e.g., `0.25`)
* `strategy`: `l1` (lowest L1 norm) or `random`
* `seed`: reproducibility for `random` and tie-breaking

### `prune.mlp_neurons`

* `prune_amount`: fraction of FFN intermediate neurons removed per layer
* `strategy`: `l1` or `random`
* `seed`: reproducibility

### `quantization`

* `enable`: enable quantization stage
* `method`:

  * `dynamic`: PyTorch dynamic INT8 for `nn.Linear` (CPU)
  * `onnx`: export FP32 ONNX + dynamic INT8 quantization with ONNX Runtime

### `rmt_lowrank`

* `enable`: enable RMT + low-rank compression
* `denoise`: whether to denoise weights before factorization
* `shrink_method`: singular value shrinkage method (`gavish_donoho`)
* `k_mode`: how rank `k` is chosen (`energy`, `ratio`, `fixed`)
* `energy`: spectral energy threshold used when `k_mode=energy`
* `apply_to`: which Linear layers to compress (`ffn_only` recommended for BERT FFN)
* `layers`: restrict compression to specific encoder layers
* `k_min`: minimum rank (prevents overly aggressive compression)

---

## Notes & Reproducibility Tips

* Quantization results are expected to differ between **CPU** and **GPU** runs (INT8 dynamic quantization is CPU-oriented).
* Structured pruning is architecture-changing, so it can produce real speedups, but overly aggressive pruning can collapse task metrics.
* Low-rank compression is most stable when applied selectively (e.g., `ffn_only`) and with conservative `energy` thresholds.

---


