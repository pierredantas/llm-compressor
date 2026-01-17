from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Union


class ModelConfig(BaseModel):
    model_name: str = "t5-small"  # "t5-base", "t5-large"
    should_train: bool = False


class DataConfig(BaseModel):
    batch_size: int
    shuffle: bool
    max_input_length: int
    task: Literal["sst2", "mrpc", "qqp"] = "sst2"
    split: Literal["train", "validation", "test"] = "validation"

class TrainerConfig(BaseModel):
    num_epochs: int = 3
    learning_rate: float = 5e-5
    save_model_path: str = "./model"


class PruneOptions(BaseModel):
    enable: bool = False
    args: Optional[dict] = {}


class QuantizationOptions(BaseModel):
    enable: bool = False
    method: str = "none"  # Opções: "dynamic", "onnx", "trt", "none"


class PruneConfig(BaseModel):
    attention_heads: Optional[PruneOptions] = None
    mlp_neurons: Optional[PruneOptions] = None
    tokens: Optional[PruneOptions] = None
    magnitude: Optional[PruneOptions] = None
    gradient: Optional[PruneOptions] = None
    stochastic: Optional[PruneOptions] = None
    fourrier: Optional[PruneOptions] = None


class RmtLowRankConfig(BaseModel):
    enable: bool = False
    denoise: bool = True
    shrink_method: Literal[
        "gavish_donoho",
        "bulk_shrink",
        "hard_threshold",
        "soft_threshold",
    ] = "gavish_donoho"

    k_mode: Literal["energy", "ratio", "fixed"] = "energy"
    energy: float = 0.995
    ratio: Optional[float] = None
    k: Optional[int] = None

    apply_to: Literal[
        "all_linear",
        "bert_linear_only",
        "exclude_classifier",
        "ffn_only",
        "ffn_plus_attn_output",
        "qkv_only",
    ] = "exclude_classifier"

    layers: Optional[list[int]] = None
    k_min: Optional[int] = None
    k_max: Optional[int] = None

class LoggingConfig(BaseModel):
    log_frequency: int = 100
    print_summary: bool = True


class DeviceConfig(BaseModel):
    cuda: bool = True


class Config(BaseModel):
    model: ModelConfig
    data: DataConfig
    trainer: TrainerConfig
    logging: LoggingConfig
    device: DeviceConfig
    prune: Optional[PruneConfig] = None
    quantization: Optional[QuantizationOptions] = None
    rmt_lowrank: Optional[RmtLowRankConfig] = None
