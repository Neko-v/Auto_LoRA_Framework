# /core/train_config.py

from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    """
    Auto_LoRA 的训练参数配置
    这里只放 Kohya 训练相关参数, 不放路径参数
    """

    # 基础训练参数
    resolution: int = 512
    max_train_epochs: int = 10
    train_batch_size: int = 1
    save_every_n_epochs: int = 1
    seed: int = 1024

    # Caption 设置
    caption_extension: str = ".txt"

    # LoRA 结构
    network_module: str = "networks.lora"
    network_dim: int = 32
    network_alpha: int = 16

    # 学习率
    learning_rate: str = "1e-4"
    text_encoder_lr: str = "5e-5"
    unet_lr: str = "1e-4"

    # 优化器
    optimizer_type: str = "AdamW8bit"

    # 模型保存
    save_model_as: str = "safetensors"
    prior_loss_weight: float = 1.0
    no_metadata: bool = True

    # 显存 / 性能优化
    enable_bucket: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"
    sdpa: bool = True
    cache_latents: bool = True

    def to_dict(self):
        """用于保存到 config_snapshot.json"""
        return asdict(self)