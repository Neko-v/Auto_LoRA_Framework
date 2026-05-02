# /core/train_profiles.py

DEFAULT_PROFILE_NAME = "SD1.5 人物 LoRA / 8GB 稳定"


TRAIN_PROFILES = {
    "SD1.5 人物 LoRA / 8GB 稳定": {
        "mode_selection": "人物 (Person)",
        "repeats": 10,
        "epochs": 10,
        "network_dim": 32,
        "network_alpha": 16,
        "learning_rate": "0.0001",
        "text_encoder_lr": "0.00005",
        "unet_lr": "0.0001",
        "batch_size": 1,
        "save_every_n_epochs": 1,
        "seed": 1024,
        "description": "适合真人 角色 人像 LoRA, 面向 8GB 显存, 稳定优先",
    },
    "SD1.5 画风 LoRA / 8GB 稳定": {
        "mode_selection": "画风 (Style)",
        "repeats": 20,
        "epochs": 10,
        "network_dim": 64,
        "network_alpha": 32,
        "learning_rate": "0.0001",
        "text_encoder_lr": "0.00005",
        "unet_lr": "0.0001",
        "batch_size": 1,
        "save_every_n_epochs": 1,
        "seed": 1024,
        "description": "适合油画 插画 画风类LoRA, Rank 稍高, 增强风格容量",
    },
    "SD1.5 低显存保守 / 6-8GB": {
        "mode_selection": "人物 (Person)",
        "repeats": 10,
        "epochs": 8,
        "network_dim": 16,
        "network_alpha": 8,
        "learning_rate": "0.00008",
        "text_encoder_lr": "0.00004",
        "unet_lr": "8e-5",
        "batch_size": 1,
        "save_every_n_epochs": 1,
        "seed": 1024,
        "description": "更保守的低显存配置, 适合显存紧张或希望降低过拟合风险的场景",
    },
}


def get_profile_names():
    """返回所有 Profile 名称"""
    return list(TRAIN_PROFILES.keys())


def get_profile_config(profile_name):
    """根据名称获取 Profile 配置"""
    if profile_name not in TRAIN_PROFILES:
        profile_name = DEFAULT_PROFILE_NAME

    return dict(TRAIN_PROFILES[profile_name])