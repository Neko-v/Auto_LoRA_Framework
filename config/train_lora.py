import subprocess

def train_lora():
    config = "training/config.toml"
    cmd = [
        "accelerate", "launch",
        "sd-scripts/train_network.py",
        f"--config_file={config}"
    ]

    subprocess.run(cmd)


if __name__ == "__main__":
    train_lora()