# /core/train_manager.py

import os
import subprocess
import sys

class TrainManager:
    def __init__(self, base_dir, sd_scripts_dir):
        """
        :param base_dir: 项目根目录
        :param sd_scripts_dir: kohya sd-scripts 的路径
        """
        self.base_dir = base_dir
        self.sd_scripts_dir = sd_scripts_dir
        
        # 训练脚本路径 (Kohya的核心脚本)
        self.train_script = os.path.join(sd_scripts_dir, "train_network.py")
        
        # 检查脚本是否存在
        if not os.path.exists(self.train_script):
            raise FileNotFoundError(f"找不到 train_network.py, 请检查 sd-scripts 路径: {self.train_script}")

    def run_training(self, 
                     base_model_path, 
                     train_data_dir, 
                     output_dir, 
                     output_name, 
                     resolution=512,
                     max_train_epochs=10):
        """
        生成命令并执行训练
        """
        print(f"开始构建训练命令...")
        print(f"底模: {os.path.basename(base_model_path)}")
        print(f"数据: {train_data_dir}")

        # 核心参数配置
        
        # 基础命令 (使用 accelerate 启动)
        cmd = [
            "accelerate", "launch",
            "--num_cpu_threads_per_process=2",
            self.train_script,
            
            # 模型路径
            f"--pretrained_model_name_or_path={base_model_path}",
            f"--train_data_dir={train_data_dir}",
            f"--output_dir={output_dir}",
            f"--output_name={output_name}",
            
            # 训练参数 (SD1.5 标准)
            f"--resolution={resolution},{resolution}",
            f"--max_train_epochs={max_train_epochs}",
            "--save_model_as=safetensors",
            "--prior_loss_weight=1.0",
            
            # 显存优化 (3060Ti 8G 关键配置)
            "--enable_bucket",  # 开启桶机制(处理不同长宽比)
            "--gradient_checkpointing",  # *以时间换显存
            "--mixed_precision=fp16",  # *半精度训练
            "--sdpa",  # *加速注意力机制
            "--cache_latents",  # *缓存潜变量, 大幅提升速度
            
            # 学习率与优化器
            "--learning_rate=1e-4",
            "--text_encoder_lr=5e-5",  # 文本编码器学习率
            "--unet_lr=1e-4",  # UNet学习率
            "--optimizer_type=AdamW8bit",  # *8bit优化器, 极其省显存
            
            # LoRA 结构参数
            "--network_module=networks.lora",
            "--network_dim=32",  # Rank (32-128都可以, 32够用了)
            "--network_alpha=16",  # Alpha (通常是Dim的一半)
            
            # 批次设置
            "--train_batch_size=1",  # 8G显存建议设为1, 最稳
            "--save_every_n_epochs=1",  # 每个epoch保存一次
            "--seed=1024"  # 固定种子
        ]

        # 打印最终命令(方便调试)
        print("\n 即将执行的命令 (你可以复制它手动运行):")
        print(" ".join(cmd))
        print("-" * 50)

        # 执行命令
        try:
            # shell=True 在 Windows 下有时需要, 但也可能导致路径空格问题
            # 这里直接传列表给 subprocess, 系统会自动处理转义, 更安全
            subprocess.run(cmd, check=True)
            print("\n 训练完成！")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n 训练出错, 错误代码: {e.returncode}")
            return False
        except FileNotFoundError:
            print("\n ERROR: 无法执行 'accelerate' 命令 请确保你已激活虚拟环境, 并且安装了 requirements.txt ")
            return False