# /core/train_manager.py

import os
import subprocess
import sys

from core.process_runner import stream_subprocess
from core.train_config import TrainConfig


class TrainManager:
    def __init__(self, base_dir, sd_scripts_dir):
        """
        :param base_dir: 项目根目录
        :param sd_scripts_dir: kohya sd-scripts 的路径
        """
        self.base_dir = base_dir
        self.sd_scripts_dir = sd_scripts_dir

        self.train_script = os.path.join(sd_scripts_dir, "train_network.py")

        if not os.path.exists(self.train_script):
            raise FileNotFoundError(
                f"找不到 train_network.py, 请检查 sd-scripts 路径: {self.train_script}"
            )

    def build_command(
        self,
        base_model_path,
        train_data_dir,
        output_dir,
        output_name,
        config=None,
        resolution=512,
        max_train_epochs=10,
    ):
        """
        构建 Kohya 训练命令

        为了兼容旧调用方式: 
        - 如果传入 config, 则优先使用 config
        - 如果没有传入 config, 则用 resolution / max_train_epochs 创建默认配置
        """
        if config is None:
            config = TrainConfig(
                resolution=resolution,
                max_train_epochs=max_train_epochs,
            )

        cmd = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_cpu_threads_per_process=2",
            self.train_script,

            # 模型路径
            f"--pretrained_model_name_or_path={base_model_path}",
            f"--train_data_dir={train_data_dir}",
            f"--output_dir={output_dir}",
            f"--output_name={output_name}",

            # Caption 文件设置
            f"--caption_extension={config.caption_extension}",

            # 训练参数
            f"--resolution={config.resolution},{config.resolution}",
            f"--max_train_epochs={config.max_train_epochs}",
            f"--save_model_as={config.save_model_as}",
            f"--prior_loss_weight={config.prior_loss_weight}",

            # 学习率与优化器
            f"--learning_rate={config.learning_rate}",
            f"--text_encoder_lr={config.text_encoder_lr}",
            f"--unet_lr={config.unet_lr}",
            f"--optimizer_type={config.optimizer_type}",

            # LoRA 结构参数
            f"--network_module={config.network_module}",
            f"--network_dim={config.network_dim}",
            f"--network_alpha={config.network_alpha}",

            # 批次设置
            f"--train_batch_size={config.train_batch_size}",
            f"--save_every_n_epochs={config.save_every_n_epochs}",
            f"--seed={config.seed}",
        ]

        # 布尔开关参数
        if config.no_metadata:
            cmd.append("--no_metadata")

        if config.enable_bucket:
            cmd.append("--enable_bucket")

        if config.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")

        if config.mixed_precision:
            cmd.append(f"--mixed_precision={config.mixed_precision}")

        if config.sdpa:
            cmd.append("--sdpa")

        if config.cache_latents:
            cmd.append("--cache_latents")

        return cmd

    def build_env(self):
        """构建训练子进程环境变量"""
        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = (
            self.sd_scripts_dir
            + os.pathsep
            + run_env.get("PYTHONPATH", "")
        )
        return run_env

    def run_training_stream(
        self,
        base_model_path,
        train_data_dir,
        output_dir,
        output_name,
        config=None,
        resolution=512,
        max_train_epochs=10,
    ):
        """流式执行训练, 实时产出日志"""
        if config is None:
            config = TrainConfig(
                resolution=resolution,
                max_train_epochs=max_train_epochs,
            )

        yield "开始构建训练命令..."
        yield f"底模: {os.path.basename(base_model_path)}"
        yield f"数据: {train_data_dir}"
        yield f"训练配置: resolution={config.resolution}, epochs={config.max_train_epochs}, rank={config.network_dim}, alpha={config.network_alpha}"

        cmd = self.build_command(
            base_model_path=base_model_path,
            train_data_dir=train_data_dir,
            output_dir=output_dir,
            output_name=output_name,
            config=config,
        )

        yield ""
        yield "即将执行的命令: "
        yield " ".join(cmd)
        yield "-" * 50

        try:
            for line in stream_subprocess(
                cmd,
                cwd=self.sd_scripts_dir,
                env=self.build_env(),
            ):
                yield line

            yield ""
            yield "训练完成！"

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"训练出错, 错误代码: {e.returncode}") from e

        except FileNotFoundError as e:
            raise RuntimeError(
                "无法执行训练命令, 请确认 accelerate 已安装, 并且当前虚拟环境已激活"
            ) from e

    def run_training(
        self,
        base_model_path,
        train_data_dir,
        output_dir,
        output_name,
        config=None,
        resolution=512,
        max_train_epochs=10,
    ):
        """兼容 main.py 的阻塞版本"""
        try:
            for line in self.run_training_stream(
                base_model_path=base_model_path,
                train_data_dir=train_data_dir,
                output_dir=output_dir,
                output_name=output_name,
                config=config,
                resolution=resolution,
                max_train_epochs=max_train_epochs,
            ):
                print(line)

            return True

        except Exception as e:
            print(f"\n训练失败: {e}")
            return False