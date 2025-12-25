# /train.py

import os
import subprocess
import sys


def main():
    # 路径配置, 自动获取当前绝对路径
    root_dir = os.getcwd()

    # 核心训练脚本路径(Kohya)
    script_path = os.path.join(root_dir, "sd-scripts", "train_network.py")

    # 配置文件路径
    config_file = os.path.join(root_dir, "config", "config.toml")

    # 数据集路径, 指向dataset/image即可, 会自动扫描40_sivi文件夹
    train_data_dir = os.path.join(root_dir, "dataset", "image")

    # 日志路径
    logging_dir = os.path.join(root_dir, "dataset", "log")

    # 检查关键文件是否存在
    if not os.path.exists(script_path):
        print(f"ERROR 找不到训练脚本: {script_path}")
        print(f"请检查 sd-scripts 文件夹是否完整")
        return

    if not os.path.exists(config_file):
        print(f"ERROR 找不到配置文件: {config_file}")
        return
    
    # 设置环境变量, 让Python能找到sd-scripts里的库
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.path.join(root_dir, 'sd-scripts')}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # 构建启动命令, 使用accelerate launch启动以获得最佳性能
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process=2",  # 稍微限制CPU占用
        script_path,  # 调用train_network.py
        "--config_file", config_file,  # 传入toml配置
        "--train_data_dir", train_data_dir,  # 传入数据位置
        "--logging_dir", logging_dir  # 传入日志位置
    ]

    print("正在启动训练...")
    print(f"数据集: {train_data_dir}")
    print(f"配置: {config_file}")
    print("=" * 50)

    try:
        # 开始执行
        subprocess.run(cmd, env=env, check=True)
        print("\n训练完成")
        print(f"请检查output文件夹查看LoRA模型: {os.path.join(root_dir, 'output')}")
    except subprocess.CalledProcessError as e:
        print(f"训练过程中发生错误: {e}")
    except KeyboardInterrupt:
        print("\n已手动停止训练")


if __name__ == "__main__":
    main()