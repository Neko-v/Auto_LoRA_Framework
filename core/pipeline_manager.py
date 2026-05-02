# /core/pipeline_manager.py

import os
import sys
import shutil
import logging
import warnings
import subprocess

from core.process_runner import stream_subprocess

# 环境配置与日志屏蔽
# 屏蔽 HuggingFace Transformers 的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# 屏蔽 MediaPipe/TensorFlow 的底层 C++ 日志 (0 = all, 1 = filter info, 2 = filter warning, 3 = filter error)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 屏蔽 PyTorch Distributed 在 Windows 下的 Redirect 警告
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
# 屏蔽 absl (MediaPipe依赖库) 的日志
logging.getLogger('absl').setLevel(logging.ERROR)

class AutoPipeline:
    def __init__(self, base_dir, source_dir, resolution=512, caption_method="blip", dataset_dir=None):
        """
        自动化处理管线
        :param base_dir: 项目根目录
        :param source_dir: 具体存放图片的文件夹路径
        :param resolution: 目标图片分辨率
        :param caption_method: blip 或 wd14
        :param dataset_dir: 本次任务的数据集根目录。若为空，则使用全局 dataset。
        """
        self.base_dir = base_dir
        self.data_dir = source_dir
        self.dataset_dir = dataset_dir or os.path.join(base_dir, "dataset")
        self.caption_method = caption_method
        self.resolution = resolution

        self.img_dir_root = os.path.join(self.dataset_dir, "image")

        print(f"初始化处理管线 (分辨率: {resolution}x{resolution})...")
        print(f"数据源锁定: {self.data_dir}")
        print(f"数据集根目录: {self.dataset_dir}")
        print(f"打标引擎: {self.caption_method.upper()}")

        print("工具加载完毕")

    def setup_directories(self, instance_name, class_name, repeats):
        """Step 1: 建目录"""
        # 构建Kohya文件夹名: "次数_触发词 类别"
        folder_name = f"{repeats}_{instance_name} {class_name}"
        target_instance_dir = os.path.join(self.img_dir_root, folder_name)

        # 目录清理与重建 (安全模式)
        if os.path.exists(target_instance_dir):
            print(f"清理旧数据: {target_instance_dir}")
            shutil.rmtree(target_instance_dir)
        
        os.makedirs(target_instance_dir, exist_ok=True)
        
        print(f"创建训练目录: {target_instance_dir}")
        return target_instance_dir
    
    def run_crop_stream(self, target_dir, mode):
        """Step 2: 裁剪，流式输出日志"""
        yield "-" * 30
        yield f"阶段 1/2: 智能裁剪与缩放 (模式: {mode})"

        if not os.path.exists(self.data_dir):
            raise RuntimeError(f"数据源目录不存在: {self.data_dir}")

        if not os.listdir(self.data_dir):
            raise RuntimeError(f"数据源目录为空: {self.data_dir}")

        this_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(this_dir, "run_crop_standalone.py")

        if not os.path.exists(script_path):
            raise RuntimeError(f"找不到裁剪子进程脚本: {script_path}")

        yield "正在启动独立子进程运行 SmartCrop 裁剪任务..."

        cmd = [
            sys.executable,
            script_path,
            "--input", self.data_dir,
            "--output", target_dir,
            "--mode", mode,
            "--resolution", str(self.resolution),
        ]

        try:
            for line in stream_subprocess(cmd):
                yield line

            yield "裁剪任务完成"

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"裁剪任务失败，返回码: {e.returncode}") from e

        except Exception as e:
            raise RuntimeError(f"启动裁剪子进程失败: {e}") from e


    def run_crop(self, target_dir, mode):
        """兼容 CLI/main.py 的普通阻塞版本"""
        try:
            for line in self.run_crop_stream(target_dir, mode):
                print(line)
            return True
        except Exception as e:
            print(f"裁剪失败: {e}")
            return False

    def run_caption_stream(self, target_dir, instance_name, class_name):
        """Step 3: 打标，流式输出日志"""
        trigger_word = f"{instance_name} {class_name}"
        method = self.caption_method.lower()

        yield "-" * 30
        yield f"阶段 2/2: {method.upper()} 自动打标 (注入触发词: '{trigger_word}')"

        this_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(this_dir, "run_caption_standalone.py")

        if not os.path.exists(script_path):
            raise RuntimeError(f"找不到打标子进程脚本: {script_path}")

        yield f"正在启动独立子进程运行 {method.upper()} 打标任务..."

        cmd = [
            sys.executable,
            script_path,
            "--folder", target_dir,
            "--trigger", trigger_word,
            "--method", method,
        ]

        try:
            for line in stream_subprocess(cmd):
                yield line

            yield "打标任务完成"

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"打标任务失败，返回码: {e.returncode}") from e

        except Exception as e:
            raise RuntimeError(f"启动打标子进程失败: {e}") from e


    def run_caption(self, target_dir, instance_name, class_name):
        """兼容 CLI/main.py 的普通阻塞版本"""
        try:
            for line in self.run_caption_stream(target_dir, instance_name, class_name):
                print(line)
            return True
        except Exception as e:
            print(f"打标失败: {e}")
            return False

    def prepare_dataset(self, instance_name, class_name, repeats=40, mode="person"):
        """
        执行完整的数据准备流程
        """
        target_dir = self.setup_directories(instance_name, class_name, repeats)

        if not self.run_crop(target_dir, mode):
            return None

        if not self.run_caption(target_dir, instance_name, class_name):
            return None

        return target_dir

if __name__ == "__main__":
    # 测试代码
    
    # 获取当前脚本所在目录的上一级作为项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(current_dir)  # 回退到根目录
    
    # 模拟数据源路径
    TEST_SOURCE = os.path.join(PROJECT_ROOT, "data", "Hu_Ge")

    # 配置
    MY_NAME = "test_user"
    MY_CLASS = "person"
    
    # 实例化并运行 (测试 512 分辨率)
    if os.path.exists(TEST_SOURCE):
        pipeline = AutoPipeline(PROJECT_ROOT, source_dir=TEST_SOURCE, resolution=512)
        
        # 运行
        pipeline.prepare_dataset(instance_name=MY_NAME, class_name=MY_CLASS, repeats=20)
    else:
        print("测试跳过: 找不到测试数据源")