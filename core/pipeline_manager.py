# /core/pipeline_manager.py

import os
import sys
import shutil
import logging
import warnings
import gc
import torch
import subprocess  # 子进程
from core.smart_crop import SmartCropper
# from core.caption_blip import BlipCaptioner
# from core.caption_wd14 import WD14Tagger

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
    def __init__(self, base_dir, source_dir, resolution=512, caption_method="blip"):
        """
        自动化处理管线
        :param base_dir: 项目根目录
        :param source_dir: 具体存放图片的文件夹路径
        :param resolution: 目标图片分辨率 (SD1.5用512, SDXL用1024)
        """
        self.base_dir = base_dir
        self.data_dir = source_dir # 原始素材来源
        self.dataset_dir = os.path.join(base_dir, "dataset")  # 训练集输出位置
        self.caption_method = caption_method
        self.resolution = resolution

        # 预先计算Kohya所需的子目录结构
        self.img_dir_root = os.path.join(self.dataset_dir, "image")
        self.log_dir = os.path.join(self.dataset_dir, "log")
        self.model_dir = os.path.join(self.dataset_dir, "model")

        print(f"初始化处理管线 (分辨率: {resolution}x{resolution})...")
        print(f"数据源锁定: {self.data_dir}")
        print(f"打标引擎: {self.caption_method.upper()}")
        
        # # 实例化工具 (加载模型)
        # # 如果内存非常紧张, 在 process 时再实例化, 用完即销毁
        # self.cropper = SmartCropper(target_size=resolution)
        
        # # 根据选择实例化打标器
        # if self.caption_method == "wd14":
        #     self.captioner = WD14Tagger()
        # else:
        #     # 默认 fallback 到 BLIP
        #     self.captioner = BlipCaptioner()

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
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        print(f"创建训练目录: {target_instance_dir}")
        return target_instance_dir
    
    def run_crop(self, target_dir, mode):
        """Step 2: 裁剪"""
        print("-" * 30)
        print(f"阶段 1/2: 智能裁剪与缩放 (模式: {mode})")
        # 检查源目录是否有图片
        if not os.path.exists(self.data_dir) or not os.listdir(self.data_dir):
            print(f"错误: 数据源目录 {self.data_dir} 不存在或为空")
            return False

        # 加载 Cropper
        print("正在加载 SmartCropper...")
        cropper = SmartCropper(target_size=self.resolution)
        
        cropper.process_folder(self.data_dir, target_dir, mode=mode)

        # 销毁模型并清理内存
        print("释放内存...")
        del cropper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return True

    def run_caption(self, target_dir, instance_name, class_name):
        """Step 3: 打标"""
        # 构造触发词
        trigger_word = f"{instance_name} {class_name}"
        method = self.caption_method.lower()
        
        print("-" * 30)
        print(f"阶段 2/2: {method.upper()} 自动打标 (注入触发词: '{trigger_word}')")
        
        # 确保显存干净
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 构造通用子进程命令
        # 相当于在命令行运行: python core/run_caption_standalone.py --folder "..." --trigger "..." --method blip
        this_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(this_dir, "run_caption_standalone.py")
        print(f"正在启动独立子进程运行 {method.upper()} 打标任务...")
        
        cmd = [
            sys.executable,
            script_path,
            "--folder", target_dir,
            "--trigger", trigger_word,
            "--method", method  # 告诉脚本用哪个模型打标
        ]

        try:
            # check=True 会在子进程报错时抛出异常
            subprocess.run(cmd, check=True)
            print(f"打标任务完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"运行失败 (返回码 {e.returncode})")
            return False
        finally:
            # 无论成功失败，最后再清一次显存，保持主进程干净
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # # 加载 Captioner
        # print(f"正在加载 {self.caption_method.upper()} 模型...")
        # if self.caption_method == "wd14":
        #     captioner = WD14Tagger()
        # else:
        #     captioner = BlipCaptioner()

        # captioner.process_folder(target_dir, trigger_word=trigger_word)
        
        # # 清理内存
        # print("释放显内存...")
        # del captioner
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        # gc.collect()

        # return True

    def prepare_dataset(self, instance_name, class_name, repeats=40, mode="person"):
        """
        执行完整的数据准备流程 (这是对上面三个步骤的封装)
        :param instance_name: 实例名字 (触发词, 如 'sivi')
        :param class_name: 类别名称 (如 'person', 'girl', 'boy')
        :param repeats: 每张图片的重复训练次数 (Kohya 文件夹命名规则)
        :param mode: 'person' (人脸裁剪) 或 'style' (画风缩放)
        :return: 最终生成的图片路径
        """
        # 1. 建立目录
        target_dir = self.setup_directories(instance_name, class_name, repeats)
        
        # 2. 裁剪
        if not self.run_crop(target_dir, mode):
            return None
            
        # 3. 打标
        self.run_caption(target_dir, instance_name, class_name)

        # print("-" * 30)
        # print("数据集准备完毕!")
        # print(f"训练图片路径: {target_dir}")
        # print(f"日志路径: {self.log_dir}")
        # print(f"模型输出: {self.model_dir}")
        
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