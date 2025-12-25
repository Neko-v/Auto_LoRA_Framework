# /main.py

import os
import sys
import time

# 将当前目录加入模块搜索路径, 确保能找到 core 包
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from core.pipeline_manager import AutoPipeline
from core.train_manager import TrainManager

# ================= 配置区域 =================

# 模式设置
# "person" (人像) | "style" (风格)
TRAIN_MODE = "style" 
# 打标"blip" (自然语言,适合写实/照片) | "wd14" (Tag词,适合二次元/画风/插画)
CAPTION_METHOD = "wd14"

# 角色/画风设定
# 如果是画风，Instance Name 最好独特一点
INSTANCE_NAME = "sivi_art"  # 触发词
CLASS_NAME    = "style"  # 类别 (person, style, landscape)

# 路径设置
# 图片文件夹名字
SOURCE_FOLDER_NAME = "Oil_Painting_1024"
DATA_SOURCE = os.path.join(current_dir, "data", SOURCE_FOLDER_NAME)

# 模型选择
# 真人/写实风 -> chilloutmix_NiPrunedFp32Fix.safetensors
# 官方底模 -> stable-diffusion-v1-5.safetensors
BASE_MODEL_NAME = "stable-diffusion-v1-5.safetensors"
BASE_MODEL_PATH = os.path.join(current_dir, "models", BASE_MODEL_NAME)

# 4. 训练参数
TRAIN_REPEATS = 40  # 图片重复次数 (图片多于20张设20, 少于20张设40)
MAX_EPOCHS    = 10  # 训练总轮数 (一共跑几遍)
RESOLUTION    = 512  # SD1.5 标准分辨率

# ==========================================================

def main():
    print(f"🚀 Auto LoRA Trainer 启动 (模式: {TRAIN_MODE} | 打标: {CAPTION_METHOD})")
    
    # 0. 环境检查
    sd_scripts_path = os.path.join(current_dir, "sd-scripts")
    if not os.path.exists(sd_scripts_path):
        print("ERROR: 根目录下找不到 sd-scripts 文件夹")
        return

    if not os.path.exists(BASE_MODEL_PATH):
        print(f"ERROR: 找不到底模文件: {BASE_MODEL_PATH}")
        print("请检查 models 文件夹下的文件名是否正确")
        return
        
    if not os.path.exists(DATA_SOURCE) or not os.listdir(DATA_SOURCE):
        print(f"ERROR: 数据源为空: {DATA_SOURCE}")
        print(f"请确保文件夹里有照片")
        return

    # 1. 数据处理阶段
    print("\n[Step 1] 准备数据...")
    pipeline = AutoPipeline(
        base_dir=current_dir,
        source_dir=DATA_SOURCE,
        resolution=RESOLUTION,
        caption_method=CAPTION_METHOD
    )
    
    dataset_img_dir = pipeline.prepare_dataset(
        instance_name=INSTANCE_NAME,
        class_name=CLASS_NAME,
        repeats=TRAIN_REPEATS,
        mode=TRAIN_MODE
    )

    if not dataset_img_dir:
        print("数据准备失败, 程序终止")
        return

    # dataset_img_dir 类似于: .../dataset/image/40_hu_ge man
    # Kohya 需要的 train_data_dir 是它的上一级, 即 .../dataset/image
    # 这样 Kohya 才能读取到 "40_hu_ge man" 这个文件夹名里的次数信息
    train_data_root = os.path.dirname(dataset_img_dir)
    
    output_dir = os.path.join(current_dir, "output", f"{INSTANCE_NAME}_lora")
    os.makedirs(output_dir, exist_ok=True)

    # 2. 训练阶段
    print("\n[Step 2] 开始训练...")
    trainer = TrainManager(
        base_dir=current_dir,
        sd_scripts_dir=sd_scripts_path
    )
    
    success = trainer.run_training(
        base_model_path=BASE_MODEL_PATH,
        train_data_dir=train_data_root,  # 指向 image 文件夹
        output_dir=output_dir,
        output_name=INSTANCE_NAME,
        resolution=RESOLUTION,
        max_train_epochs=MAX_EPOCHS
    )

    if success:
        print("\n" + "="*40)
        print(f"恭喜 LoRA 训练完成 ")
        print(f"模型保存在: {output_dir}")
        print("="*40)
    else:
        print("\n训练过程中出现错误, 请检查上方的报错信息")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户手动停止")
    except Exception as e:
        print(f"发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n")
        input("按回车键关闭窗口...")