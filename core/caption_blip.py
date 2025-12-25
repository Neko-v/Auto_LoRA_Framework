# /core/caption_blip.py

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm
import gc
import time

class BlipCaptioner:
    def __init__(self):
        # 自动判断设备(优先使用GPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在加载BLIP模型(使用设备: {self.device})...")

        # 加载模型(首次运行会自动下载1GB)
        # 使用Salesforce的基础模型, 效果好且速度快
        model_id = "Salesforce/blip-image-captioning-base"
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForVision2Seq.from_pretrained(model_id).to(self.device)
            print("BLIP模型加载成功")
        except Exception as e:
            print(f"模型加载失败, 请检查网络或HuggingFace连接, 错误信息: {e}")
    

    def generate_caption(self, image_path, trigger_word=""):
        """
        对单张图片生成描述
        :param image_path 图片路径
        :param trigger_word 触发词 (e.g. ohwx person), 会加在句首
        """
        try:
            raw_image = Image.open(image_path).convert('RGB')

            # 预处理并推理
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)

            # 生成描述(max_new_tokens = 50 控制描述长度)
            # 加上 torch.no_grad() 防止显存累积导致卡死
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=50)

            # 解码
            caption = self.processor.decode(out[0], skip_special_tokens=True)

            # 处理触发词逻辑
            if trigger_word:
                # 确保触发词后面有个逗号和空格
                final_caption = f"{trigger_word}, {caption}"
            else:
                final_caption = caption
            
            return final_caption
        except Exception as e:
            print(f"处理图片 {os.path.basename(image_path)} 时出错: {e}")
            return ""
    
    def process_folder(self, folder_path, trigger_word=""):
        """
        遍历文件夹, 为每张图片生成同名的.txt文件
        """
        # 支持的图片格式
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

        # 获取所有图片文件
        if not os.path.exists(folder_path):
            print(f"错误: 路径不存在 {folder_path}")
            return

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

        if not files:
            print(f"文件夹 {folder_path} 中没有找到图片")
            return
        
        print(f"开始处理 {len(files)} 张图片")

        # 开始前清理显存, 防止和上一步(裁剪)冲突
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 使用tqdm显示进度条
        for i, file in enumerate(tqdm(files, desc="BLIP打标中")):
            image_path = os.path.join(folder_path, file)
            caption = self.generate_caption(image_path, trigger_word)

            if caption:
                txt_path = os.path.splitext(image_path)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
            
            # 每处理 5 张, 手动释放显存, 防止小显存卡死
            if i > 0 and i % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                time.sleep(0.02)  # 防止卡住后台

# 测试代码块
if __name__ == "__main__":
    # 这是测试入口, 验证代码是否可用

    # 实例化
    captioner = BlipCaptioner()

    # 设置data文件夹路径, 确保data文件夹中至少有一张图片
    # 注意: 这里为了兼容性, 使用相对路径获取测试目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    test_data_path = os.path.join(project_root, "data", "Hu_Ge")

    # 运行测试(假设触发词是 sivi person)
    if os.path.exists(test_data_path):
        captioner.process_folder(test_data_path, trigger_word="sivi person")
        print("测试完成, 请去data文件夹看看有没有生成.txt文件")
    else:
        print(f"测试跳过: 找不到路径 {test_data_path}")