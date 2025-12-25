# /core/caption_wd14.py

import os
import csv
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import pandas as pd
from tqdm import tqdm

class WD14Tagger:
    def __init__(self, model_repo="SmilingWolf/wd-v1-4-convnext-tagger-v2", threshold=0.35):
        """
        初始化 WD14 Tagger
        :param model_repo: HuggingFace 上的模型仓库 ID
        :param threshold: 置信度阈值, 大于这个值的标签才会被保留
        """
        self.repo_id = model_repo
        self.threshold = threshold
        self.model_path = None
        self.csv_path = None
        self.ort_session = None
        self.labels = []
        
        # 自动初始化
        self._load_model()

    def _load_model(self):
        print("[WD14] 正在检查/下载模型文件...")
        
        # 定义模型缓存路径
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "wd14")
        os.makedirs(cache_dir, exist_ok=True)

        # 下载 model.onnx
        try:
            self.model_path = hf_hub_download(repo_id=self.repo_id, filename="model.onnx", cache_dir=cache_dir)
            self.csv_path = hf_hub_download(repo_id=self.repo_id, filename="selected_tags.csv", cache_dir=cache_dir)
        except Exception as e:
            print(f"[WD14] 模型下载失败，请检查网络连接: {e}")
            return

        print("[WD14] 正在加载 ONNX Runtime...")
        # 优先使用 GPU (CUDAExecutionProvider), 如果没有则使用 CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception as e:
             print(f"[WD14] 加载推理引擎失败: {e}")
             return

        # 加载标签列表
        try:
            df = pd.read_csv(self.csv_path)
            self.labels = df['name'].tolist()
            print("[WD14] Tagger 初始化完成")
        except Exception as e:
            print(f"[WD14] 读取标签文件失败: {e}")

    def _preprocess(self, image: Image.Image):
        """
        预处理图片: 缩放 -> 补白 -> 归一化 -> 转 NCHW
        """
        # WD14 模型输入通常是 448x448
        size = 448
        
        # 保持比例缩放
        image.thumbnail((size, size), Image.Resampling.BICUBIC)
        
        # 创建正方形白底图
        new_img = Image.new("RGB", (size, size), (255, 255, 255))
        new_img.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        
        # 转 numpy
        img_np = np.array(new_img).astype(np.float32)
        
        # BGR -> RGB (PIL 默认是 RGB, cv2 是 BGR, 这里 WD14 训练时是 BGR)
        # SmilingWolf 的模型通常需要 BGR 输入, 所以要做 RGB -> BGR 转换
        img_np = img_np[:, :, ::-1] 
        
        # 归一化
        img_np = np.expand_dims(img_np, 0) # 增加 batch 维度
        return img_np

    def tag_image(self, image_path):
        """
        对单张图片进行打标
        :return: tag_string (逗号分隔的字符串)
        """
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self._preprocess(image)
            
            # 推理
            input_name = self.ort_session.get_inputs()[0].name
            probs = self.ort_session.run(None, {input_name: input_tensor})[0]
            
            # probs[0] 是一个概率数组，对应 self.labels
            probs = probs[0]
            
            # 筛选标签
            tags = []
            for i, p in enumerate(probs):
                if p > self.threshold:
                    tag_name = self.labels[i]
                    
                    # 黑名单过滤 (Minimalist)
                    # 只过滤 Rating (元数据), 不过滤风格描述
                    blacklist = ["rating:safe", "rating:questionable", "rating:explicit"]
                    
                    if tag_name not in blacklist:
                        tags.append(tag_name.replace("_", " "))
            
            # 组合成字符串
            return ", ".join(tags)

        except Exception as e:
            print(f"打标失败 {image_path}: {e}")
            return ""

    def process_folder(self, folder_path, trigger_word=None):
        """
        批量处理文件夹中的图片, 生成同名的 .txt 文件
        """
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        import glob
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            
        print(f"[WD14] 开始打标 {len(image_files)} 张图片...")
        
        for img_path in tqdm(image_files):
            tags = self.tag_image(img_path)
            
            # 插入触发词逻辑
            if trigger_word and tags:
                final_tags = f"{trigger_word}, {tags}"
            elif trigger_word:
                final_tags = trigger_word
            else:
                final_tags = tags

            if final_tags:
                txt_path = os.path.splitext(img_path)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(final_tags)

# 测试代码
if __name__ == "__main__":
    tagger = WD14Tagger()
    
    # 测试
    test_dir = r"V:\Auto_LoRA\LoRA-AutoTrainer\dataset\Oil_Painting"
    if os.path.exists(test_dir):
        tagger.process_folder(test_dir, trigger_word="sivi")
        print("打标完成，请检查 txt 文件")