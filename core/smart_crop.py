# /core/smart_crop.py

import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from tqdm import tqdm

class SmartCropper:
    def __init__(self, target_size=512):
        self.target_size = target_size

        # 初始化 MediaPipe 人脸检测
        self.mp_face_detection = mp.solutions.face_detection
        
        # 这就是你之前代码里的设置, 对于很多高清图, 这个模式反而比近距离模式更准
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5
        )
        print(f"SmartCropper 已初始化, 目标尺寸: {target_size} x {target_size}")

    def _get_center_crop_box(self, h, w):
        """
        [内部函数] 计算中心裁剪的正方形坐标
        """
        crop_size = min(h, w)
        cx, cy = w // 2, h // 2

        half_size = crop_size // 2
        x1 = cx - half_size
        y1 = cy - half_size
        x2 = cx + half_size
        y2 = cy + half_size

        return int(x1), int(y1), int(x2), int(y2)

    def _process_style_resize(self, image):
        """
        Style模式逻辑: 短边缩放 + 中心裁剪
        - 正方形直接resize
        - 长方向先按比例缩放短边到target_size, 再裁剪中心
        """
        h, w, _ = image.shape

        # 计算缩放比例, 短边 = target_size
        scale = self.target_size / min(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 缩放 (使用 INTER_AREA 保持画质)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 如果缩放后刚好是正方形 如 1024x1024 -> 512x512
        if new_w == self.target_size and new_h == self.target_size:
            return resized
            
        # 如果不是正方形, 进行中心裁剪
        # 计算中心点
        center_x = new_w // 2
        center_y = new_h // 2
        half = self.target_size // 2
        
        x1 = center_x - half
        y1 = center_y - half
        x2 = x1 + self.target_size
        y2 = y1 + self.target_size
        
        # 裁剪出
        final_img = resized[y1:y2, x1:x2]
        return final_img



    def get_crop_box(self, image_np, mode="person"):
        """
        裁剪算法逻辑
        """
        h, w, _ = image_np.shape

        # 分支A: 风格模式(style)
        if mode == "style":
            return self._get_center_crop_box(h, w)

        # 分支B: 人像模式(person)
        results = self.face_detection.process(image_np)

        cx, cy = w // 2, h // 2 
        crop_size = min(h, w)

        # 如果检测到人脸
        if results.detections:
            # 遍历检测结果
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box

                # 获取人脸具体宽高
                face_w = int(bboxC.width * w)
                face_h = int(bboxC.height * h)

                # 计算人脸中心点
                face_cx = int((bboxC.xmin + bboxC.width / 2) * w)
                face_cy = int((bboxC.ymin + bboxC.height / 2) * h)

                # 核心逻辑还原 2.5倍扩展
                face_max_side = max(face_w, face_h)
                crop_size = int(face_max_side * 2.5)

                # 限制裁剪框
                crop_size = min(crop_size, min(h, w))
                crop_size = max(crop_size, 256) 

                # 计算裁剪框坐标
                half_size = crop_size // 2
                x1 = face_cx - half_size
                y1 = face_cy - half_size
                x2 = face_cx + half_size
                y2 = face_cy + half_size

                # 边界修正逻辑还原
                if x1 < 0: 
                    x2 -= x1 
                    x1 = 0
                if y1 < 0: 
                    y2 -= y1
                    y1 = 0
                if x2 > w: 
                    x1 -= (x2 - w)
                    x2 = w
                if y2 > h: 
                    y1 -= (y2 - h)
                    y2 = h

                # 二次检查
                x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
                return int(x1), int(y1), int(x2), int(y2)

        # Fallback
        return self._get_center_crop_box(h, w)

    def process_image(self, image_path, output_path, mode="person"):
        try:
            # 保持这个读取逻辑, 防止中文路径报错
            img_stream = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
            
            if image is None:
                print(f"无法读取图片: {image_path}")
                return False
            
            final_img = None
            if mode == "style":
                final_img = self._process_style_resize(image)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # 获取裁剪坐标
                x1, y1, x2, y2 = self.get_crop_box(image_rgb, mode=mode)
                # 裁剪
                cropped = image[y1:y2, x1:x2]
                # Resize
                final_img = cv2.resize(cropped, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

            # Save
            is_success, buffer = cv2.imencode(".png", final_img)
            if is_success:
                buffer.tofile(output_path)
            
            return True
        
        except Exception as e:
            print(f"处理图片出错 {image_path}: {e}")
            return False
    
    def process_folder(self, input_folder, output_folder, mode="person"):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

        print(f"[SmartCrop] 发现 {len(files)} 张图片. 模式: {mode}")

        for file in tqdm(files):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)
            output_path = os.path.splitext(output_path)[0] + ".png"

            self.process_image(input_path, output_path, mode=mode)