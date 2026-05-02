# /core/dataset_validator.py

import os
from dataclasses import dataclass, field
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass
class DatasetValidationResult:
    valid: bool
    image_count: int = 0
    caption_count: int = 0
    missing_caption_count: int = 0
    empty_caption_count: int = 0
    corrupted_image_count: int = 0
    orphan_caption_count: int = 0

    missing_captions: list[str] = field(default_factory=list)
    empty_captions: list[str] = field(default_factory=list)
    corrupted_images: list[str] = field(default_factory=list)
    orphan_captions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def report_lines(self):
        lines = []

        if self.valid:
            lines.append("数据集检查通过")
        else:
            lines.append("数据集检查失败")

        lines.append(f"图片数量: {self.image_count}")
        lines.append(f"Caption 数量: {self.caption_count}")
        lines.append(f"缺失 Caption: {self.missing_caption_count}")
        lines.append(f"空 Caption: {self.empty_caption_count}")
        lines.append(f"损坏图片: {self.corrupted_image_count}")
        lines.append(f"孤立 Caption: {self.orphan_caption_count}")

        if self.missing_captions:
            lines.append("缺失 Caption 文件:")
            for item in self.missing_captions[:10]:
                lines.append(f"  - {item}")
            if len(self.missing_captions) > 10:
                lines.append(f"  ... 还有 {len(self.missing_captions) - 10} 个")

        if self.empty_captions:
            lines.append("空 Caption 文件:")
            for item in self.empty_captions[:10]:
                lines.append(f"  - {item}")
            if len(self.empty_captions) > 10:
                lines.append(f"  ... 还有 {len(self.empty_captions) - 10} 个")

        if self.corrupted_images:
            lines.append("损坏或无法读取的图片:")
            for item in self.corrupted_images[:10]:
                lines.append(f"  - {item}")
            if len(self.corrupted_images) > 10:
                lines.append(f"  ... 还有 {len(self.corrupted_images) - 10} 个")

        if self.orphan_captions:
            lines.append("孤立 Caption 文件, 即存在 txt 但没有同名图片:")
            for item in self.orphan_captions[:10]:
                lines.append(f"  - {item}")
            if len(self.orphan_captions) > 10:
                lines.append(f"  ... 还有 {len(self.orphan_captions) - 10} 个")

        if self.warnings:
            lines.append("警告:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return lines


def _is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def _is_caption_file(filename):
    return filename.lower().endswith(".txt")


def _check_image_readable(image_path):
    """
    检查图片是否能被 PIL 正常读取
    Image.verify() 会验证文件结构, 但不会完整解码所有像素, 速度较快
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def validate_dataset_folder(folder_path):
    """
    校验 Kohya 单个实例训练文件夹, 例如: 
    runs/job_xxx/image/10_sivi person/

    要求: 
    - 至少有 1 张图片
    - 每张图片有同名 .txt
    - .txt 内容非空
    - 图片可以被正常读取
    """
    result = DatasetValidationResult(valid=False)

    if not os.path.exists(folder_path):
        result.warnings.append(f"数据集目录不存在: {folder_path}")
        return result

    if not os.path.isdir(folder_path):
        result.warnings.append(f"数据集路径不是文件夹: {folder_path}")
        return result

    folder_name = os.path.basename(folder_path)

    # 简单检查 Kohya DreamBooth 文件夹命名格式, 例如 10_sivi person
    if "_" not in folder_name:
        result.warnings.append(
            f"训练文件夹名可能不符合 Kohya 格式: {folder_name}, 建议类似 10_sivi person"
        )
    else:
        repeat_part = folder_name.split("_", 1)[0]
        if not repeat_part.isdigit():
            result.warnings.append(
                f"训练文件夹名的 repeats 部分不是数字: {folder_name}"
            )

    files = os.listdir(folder_path)

    image_files = sorted(f for f in files if _is_image_file(f))
    caption_files = sorted(f for f in files if _is_caption_file(f))

    result.image_count = len(image_files)
    result.caption_count = len(caption_files)

    image_stems = {os.path.splitext(f)[0] for f in image_files}
    caption_stems = {os.path.splitext(f)[0] for f in caption_files}

    # 没有图片直接失败
    if result.image_count == 0:
        result.warnings.append("数据集中没有找到图片文件")
        return result

    # 检查每张图片
    for image_file in image_files:
        stem = os.path.splitext(image_file)[0]
        image_path = os.path.join(folder_path, image_file)
        caption_path = os.path.join(folder_path, stem + ".txt")

        if not _check_image_readable(image_path):
            result.corrupted_images.append(image_file)

        if not os.path.exists(caption_path):
            result.missing_captions.append(image_file)
        else:
            try:
                with open(caption_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if not content:
                    result.empty_captions.append(stem + ".txt")

            except Exception:
                result.empty_captions.append(stem + ".txt")

    # 检查孤立 txt
    orphan_stems = caption_stems - image_stems
    result.orphan_captions = sorted(stem + ".txt" for stem in orphan_stems)

    result.missing_caption_count = len(result.missing_captions)
    result.empty_caption_count = len(result.empty_captions)
    result.corrupted_image_count = len(result.corrupted_images)
    result.orphan_caption_count = len(result.orphan_captions)

    # 孤立 caption 只警告, 不阻止训练
    if result.orphan_caption_count > 0:
        result.warnings.append(
            "发现孤立 Caption 文件, 它们不会被 Kohya 使用, 建议清理"
        )

    result.valid = (
        result.image_count > 0
        and result.missing_caption_count == 0
        and result.empty_caption_count == 0
        and result.corrupted_image_count == 0
    )

    return result