# /core/run_crop_standalone.py

import sys
import os
import argparse
import traceback


# 路径补丁: 保证子进程可以正确 import core
current_file_path = os.path.abspath(__file__)
core_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(core_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="原始图片文件夹路径")
    parser.add_argument("--output", type=str, required=True, help="裁剪后输出文件夹路径")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["person", "style"],
        help="裁剪模式: person 或 style",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="目标分辨率，例如 512 或 1024",
    )

    args = parser.parse_args()

    print("[裁剪子进程] 启动 SmartCrop 任务")
    print(f"[裁剪子进程] 输入目录: {args.input}")
    print(f"[裁剪子进程] 输出目录: {args.output}")
    print(f"[裁剪子进程] 模式: {args.mode}")
    print(f"[裁剪子进程] 分辨率: {args.resolution}")

    try:
        from core.smart_crop import SmartCropper

        cropper = SmartCropper(target_size=args.resolution)
        cropper.process_folder(
            input_folder=args.input,
            output_folder=args.output,
            mode=args.mode,
        )

        print("[裁剪子进程] SmartCrop 裁剪完成")
        sys.exit(0)

    except Exception as e:
        print(f"[裁剪子进程] 发生严重错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()