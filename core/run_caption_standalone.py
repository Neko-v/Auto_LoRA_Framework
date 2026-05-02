# /core/run_caption_standalone.py

import sys
import os
import argparse
import traceback
import gc


# 路径补丁 (子进程必须加)
current_file_path = os.path.abspath(__file__)
core_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(core_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="图片文件夹路径")
    parser.add_argument("--trigger", type=str, default="", help="触发词")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["blip", "wd14"],
        help="打标方法",
    )
    args = parser.parse_args()

    print(f"[子进程] 启动独立打标任务: {args.method.upper()}", flush=True)
    print(f"[子进程] 目标文件夹: {args.folder}", flush=True)

    exit_code = 0

    try:
        if args.method == "wd14":
            from core.caption_wd14 import WD14Tagger

            tagger = WD14Tagger()
            tagger.process_folder(args.folder, trigger_word=args.trigger)

            # 主动释放引用
            del tagger

        elif args.method == "blip":
            from core.caption_blip import BlipCaptioner

            captioner = BlipCaptioner()
            captioner.process_folder(args.folder, trigger_word=args.trigger)

            # 主动释放引用
            del captioner

        print(f"[子进程] {args.method.upper()} 打标完成", flush=True)

    except Exception as e:
        exit_code = 1
        print(f"[子进程] 发生严重错误: {e}", flush=True)
        traceback.print_exc()

    finally:
        # 尽量正常释放 Python 层资源
        try:
            gc.collect()
        except Exception:
            pass

        # 如果当前子进程中加载过 torch, 则清理 CUDA 缓存
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        print("[子进程] 正在强制退出, 释放深度学习运行时资源...", flush=True)

        # 确保日志先刷出来
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        # 在 Windows + CUDA + PyTorch / Transformers 环境下
        # sys.exit() 有概率卡在解释器清理阶段
        # 这里是独立子进程, 任务已经完成, 使用 os._exit 是可接受的
        os._exit(exit_code)


if __name__ == "__main__":
    main()