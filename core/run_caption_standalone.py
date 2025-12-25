# /core/run_caption_standalone.py

import sys
import os
import argparse


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
    parser.add_argument("--method", type=str, required=True, choices=["blip", "wd14"], help="打标方法")
    args = parser.parse_args()

    print(f"[子进程] 启动独立打标任务: {args.method.upper()}")
    print(f"[子进程] 目标文件夹: {args.folder}")

    try:
        if args.method == "wd14":
            # 只有用到的时候才 import, 节省资源
            from core.caption_wd14 import WD14Tagger
            tagger = WD14Tagger()
            # WD14 的 process_folder 参数逻辑和 BLIP 稍微有点区别, 这里统一一下
            # WD14 内部已经有了 process_folder, 这里直接调
            tagger.process_folder(args.folder, trigger_word=args.trigger)
            
        elif args.method == "blip":
            from core.caption_blip import BlipCaptioner
            captioner = BlipCaptioner()
            captioner.process_folder(args.folder, trigger_word=args.trigger)

        print(f"[子进程] {args.method.upper()} 打标完成")

    except Exception as e:
        print(f"[子进程] 发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 退出子进程
    print("[子进程]正在退出...")
    os._exit(0)

if __name__ == "__main__":
    main()