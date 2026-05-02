# /core/job_manager.py

import os
import json
import subprocess
from datetime import datetime


class JobManager:
    """
    负责管理单次 Auto_LoRA 任务的目录 日志 状态 配置快照和训练命令
    
    最终结构: 
    dataset/jobs/job_xxx/
        image/
            10_sivi person/
                xxx.png
                xxx.txt
        pipeline.log
        status.json
        config_snapshot.json
        train_command.txt
    """

    def __init__(self, project_root, trigger_word):
        self.project_root = project_root
        self.trigger_word = trigger_word

        self.job_name = self._build_job_name(trigger_word)

        # 一个 job 一个总目录
        self.job_dir = os.path.join(
            self.project_root,
            "runs",
            self.job_name,
        )

        # Kohya 训练数据根目录
        # train_data_dir 会指向这里
        self.image_root = os.path.join(self.job_dir, "image")
        self.output_dir = os.path.join(self.job_dir, "output")

        os.makedirs(self.image_root, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # 任务记录文件全部放在 job_dir 根目录
        self.pipeline_log_path = os.path.join(self.job_dir, "pipeline.log")
        self.status_path = os.path.join(self.job_dir, "status.json")
        self.config_snapshot_path = os.path.join(self.job_dir, "config_snapshot.json")
        self.train_command_path = os.path.join(self.job_dir, "train_command.txt")

    @staticmethod
    def safe_name(name):
        """清理文件夹名中的非法字符"""
        invalid_chars = '<>:"/\\|?*'
        for ch in invalid_chars:
            name = name.replace(ch, "_")
        return name.strip()

    def _build_job_name(self, trigger_word):
        """生成 job 名称"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_trigger = self.safe_name(trigger_word)
        return f"job_{timestamp}_{safe_trigger}"

    def write_json(self, path, data):
        """写 JSON 文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def update_status(self, status, stage=None, message=None):
        """更新当前任务状态"""
        data = {
            "status": status,
            "stage": stage,
            "message": message,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.write_json(self.status_path, data)

    def append_log(self, msg):
        """把完整日志追加写入 pipeline.log"""
        with open(self.pipeline_log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    def save_config_snapshot(self, config):
        """保存本次任务配置快照"""
        config = dict(config)
        config["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.write_json(self.config_snapshot_path, config)

    def save_train_command(self, train_cmd):
        """
        保存训练命令
        使用 subprocess.list2cmdline(), 让 Windows 下的路径和空格更安全
        """
        with open(self.train_command_path, "w", encoding="utf-8") as f:
            f.write(subprocess.list2cmdline(train_cmd))

        return self.train_command_path