# /gui_en.py

import gradio as gr
import os
import sys

# 引入核心模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from core.pipeline_manager import AutoPipeline
from core.train_manager import TrainManager
from core.job_manager import JobManager
from core.train_config import TrainConfig
from core.dataset_validator import validate_dataset_folder
from core.process_runner import cancel_active_process
from core.train_profiles import get_profile_names, get_profile_config, DEFAULT_PROFILE_NAME

# 全局常量
PROJECT_ROOT = current_dir
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SD_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "sd-scripts")

ACTIVE_JOB = {
    "job": None
}

STOP_REQUESTED = {
    "value": False
}

ACTIVE_UI_LOGS = {
    "lines": []
}

MAX_LOG_LINES = 60

def reset_ui_logs():
    ACTIVE_UI_LOGS["lines"] = []

def push_ui_log(msg):
    logs = ACTIVE_UI_LOGS["lines"]
    logs.append(str(msg))

    if len(logs) > MAX_LOG_LINES:
        ACTIVE_UI_LOGS["lines"] = logs[-MAX_LOG_LINES:]

    return "\n".join(ACTIVE_UI_LOGS["lines"])


def clear_active_job(job):
    """清理当前活跃任务引用"""
    if ACTIVE_JOB.get("job") is job:
        ACTIVE_JOB["job"] = None


def stop_current_job():
    """
    GUI 停止按钮调用
    负责标记取消状态, 终止当前活跃子进程, 并把停止信息写回主日志框
    """
    STOP_REQUESTED["value"] = True

    killed = cancel_active_process()

    job = ACTIVE_JOB.get("job")

    if killed:
        messages = [
            "Stop button clicked. Terminating the current task...",
            "Task stopped by user",
        ]
    else:
        messages = [
            "Stop button clicked",
            "No active subprocess was detected. The task may have already finished or has not started yet",
        ]

    for msg in messages:
        print(msg)

        if job is not None:
            job.append_log(msg)

        push_ui_log(msg)

    if job is not None:
        job.update_status(
            status="cancelled",
            stage="cancelled",
            message="Task stopped by user",
        )
        clear_active_job(job)

    return "\n".join(ACTIVE_UI_LOGS["lines"])

def get_base_models():
    """自动扫描 models 文件夹下的 .safetensors 文件"""
    if not os.path.exists(MODELS_DIR):
        return ["Model file not found"]

    files = sorted(
        f for f in os.listdir(MODELS_DIR)
        if f.lower().endswith(".safetensors")
    )
    return files if files else ["Model file not found"]

def apply_profile(profile_name):
    """将训练 Profile 应用到 GUI 控件"""
    profile = get_profile_config(profile_name)

    return (
        profile["mode_selection"],
        profile["repeats"],
        profile["epochs"],
        profile["network_dim"],
        profile["network_alpha"],
        profile["learning_rate"],
        profile["text_encoder_lr"],
        profile["unet_lr"],
        profile["batch_size"],
        profile["save_every_n_epochs"],
        profile["seed"],
        profile["description"],
    )

def run_training_process(
    folder_path,
    trigger_word,
    mode_selection,
    profile_name,
    base_model_name,
    repeats,
    epochs,
    network_dim,
    network_alpha,
    learning_rate,
    text_encoder_lr,
    unet_lr,
    batch_size,
    save_every_n_epochs,
    seed,
):
    """
    连接 UI 和 核心逻辑
    """
    STOP_REQUESTED["value"] = False
    reset_ui_logs()
    # 1. 基础校验
    trigger_word = trigger_word.strip() if trigger_word else ""
    if not folder_path or not os.path.exists(folder_path):
        yield "ERROR: Image folder path does not exist", None
        return
    
    if not trigger_word:
        yield "ERROR: Trigger word is required", None
        return

    if not base_model_name or base_model_name in ["Model file not found", "No models"]:
        yield "ERROR: Please place the bottom mold in the models folder first", None
        return

    # 2. 确定参数
    # 根据用户选择的模式, 决定内部参数
    if mode_selection == "人物 (Person)":
        train_mode = "person"
        caption_method = "blip"
        class_name = "person"
    else: # 画风
        train_mode = "style"
        caption_method = "wd14"
        class_name = "style"

    # 构造完整底模路径
    base_model_path = os.path.join(MODELS_DIR, base_model_name)
    if not os.path.exists(base_model_path):
        yield f"ERROR: Base model file not found: {base_model_path}", None
        return
    
    # Alpha 校验
    if int(network_alpha) > int(network_dim):
        yield "ERROR: LoRA Alpha current recommendation is less than or equal to Rank / Network Dim", None
        return
    
    try:
        float(str(learning_rate).strip())
        float(str(text_encoder_lr).strip())
        float(str(unet_lr).strip())
    except ValueError:
        yield "ERROR: Invalid learning rate format. Please use a format such as 1e-4 or 0.0001", None
        return
    
    # 创建本次任务的日志目录
    job = JobManager(PROJECT_ROOT, trigger_word)
    ACTIVE_JOB["job"] = job
    job.update_status(status="running", stage="init", message="Task Initialization")

    train_config = TrainConfig(
        resolution=512,
        max_train_epochs=int(epochs),
        network_dim=int(network_dim),
        network_alpha=int(network_alpha),
        learning_rate=str(learning_rate).strip(),
        text_encoder_lr=str(text_encoder_lr).strip(),
        unet_lr=str(unet_lr).strip(),
        train_batch_size=int(batch_size),
        save_every_n_epochs=int(save_every_n_epochs),
        seed=int(seed),
    )

    config_snapshot = {
        "job_name": job.job_name,
        "job_dir": job.job_dir,
        "job_image_root": job.image_root,
        "job_output_dir": job.output_dir,

        "trigger_word": trigger_word,
        "profile_name": profile_name,
        "mode_selection": mode_selection,
        "train_mode": train_mode,
        "caption_method": caption_method,
        "class_name": class_name,
        "folder_path": folder_path,
        "base_model_name": base_model_name,
        "base_model_path": base_model_path,
        "repeats": int(repeats),
        "epochs": int(epochs),
        "resolution": 512,
        "train_config": train_config.to_dict(),
    }

    job.save_config_snapshot(config_snapshot)
    
    # 定义 Log 记录器
    def update_log(msg):
        print(msg)
        job.append_log(msg)
        return push_ui_log(msg)

    # 阶段1 数据准备
    yield update_log(f"Task started: {trigger_word}"), None
    yield update_log(f"Data source: {folder_path}"), None
    yield update_log(f"Mode: {train_mode} | Caption method: {caption_method}"), None
    yield update_log(f"Job Log Directory: {job.job_dir}"), None

    yield update_log("Initializing pipeline..."), None
    
    try:
        # 实例化 Pipeline
        pipeline = AutoPipeline(
            base_dir=PROJECT_ROOT,
            source_dir=folder_path,
            resolution=512,
            caption_method=caption_method,
            dataset_dir=job.job_dir,
        )
        
        # Step 1: 建立目录
        yield update_log("Creating training directory..."), None
        dataset_path = pipeline.setup_directories(
            instance_name=trigger_word,
            class_name=class_name,
            repeats=int(repeats)
        )
        
        # Step 2: 智能裁剪
        yield update_log("Running Smart Crop..."), None
        job.update_status(status="running", stage="crop", message="Cropping image")

        try:
            for line in pipeline.run_crop_stream(dataset_path, mode=train_mode):
                yield update_log(line), None
        except Exception as e:
            if STOP_REQUESTED["value"]:
                job.update_status(status="cancelled", stage="crop", message="Task stopped by user")
            else:
                job.update_status(status="failed", stage="crop", message=str(e))
                yield update_log(f"Cropping failed: {e}"), None

            clear_active_job(job)
            return
            
        # Step 3: 自动打标
        yield update_log(f"Running automatic captioning ({caption_method})..."), None
        job.update_status(status="running", stage="caption", message="Automatically captioning")

        try:
            for line in pipeline.run_caption_stream(dataset_path, trigger_word, class_name):
                yield update_log(line), None
        except Exception as e:
            if STOP_REQUESTED["value"]:
                job.update_status(status="cancelled", stage="caption", message="Task stopped by user")
            else:
                job.update_status(status="failed", stage="caption", message=str(e))
                yield update_log(f"Captioning Failed: {e}"), None

            clear_active_job(job)
            return

        # Step 4: 数据集校验
        yield update_log("Validating training dataset..."), None
        job.update_status(status="running", stage="validate", message="Checking training dataset")

        validation_result = validate_dataset_folder(dataset_path)

        for line in validation_result.report_lines():
            yield update_log(line), None

        if not validation_result.valid:
            job.update_status(
                status="failed",
                stage="validate",
                message="Dataset validation failed",
            )
            yield update_log("ERROR: Dataset validation failed, training has stopped"), None
            clear_active_job(job)
            return

        yield update_log("Dataset preparation completed"), None

    except Exception as e:
        if STOP_REQUESTED["value"]:
            job.update_status(status="cancelled", stage="prepare", message="Task stopped by user")
        else:
            job.update_status(status="failed", stage="prepare", message=str(e))
            yield update_log(f"Stage 1 Exception Occurs: {str(e)}"), None
            import traceback
            traceback.print_exc()

        clear_active_job(job)
        return

    # 阶段2 开始训练
    yield update_log("Starting Kohya training backend..."), None
    job.update_status(status="running", stage="train", message="Starting training")
    yield update_log("Training logs will be displayed here in real time..."), None
    
    train_data_root = os.path.dirname(dataset_path)
    output_instance_dir = job.output_dir
    os.makedirs(output_instance_dir, exist_ok=True)

    try:
        trainer = TrainManager(base_dir=PROJECT_ROOT, sd_scripts_dir=SD_SCRIPTS_DIR)

        train_cmd = trainer.build_command(
            base_model_path=base_model_path,
            train_data_dir=train_data_root,
            output_dir=output_instance_dir,
            output_name=trigger_word,
            config=train_config,
        )

        train_command_path = job.save_train_command(train_cmd)

        yield update_log(f"Training command saved: {train_command_path}"), None

        for line in trainer.run_training_stream(
            base_model_path=base_model_path,
            train_data_dir=train_data_root,
            output_dir=output_instance_dir,
            output_name=trigger_word,
            config=train_config,
        ):
            yield update_log(line), None

        model_file = os.path.join(output_instance_dir, f"{trigger_word}.safetensors")

        job.update_status(status="success", stage="done", message=f"Model saved: {model_file}")
        
        yield update_log("Training completed successfully"), model_file
        yield update_log(f"Model path: {model_file}"), model_file

        clear_active_job(job)

    except Exception as e:
        if STOP_REQUESTED["value"]:
            job.update_status(status="cancelled", stage="train", message="Task stopped by user")
        else:
            job.update_status(status="failed", stage="train", message=str(e))
            yield update_log(f"Training Failed: {e}"), None

        clear_active_job(job)
        return


# UI 布局
CUSTOM_CSS = """
#advanced-settings .label-wrap {
    font-size: 18px !important;
    font-weight: 700 !important;
}

#advanced-settings .label-wrap span {
    font-size: 18px !important;
    font-weight: 700 !important;
}
"""

with gr.Blocks(
    title="Auto_LoRA Trainer Pro",
    theme=gr.themes.Soft(),
    css=CUSTOM_CSS
) as demo:
    
    default_profile = get_profile_config(DEFAULT_PROFILE_NAME)

    gr.Markdown("# Auto_LoRA")
    gr.Markdown("Prepare image datasets and train LoRA models with one click.")

    with gr.Row():
        # 左侧设置区
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 1. Data Source")
                # 默认值
                folder_input = gr.Textbox(
                    label="Image Folder Path", 
                    value=r"V:\Auto_LoRA\LoRA-AutoTrainer\data\Your_file",
                    placeholder="Example: V:\Auto_LoRA\data\My_Photos"
                )
                trigger_input = gr.Textbox(
                    label="Trigger Word / Instance Name", 
                    value="sivi",
                    placeholder="Example: sivi"
                )

            with gr.Group():
                gr.Markdown("### 2. Training Configuration")

                profile_dropdown = gr.Dropdown(
                    choices=get_profile_names(),
                    value=DEFAULT_PROFILE_NAME,
                    label="Training Profile"
                )

                profile_description = gr.Markdown(
                    get_profile_config(DEFAULT_PROFILE_NAME)["description"]
                )

                mode_radio = gr.Radio(
                    choices=["人物 (Person)", "画风 (Style)"],
                    value=default_profile["mode_selection"],
                    label="Training Mode"
                )
                
                # 自动读取 models 文件夹
                model_list = get_base_models()
                model_dropdown = gr.Dropdown(
                    choices=model_list,
                    value=model_list[0] if model_list else None,
                    label="Base Model"
                )
                
                with gr.Row():
                    repeats_slider = gr.Slider(
                        minimum=10,
                        maximum=100,
                        step=10,
                        value=default_profile["repeats"],
                        label="Repeats Per Image"
                    )
                    epochs_slider = gr.Slider(
                        minimum=1,
                        maximum=30,
                        step=1,
                        value=default_profile["epochs"],
                        label="Epochs"
                    )
            
            with gr.Group():
                with gr.Accordion("Advanced Training Parameters", open=False, elem_id="advanced-settings"):
                    gr.Markdown(
                        "These settings are intended for advanced users. For 6-8GB GPUs, keep Batch Size = 1"
                    )
                    with gr.Row():
                        network_dim_slider = gr.Slider(
                            minimum=4,
                            maximum=128,
                            step=4,
                            value=default_profile["network_dim"],
                            label="LoRA Rank / Network Dim"
                        )
                        network_alpha_slider = gr.Slider(
                            minimum=1,
                            maximum=128,
                            step=1,
                            value=default_profile["network_alpha"],
                            label="LoRA Alpha"
                        )
                        gr.Markdown(
                            "Rank controls LoRA capacity. Higher values can learn more detail but may overfit more easily. Alpha is usually set to half of Rank or equal to Rank"
                        )

                    with gr.Row():
                        learning_rate_input = gr.Textbox(
                            label="Learning Rate",
                            value=default_profile["learning_rate"]
                        )
                        text_encoder_lr_input = gr.Textbox(
                            label="Text Encoder LR",
                            value=default_profile["text_encoder_lr"]
                        )
                        unet_lr_input = gr.Textbox(
                            label="UNet LR",
                            value=default_profile["unet_lr"]
                        )
                        gr.Markdown(
                            "Learning Rate controls training strength. Too high may cause overfitting or unstable results; too low may underfit"
                        )

                    with gr.Row():
                        batch_size_slider = gr.Slider(
                            minimum=1,
                            maximum=8,
                            step=1,
                            value=default_profile["batch_size"],
                            label="Batch Size"
                        )
                        gr.Markdown(
                            "Higher Batch Size uses more VRAM. For consumer 6-8GB GPUs, Batch Size = 1 is recommended"
                        )
                        save_every_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=default_profile["save_every_n_epochs"],
                            label="Save Every N Epochs"
                        )
                        seed_input = gr.Number(
                            label="Seed",
                            value=default_profile["seed"],
                            precision=0
                        )
                
            with gr.Row():
                start_btn = gr.Button("Start Training", variant="primary", size="lg")
                stop_btn = gr.Button("Stop Training", variant="stop", size="lg")

        # 右侧日志区
        with gr.Column(scale=6):
            gr.Markdown("### Runtime Log")
            log_output = gr.Code(
                label="System Status", 
                language="shell", 
                lines=20,
                interactive=False
            )
            
            gr.Markdown("### Training Result")
            result_file = gr.File(label="Download Generated Model", interactive=False)

    # 绑定点击事件
    profile_dropdown.change(
        fn=apply_profile,
        inputs=[profile_dropdown],
        outputs=[
            mode_radio,
            repeats_slider,
            epochs_slider,
            network_dim_slider,
            network_alpha_slider,
            learning_rate_input,
            text_encoder_lr_input,
            unet_lr_input,
            batch_size_slider,
            save_every_slider,
            seed_input,
            profile_description,
        ],
    )
    run_event = start_btn.click(
        fn=run_training_process,
        inputs=[
            folder_input,
            trigger_input,
            mode_radio,
            profile_dropdown,
            model_dropdown,
            repeats_slider,
            epochs_slider,
            network_dim_slider,
            network_alpha_slider,
            learning_rate_input,
            text_encoder_lr_input,
            unet_lr_input,
            batch_size_slider,
            save_every_slider,
            seed_input,
        ],
        outputs=[log_output, result_file]
    )

    stop_btn.click(
        fn=stop_current_job,
        inputs=[],
        outputs=[log_output],
        cancels=[run_event],
    )

if __name__ == "__main__":
    print("Launching WebUI...")
    # inbrowser=True 会自动在浏览器打开
    demo.queue().launch(inbrowser=True, show_error=True)