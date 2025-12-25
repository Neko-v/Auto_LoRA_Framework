# /gui.py

import gradio as gr
import os
import sys

# 引入核心模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from core.pipeline_manager import AutoPipeline
from core.train_manager import TrainManager

# 全局常量
PROJECT_ROOT = current_dir
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
SD_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "sd-scripts")

def get_base_models():
    """自动扫描 models 文件夹下的 .safetensors 文件"""
    if not os.path.exists(MODELS_DIR):
        return []
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".safetensors")]
    return files if files else ["No models"]

def run_training_process(folder_path, trigger_word, mode_selection, base_model_name, repeats, epochs):
    """
    连接 UI 和 核心逻辑
    """
    # 1. 基础校验
    if not folder_path or not os.path.exists(folder_path):
        yield "ERROR: 图片文件夹路径不存在", None
        return
    
    if not trigger_word:
        yield "ERROR: 必须填写触发词", None
        return

    if not base_model_name or base_model_name == "未找到模型文件":
        yield "ERROR: 请先在 models 文件夹放入底模", None
        return

    # 2. 确定参数
    # 根据用户选择的模式，决定内部参数
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
    
    # 定义 Log 记录器
    logs = []
    def update_log(msg):
        print(msg) # 同时打印到控制台
        logs.append(msg)
        return "\n".join(logs)

    # 阶段1 数据准备
    yield update_log(f"开始任务: {trigger_word}"), None
    yield update_log(f"数据源: {folder_path}"), None
    yield update_log(f"模式: {train_mode} | 打标: {caption_method}"), None
    
    yield update_log("正在初始化处理管线..."), None
    
    try:
        # 实例化 Pipeline
        pipeline = AutoPipeline(
            base_dir=PROJECT_ROOT,
            source_dir=folder_path,
            resolution=512,
            caption_method=caption_method
        )
        
        # Step 1: 建立目录
        yield update_log("正在建立训练目录..."), None
        dataset_path = pipeline.setup_directories(
            instance_name=trigger_word,
            class_name=class_name,
            repeats=int(repeats)
        )
        
        # Step 2: 智能裁剪
        yield update_log("正在进行智能裁剪 (Smart Crop)..."), None
        success_crop = pipeline.run_crop(dataset_path, mode=train_mode)
        
        if not success_crop:
            yield update_log("裁剪失败，请检查源文件夹是否为空"), None
            return
            
        # Step 3: 自动打标
        yield update_log(f"正在进行自动打标 ({caption_method})..."), None
        pipeline.run_caption(dataset_path, trigger_word, class_name)
        
        yield update_log("数据准备完成"), None

    except Exception as e:
        yield update_log(f"阶段1发生异常: {str(e)}"), None
        import traceback
        traceback.print_exc()
        return

    # 阶段2 开始训练
    yield update_log("正在启动 Kohya 训练内核..."), None
    yield update_log("P.S: 详细的进度条 (0% -> 100%) 请查看命令行窗口"), None
    
    train_data_root = os.path.dirname(dataset_path)
    output_instance_dir = os.path.join(OUTPUT_DIR, f"{trigger_word}_lora")
    os.makedirs(output_instance_dir, exist_ok=True)

    try:
        trainer = TrainManager(base_dir=PROJECT_ROOT, sd_scripts_dir=SD_SCRIPTS_DIR)
        
        # 开始训练
        success = trainer.run_training(
            base_model_path=base_model_path,
            train_data_dir=train_data_root,
            output_dir=output_instance_dir,
            output_name=trigger_word,
            resolution=512,
            max_train_epochs=int(epochs)
        )
        
        if success:
            model_file = os.path.join(output_instance_dir, f"{trigger_word}.safetensors")
            yield update_log(f"训练成功"), model_file
            yield update_log(f"模型位置: {model_file}"), model_file
        else:
            yield update_log("训练失败 (Kohya 报错)，请查看控制台信息"), None

    except Exception as e:
        yield update_log(f"阶段2发生异常: {str(e)}"), None


# UI 布局
with gr.Blocks(title="Auto_LoRA Trainer Pro", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# Auto_LoRA")
    gr.Markdown("上传图片，一键生成 LoRA 模型")

    with gr.Row():
        # 左侧设置区
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 1. 数据来源")
                # 默认值我填了你的路径，方便测试
                folder_input = gr.Textbox(
                    label="图片文件夹路径 (绝对路径)", 
                    value=r"V:\Auto_LoRA\LoRA-AutoTrainer\data\Hu_Ge",
                    placeholder="例如: V:\Auto_LoRA\data\My_Photos"
                )
                trigger_input = gr.Textbox(
                    label="触发词 (Instance Name)", 
                    value="sivi",
                    placeholder="例如: sivi"
                )

            with gr.Group():
                gr.Markdown("### 2. 训练配置")
                mode_radio = gr.Radio(
                    choices=["人物 (Person)", "画风 (Style)"],
                    value="人物 (Person)",
                    label="训练模式 (会自动调整裁剪和打标策略)"
                )
                
                # 自动读取 models 文件夹
                model_list = get_base_models()
                model_dropdown = gr.Dropdown(
                    choices=model_list,
                    value=model_list[0] if model_list else None,
                    label="选择底模 (Base Model)"
                )
                
                with gr.Row():
                    repeats_slider = gr.Slider(minimum=10, maximum=100, step=10, value=40, label="单图重复次数 (Repeats)")
                    epochs_slider = gr.Slider(minimum=1, maximum=30, step=1, value=10, label="训练轮数 (Epochs)")

            start_btn = gr.Button("立即开始训练", variant="primary", size="lg")

        # 右侧日志区
        with gr.Column(scale=6):
            gr.Markdown("### 运行日志")
            log_output = gr.Code(
                label="系统状态", 
                language="shell", 
                lines=20,
                interactive=False
            )
            
            gr.Markdown("### 训练结果")
            result_file = gr.File(label="下载生成的模型", interactive=False)

    # 绑定点击事件
    start_btn.click(
        fn=run_training_process,
        inputs=[folder_input, trigger_input, mode_radio, model_dropdown, repeats_slider, epochs_slider],
        outputs=[log_output, result_file]
    )

if __name__ == "__main__":
    print("启动 WebUI...")
    # inbrowser=True 会自动在浏览器打开
    demo.queue().launch(inbrowser=True, show_error=True)