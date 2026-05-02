# Auto_LoRA

Auto_LoRA is a lightweight automated LoRA training framework designed for small image datasets and consumer-grade GPUs.

It provides an end-to-end workflow for preparing image datasets, generating captions or tags, validating the dataset, and launching LoRA training through Kohya `sd-scripts`.

## Features

- Gradio WebUI
- Chinese GUI: `gui.py`
- English GUI: `gui_en.py`
- Person mode: SmartCrop + BLIP captioning
- Style mode: style resize/crop + WD14 tagging
- Dataset validation before training
- Real-time training logs in the GUI
- Stop training button
- Training profiles for 6-8GB consumer GPUs
- Isolated training jobs under `runs/`
- Per-job logs, configs, commands, processed images, captions, and model outputs

## Project Structure

```text
Auto_LoRA/
├── core/
│   ├── caption_blip.py
│   ├── caption_wd14.py
│   ├── dataset_validator.py
│   ├── job_manager.py
│   ├── pipeline_manager.py
│   ├── process_runner.py
│   ├── run_caption_standalone.py
│   ├── run_crop_standalone.py
│   ├── smart_crop.py
│   ├── train_config.py
│   ├── train_manager.py
│   └── train_profiles.py
│
├── data/
│   └── Your_Image_Folder/
│
├── models/
│   ├── your_base_model.safetensors
│   └── wd14/
│
├── runs/
│   └── job_YYYYMMDD_HHMMSS_trigger/
│       ├── image/
│       │   └── 10_trigger class/
│       │       ├── image_001.png
│       │       └── image_001.txt
│       ├── output/
│       │   └── trigger.safetensors
│       ├── pipeline.log
│       ├── status.json
│       ├── config_snapshot.json
│       └── train_command.txt
│
├── sd-scripts/
├── gui.py
├── gui_en.py
├── main.py
└── README.md
```

## Directory Overview

### `data/`

Stores the original source images.

Example:

```text
data/Hu_Ge/
data/Oil_Painting_1024/
```

### `models/`

Stores Stable Diffusion base models and auxiliary models.

Example:

```text
models/chilloutmix_NiPrunedFp32Fix.safetensors
models/stable-diffusion-v1-5.safetensors
models/wd14/
```

### `runs/`

Stores every training job as an isolated run.

Each job contains:

```text
image/                 Processed training images and captions
output/                Generated LoRA models
pipeline.log           Full runtime log
status.json            Current or final job status
config_snapshot.json   Full configuration snapshot
train_command.txt      Reproducible Kohya training command
```

### `sd-scripts/`

Contains Kohya `sd-scripts`, which is used as the underlying LoRA training backend.

## Installation

Python 3.10 is recommended.

### 1. Create and activate an environment

Using Conda:

```bash
conda create -n Auto_LoRA python=3.10
conda activate Auto_LoRA
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare `sd-scripts`

Make sure `sd-scripts` exists in the project root:

```text
Auto_LoRA/sd-scripts/
```

### 4. Place base models

Place your Stable Diffusion base model files under:

```text
models/
```

Supported model formats are typically:

```text
.safetensors
.ckpt
```

## Usage

### Chinese GUI

```bash
python gui.py
```

### English GUI

```bash
python gui_en.py
```

After launching, open the local Gradio URL shown in the terminal.

Usually it looks like:

```text
http://127.0.0.1:7860
```

## Basic Workflow

1. Put original images into a folder under `data/`.
2. Launch `gui.py` or `gui_en.py`.
3. Select the image folder.
4. Enter a trigger word.
5. Select a base model.
6. Choose a training profile.
7. Optionally adjust advanced training parameters.
8. Click the training button.
9. Check the generated job folder under `runs/`.

## Training Modes

### Person Mode

Recommended for:

- Real people
- Character LoRA
- Portrait datasets

Pipeline:

```text
SmartCrop → BLIP Captioning → Dataset Validation → LoRA Training
```

Person mode uses face-aware smart cropping and BLIP natural-language captioning.

### Style Mode

Recommended for:

- Painting style
- Illustration style
- Visual style LoRA
- Anime or stylized datasets

Pipeline:

```text
Style Resize/Crop → WD14 Tagging → Dataset Validation → LoRA Training
```

Style mode avoids face-focused cropping and uses WD14 tag generation.

## Training Profiles

Auto_LoRA includes built-in training profiles for common small-dataset use cases.

Example profiles:

- SD1.5 Person LoRA / 8GB Stable
- SD1.5 Style LoRA / 8GB Stable
- SD1.5 Low VRAM Conservative / 6-8GB

Profiles automatically fill in recommended values for:

- Training mode
- Repeats
- Epochs
- LoRA Rank
- LoRA Alpha
- Learning Rate
- Text Encoder LR
- UNet LR
- Batch Size
- Save interval
- Seed

Advanced users can still manually adjust these parameters.

For 6-8GB GPUs, Batch Size = 1 is recommended.

## Advanced Training Parameters

### LoRA Rank / Network Dim

Controls the capacity of the LoRA.

Higher values can learn more detail, but may increase overfitting risk and VRAM usage.

### LoRA Alpha

Controls LoRA scaling.

A common stable choice is half of Rank or equal to Rank.

Example:

```text
Rank = 32
Alpha = 16
```

### Learning Rate

Controls overall training strength.

If the learning rate is too high, the LoRA may overfit or produce unstable results.

If the learning rate is too low, the LoRA may underfit.

### Text Encoder LR

Controls the learning rate of the text encoder.

For character/person LoRA, a small value such as `5e-5` is commonly used.

### UNet LR

Controls the learning rate of the UNet.

A common SD1.5 LoRA value is `1e-4`.

### Batch Size

Controls how many images are processed per step.

For consumer 6-8GB GPUs, keep this at `1`.

### Save Every N Epochs

Controls how often intermediate models are saved.

### Seed

Controls reproducibility.

Using the same seed and config can help reproduce similar training behavior.

## Job Output

Each training task creates a separate job folder:

```text
runs/job_YYYYMMDD_HHMMSS_trigger/
```

Example:

```text
runs/job_20260502_230410_sivi/
```

Inside the job folder:

```text
image/
```

Contains the processed dataset in Kohya DreamBooth format:

```text
image/10_sivi person/
├── image_001.png
├── image_001.txt
├── image_002.png
└── image_002.txt
```

```text
output/
```

Contains generated LoRA model files.

```text
pipeline.log
```

Contains the complete runtime log.

```text
status.json
```

Stores the current or final task status.

Possible statuses include:

```text
running
success
failed
cancelled
```

```text
config_snapshot.json
```

Stores the full configuration used for this training job.

```text
train_command.txt
```

Stores the exact Kohya training command for reproduction.

## Dataset Validation

Before training starts, Auto_LoRA validates the processed dataset.

It checks:

- Image count
- Caption count
- Missing caption files
- Empty caption files
- Corrupted or unreadable images
- Kohya folder naming format

If validation fails, training is stopped before Kohya starts.

Example successful validation:

```text
Dataset validation passed
Image count: 20
Caption count: 20
Missing captions: 0
Empty captions: 0
Corrupted images: 0
```

## Stopping Training

The GUI provides a stop button.

When clicked, Auto_LoRA attempts to terminate the active subprocess tree and updates the job status to:

```json
{
  "status": "cancelled",
  "message": "Task stopped by user"
}
```

This is safer than closing the terminal with `Ctrl+C`.

## Notes

- Auto_LoRA is designed for small datasets and consumer GPUs.
- It intentionally avoids aggressive batch acceleration to reduce VRAM risk.
- Each job is isolated under `runs/`.
- The final LoRA model is stored inside the job's `output/` folder.
- Large model files should not be committed to Git.
- `models/`, `runs/`, and large output files should usually be ignored by version control.

## Recommended `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc

# Virtual environments
pyenv/
venv/
.env/

# Models and outputs
models/*.safetensors
models/*.ckpt
models/*.pt
models/*.pth
runs/
output/

# Logs
*.log

# OS
.DS_Store
Thumbs.db
```

## Troubleshooting

### The GUI cannot find my base model

Make sure the model file is placed under:

```text
models/
```

The GUI scans `.safetensors` files automatically.

### The dataset validation fails

Check whether every processed image has a matching `.txt` caption file.

Example:

```text
image_001.png
image_001.txt
```

### Training seems slow at the beginning

The first part of training may be slower because models, latents, optimizer states, and caches are being initialized.

### I see Windows warnings from PyTorch

Some PyTorch or `accelerate` warnings on Windows are harmless.

Auto_LoRA filters some high-frequency noisy logs where possible.

### The stop button says no active subprocess was detected

This usually means the task has already ended, has not started yet, or the current stage is not running an external subprocess.

## Project Positioning

Auto_LoRA is intended as a lightweight, automated LoRA training framework for:

- Small image datasets
- Personal character LoRA training
- Style LoRA training
- Consumer-grade GPUs
- Users who want a simple GUI workflow

It is not intended to be a large-scale distributed training system.

## MIT License

Copyright (c) 2025 张式微

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.