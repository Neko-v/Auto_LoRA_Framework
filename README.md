# Auto_LoRA_Framework

A lightweight automation framework for training LoRA models, integrating `sd-scripts` for underlying training tasks.

This project supports both a graphical user interface (GUI) and a command-line interface (CLI) for flexible training workflows.

## Installation

Ensure you have Python 3.10+ installed.

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

**GUI Mode(Recommended)**
To launch the graphical interface for configuration and monitoring:
    python gui.py

**CLI Mode**
To run the trainer directly from the command line:
    python main.py

## Project Structure

- `core/`: Core application logic, configuration handlers, and process wrappers.
- `models/`: Directory for storing base models (excluded from version control).
- `output/`: Default directory for training artifacts (excluded from version control).
- `sd-scripts/`: Submodule containing the underlying Stable Diffusion training scripts.
- `gui.py`: Entry point for the graphical interface.
- `main.py`: Entry point for command-line execution.

## Notes

- Large model files (e.g., .safetensors, .ckpt) are ignored by .gitignore to keep the repository size manageable.
- Please ensure your base model is placed in the models/ directory before starting a training session.