# Local LLM Testing Platform

A simple application for testing Hugging Face models on your local NVIDIA GPU.

## Features

- Test multiple Hugging Face models (text, vision, multimodal)
- Simple UI for loading models and generating outputs
- Optimized for NVIDIA GPUs (tested on RTX 4090)
- Monitor performance and memory usage

## Requirements

- NVIDIA GPU with 8GB+ VRAM (RTX 4090 recommended)
- Windows 11 with latest NVIDIA drivers
- Python 3.10
- CUDA Toolkit 12.1+

## Setup

```bash
# Clone repository
git clone https://github.com/yourusername/local-llm-testing.git
cd local-llm-testing

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

- Run the application: python app.py
- Select and load a model in the UI
- Use the text or vision tabs to test models
- Adjust parameters like temperature as needed

### Configuration
Edit `config/models.json` to add or modify models:

```json
{
  "text_models": {
    "phi-4-reasoning-plus": {
      "model_id": "microsoft/Phi-4-reasoning-plus",
      "type": "text",
      "precision": "fp16"
    }
  }
}
```