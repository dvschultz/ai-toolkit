# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Toolkit by Ostris — an all-in-one training suite for diffusion models (image and video). Supports LoRA, DreamBooth, full fine-tuning, sliders, and more across consumer-grade GPU hardware. Runs as CLI or web UI.

## Commands

### Running a training job (CLI)
```bash
python run.py config/my_config.yml
# Multiple configs sequentially:
python run.py config/a.yml config/b.yml
# Continue on failure:
python run.py -r config/a.yml config/b.yml
# Replace [name] tag in config:
python run.py -n my_run config/my_config.yml
```

### Running the Web UI
```bash
cd ui
npm run build_and_start
# Accessible at http://localhost:8675
# Secure with: AI_TOOLKIT_AUTH=password npm run build_and_start
```

### UI development
```bash
cd ui
npm run dev  # Next.js dev server on port 3000
```

### Gradio UI (simple LoRA trainer)
```bash
python flux_train_ui.py
```

### Installation
```bash
python3 -m venv venv && source venv/bin/activate
pip3 install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt
```

### Debug mode
Set `DEBUG_TOOLKIT=1` env var to enable `torch.autograd.set_detect_anomaly(True)`.

## Architecture

### Job System (entry point: `run.py` → `toolkit/job.py`)

Everything starts with a YAML/JSON config file. The `job` key determines which Job class handles it:

| `job` value | Class | Purpose |
|---|---|---|
| `extension` | `ExtensionJob` | **Primary path** — delegates to extensions via `type` in process config |
| `train` | `TrainJob` | Legacy training (sliders, VAE, ESRGAN, etc.) |
| `extract` | `ExtractJob` | Extract LoRA/LoCon from model diffs |
| `generate` | `GenerateJob` | Image generation |
| `mod` | `ModJob` | Modify/rescale models |

**Most training goes through `ExtensionJob`** with `type: "sd_trainer"` in the process config.

### Config System (`toolkit/config.py`)

- Configs can be JSON, JSONC, or YAML
- Looks first in `config/` directory, then as absolute/relative path
- Supports `${ENV_VAR}` substitution and `[name]` tag replacement
- Required structure: `{ job: "...", config: { name: "...", process: [...] } }`
- Example configs in `config/examples/`

### Extension System (`toolkit/extension.py`)

Extensions are discovered by scanning `extensions/` (user) and `extensions_built_in/` (built-in) directories. Each package exposes an `AI_TOOLKIT_EXTENSIONS` list of Extension subclasses. Each Extension has a `uid` (used as the `type` in config) and a `get_process()` classmethod returning the process class.

Key built-in extensions:
- `sd_trainer` — Main LoRA/DreamBooth/fine-tune trainer (`extensions_built_in/sd_trainer/SDTrainer.py`)
- `diffusion_trainer` — Universal trainer for UI/API
- `concept_slider`, `ultimate_slider_trainer` — Slider training
- `reference_generator` — Reference-based generation
- `dataset_tools` — Dataset preparation utilities

### Model System (`toolkit/util/get_model.py`)

Models are registered via `AI_TOOLKIT_MODELS` lists in extension packages. The system matches `ModelConfig.arch` to find the right class. Falls back to `StableDiffusion` (the legacy catch-all in `toolkit/stable_diffusion_model.py`).

Newer models live in `extensions_built_in/diffusion_models/` (Chroma, HiDream, Flux2, Wan2.2, OmniGen2, Kontext, Qwen, LTX2, etc.). Some older models live in `toolkit/models/` (Wan2.1, CogView4).

### Process Class Hierarchy

```
BaseProcess (jobs/process/BaseProcess.py)
├── BaseTrainProcess
│   └── BaseSDTrainProcess — core SD training loop, used by SDTrainer
├── BaseExtensionProcess — base for extension processes
├── BaseExtractProcess
└── GenerateProcess
```

`BaseSDTrainProcess` (`jobs/process/BaseSDTrainProcess.py`) contains the heavy training logic: dataset loading, optimizer setup, training loop, sampling, checkpointing. `SDTrainer` extends it with model-specific forward pass and loss computation.

### Key Modules in `toolkit/`

- `stable_diffusion_model.py` / `models/base_model.py` — Model loading, inference, quantization. Wraps diffusers pipelines.
- `config_modules.py` — Pydantic-style config dataclasses: `ModelConfig`, `TrainConfig`, `NetworkConfig`, `SaveConfig`, `SampleConfig`, `DatasetConfig`
- `data_loader.py` — Dataset loading with bucketing, latent caching, caption handling. Uses mixins from `dataloader_mixins.py`.
- `lora_special.py` / `lycoris_special.py` — LoRA and LyCORIS network implementations
- `optimizer.py` — Optimizer factory (AdamW, AdamW8bit, Prodigy, etc.)
- `sampler.py` / `samplers/` — Sampling during training for visual progress
- `saving.py` — Checkpoint saving in safetensors/diffusers format
- `accelerator.py` — Singleton HuggingFace Accelerator wrapper

### UI (`ui/`)

Next.js app (TypeScript, Tailwind, Prisma). Manages training jobs via the web. The Python backend communicates through the same config/job system.

### Docker

`docker-compose.yml` mounts HF cache, datasets, output, and config. Exposes port 8675 for the UI.

## Config File Conventions

All modern training configs use `job: extension` with `type: "sd_trainer"`. The process config contains:
- `network` — LoRA/LoKr config (type, rank, alpha)
- `train` — Training params (steps, lr, optimizer, dtype, batch_size, gradient_checkpointing)
- `model` — Model source (name_or_path, quantize, is_flux, etc.)
- `datasets` — List of dataset folders with caption settings, resolution, caching
- `save` — Checkpoint frequency, format, hub push settings
- `sample` — Sampling config for visual progress during training

## Environment Variables

- `HF_TOKEN` — HuggingFace token (or use `.env` file)
- `DEBUG_TOOLKIT=1` — Enable PyTorch anomaly detection
- `MODELS_PATH` — Override default models directory
- `COMFY_PATH` — ComfyUI integration path
- `AI_TOOLKIT_AUTH` — Auth token for the web UI
