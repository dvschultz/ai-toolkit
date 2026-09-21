# Ostris AI Toolkit

AI Toolkit is an easy to use all in one training suite for diffusion models. I try to support all the latest models on consumer grade hardware. Image and video models. It can be run as a GUI or CLI. It is designed to be easy to use but still have every feature imaginable. Free and open source.



## Supported Models

The following models are supported for LoRA training (and in some cases full fine-tuning). Example configs are in `config/examples/`. Models without a listed HuggingFace path or example config may require manual setup or are still experimental. All models use `job: extension` with `type: "sd_trainer"` in the config. For newer models, arch flags and any version-specific notes are documented in that model's README under `extensions_built_in/diffusion_models/<model>/`.

### Image

| Model | HuggingFace Path | Config `arch` / flags | Example Config | Min VRAM |
|---|---|---|---|---|
| FLUX.1-dev | `black-forest-labs/FLUX.1-dev` | `is_flux: true` | `train_lora_flux_24gb.yaml` | 24GB |
| FLUX.1-schnell | `black-forest-labs/FLUX.1-schnell` | `is_flux: true` + `assistant_lora_path` | `train_lora_flux_schnell_24gb.yaml` | 24GB |
| FLUX.2-dev | `black-forest-labs/FLUX.2-dev` | `arch: flux2` | — | 24GB |
| FLUX.2-Klein-4B | `black-forest-labs/FLUX.2-klein-base-4B` | `arch: flux2_klein_4b` | — | 24GB |
| FLUX.2-Klein-9B | `black-forest-labs/FLUX.2-klein-base-9B` | `arch: flux2_klein_9b` | — | 24GB |
| Flex.1-alpha | `ostris/Flex.1-alpha` | `is_flux: true` | `train_lora_flex_24gb.yaml` | 24GB |
| Flex.2-preview | `ostris/Flex.2-preview` | `arch: flex2` | `train_lora_flex2_24gb.yaml` | 24GB |
| Stable Diffusion 3.5 Large | `stabilityai/stable-diffusion-3.5-large` | `is_v3: true` | `train_lora_sd35_large_24gb.yaml` | 24GB |
| Lumina Image 2.0 | `Alpha-VLLM/Lumina-Image-2.0` | `is_lumina2: true` | `train_lora_lumina.yaml` | 20GB |
| Chroma | `lodestones/Chroma1-Base` | `arch: chroma` | `train_lora_chroma_24gb.yaml` | 24GB |
| Chroma Radiance | — | `arch: chroma_radiance` | — | 24GB |
| Qwen-Image | `Qwen/Qwen-Image` | `arch: qwen_image` | `train_lora_qwen_image_24gb.yaml` | 24GB |
| Qwen-Image-2512 | `Qwen/Qwen-Image-2512` | `arch: qwen_image` | — | 24GB |
| Qwen-Image-2.1 | `Qwen/Qwen-Image-2.1` | `arch: qwen_image_2` | — | 24GB |
| HiDream-I1-Full | `HiDream-ai/HiDream-I1-Full` | `arch: hidream` | `train_lora_hidream_48.yaml` | 48GB |
| HiDream-O1 | `HiDream-ai/HiDream-O1-Image` | `arch: hidream_o1` | — | 48GB |
| OmniGen2 | `OmniGen2/OmniGen2` | `arch: omnigen2` | `train_lora_omnigen2_24gb.yaml` | 24GB |
| Z-Image | `Tongyi-MAI/Z-Image` | `arch: zimage` | — | 24GB |
| Z-Image Turbo | `Tongyi-MAI/Z-Image-Turbo` | `arch: zimage` + assistant adapter | `train_lora_zimage_turbo_style.yaml` | 24GB |
| Z-Image De-Turbo | `ostris/Z-Image-De-Turbo` | `arch: zimage` | — | 24GB |
| Krea 2 | `krea/Krea-2-Raw` | `arch: krea2` | — | 24GB |
| ERNIE-Image | `baidu/ERNIE-Image` | `arch: ernie_image` | — | — |
| Nucleus-Image | `NucleusAI/Nucleus-Image` | `arch: nucleus_image` | — | — |
| PRXPixel | `Photoroom/prxpixel-t2i` | `arch: prx_pixel` | — | — |
| CogView4 | — | `arch: cogview4` | — | — |
| F-Lite | — | `arch: f` | — | — |
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` | `is_xl: true` | — | 12GB |
| Stable Diffusion 1.5 | `stable-diffusion-v1-5/stable-diffusion-v1-5` | (default) | — | 8GB |
| Stable Diffusion 2.x | — | `is_v2: true` | — | 8GB |
| Boogu Image 0.1 | `Boogu/Boogu-Image-0.1-Base` | `arch: boogu_image` | — | — |
| Anima | `circlestone-labs/Anima-Base-v1.0-Diffusers` | `arch: anima` | — | — |
| Z-Image L2P | `zhen-nan/L2P` | `arch: zimage_l2p` | — | — |
| Krea 2 Turbo | `krea/Krea-2-Turbo` | `arch: krea2` | — | 24GB |
| Ideogram 4 FP8 | `ideogram-ai/ideogram-4-fp8` | `arch: ideogram4` | — | — |
| Mage-Flow | `microsoft/Mage-Flow-Base` | `arch: mageflow` | — | — |

### Instruction / Edit

| Model | HuggingFace Path | Config `arch` / flags | Example Config | Min VRAM |
|---|---|---|---|---|
| FLUX.1-Kontext-dev | `black-forest-labs/FLUX.1-Kontext-dev` | `arch: flux_kontext` | `train_lora_flux_kontext_24gb.yaml` | 24GB |
| Qwen-Image-Edit | `Qwen/Qwen-Image-Edit` | `arch: qwen_image_edit` | `train_lora_qwen_image_edit_32gb.yaml` | 32GB |
| Qwen-Image-Edit-2509 | `Qwen/Qwen-Image-Edit-2509` | `arch: qwen_image_edit_plus` | `train_lora_qwen_image_edit_2509_32gb.yaml` | 32GB |
| Qwen-Image-Edit-2511 | `Qwen/Qwen-Image-Edit-2511` | `arch: qwen_image_edit_plus:2511` | — | 32GB |
| Qwen-Image-2.1 | `Qwen/Qwen-Image-2.1` | `arch: qwen_image_2` | — | 24GB |
| HiDream-E1-1 | `HiDream-ai/HiDream-E1-1` | `arch: hidream_e1` | — | 48GB |
| Boogu Image 0.1 | `Boogu/Boogu-Image-0.1-Base` | `arch: boogu_image` | — | 24GB |
| Boogu Image Edit | `Boogu/Boogu-Image-0.1-Edit` | `arch: boogu_image_edit` | — | 24GB |
| Krea 2 (edit training) | `krea/Krea-2-Raw` | `arch: krea2` | — | 24GB |
| Krea 2 Turbo (edit training) | `krea/Krea-2-Turbo` | `arch: krea2` | — | 24GB |
| Mage-Flow Edit | `microsoft/Mage-Flow-Edit-Base` | `arch: mageflow_edit` | — | — |

### Video

| Model | HuggingFace Path | Config `arch` / flags | Example Config | Min VRAM |
|---|---|---|---|---|
| Wan 2.1 T2V 1.3B | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `arch: wan21` | `train_lora_wan21_1b_24gb.yaml` | 24GB |
| Wan 2.1 T2V 14B | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | `arch: wan21` | `train_lora_wan21_14b_24gb.yaml` | 24GB |
| Wan 2.1 I2V 14B-480P | `Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` | `arch: wan21_i2v` | — | 24GB |
| Wan 2.1 I2V 14B-720P | `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` | `arch: wan21_i2v` | — | 24GB |
| Wan 2.2 T2V 14B | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `arch: wan22_14b` | `train_lora_wan22_14b_24gb.yaml` | 24GB |
| Wan 2.2 I2V 14B | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | `arch: wan22_14b_i2v` | — | 24GB |
| Wan 2.2 TI2V 5B | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | `arch: wan22_5b` | — | 24GB |
| LTX-2 | `Lightricks/LTX-2` | `arch: ltx2` | — | — |
| LTX-2.3 | `Lightricks/LTX-2.3` | `arch: ltx2` | — | — |
| LTX-2.5 | `Lightricks/LTX-2.5` | `arch: ltx2` | — | — |
| MiniMax-H3 | `Comfy-Org/MiniMax-H3` [^h3] | `arch: minimax_h3` (also `minimax_h3_ref2va`, `minimax_h3_vsa`) | — | — |

[^h3]: Use the **Comfy-Org repack**, not `MiniMaxAI/MiniMax-H3`. A hub-style
`name_or_path` is treated as a replacement weights repo, and the original
MiniMax repo does not contain the repacked transformer/text-encoder/VAE files
the loader asks for — pointing at it fails with a 404 while the model is
loading. (The original repo is still used for tokenizer/processor config; that
path is resolved internally and needs no configuration.) The weights are
already int8-ConvRot + nvfp4 quantized, so leave `quantize`/`quantize_te` off.

### Audio

| Model | HuggingFace Path | Config `arch` / flags | Example Config | Min VRAM |
|---|---|---|---|---|
| Ace Step 1.5 | `ACE-Step/Ace-Step1.5` | `arch: ace_step_15` | — | — |
| Ace Step 1.5 XL | `ACE-Step/acestep-v15-xl-base` | `arch: ace_step_15_xl` | — | — |
| YuE2 | `m-a-p/YuE2-3B` | `arch: yue2` | — | — |

YuE2 training uses the community tokenizer by Kytra ([@sin_ceriously](https://x.com/sin_ceriously)), [`Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4`](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4), because the official audio-to-token encoder is unreleased.

### LLM

| Model | HuggingFace Path | Config `arch` / flags | Example Config | Min VRAM |
|---|---|---|---|---|
| Qwen2.5-Omni | `Qwen/Qwen2.5-Omni-7B` | — | — | — |

### Experimental

| Model | HuggingFace Path | Config `arch` / flags | Example Config | Min VRAM |
|---|---|---|---|---|
| Zeta Chroma | `lodestones/Zeta-Chroma` | `arch: zeta_chroma` | — | — |

## Installation

### Install with the AI Toolkit Manager (experimental)

The recommended way to install and run AI Toolkit is with the **AI Toolkit
Manager**, built into this repo. The manager detects your hardware and sets up
the right PyTorch build, creates the python environment, and grabs local copies
of Node.js and FFmpeg — everything stays inside the ai-toolkit folder, nothing
is installed system-wide. On every launch the manager checks for updates and
applies them (your local changes are never overwritten — if you have modified
files, the update is skipped with a warning), then starts the UI at
`http://localhost:8675`.

The manager is still **experimental** — please let me know if you have any
issues with it. The manual instructions below still work if you prefer them
or run into problems.

The only requirement is **git** (on Windows the manager can even fetch a
portable git for updates, but you need one installed to clone the repo first).

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
```

Then start the manager with the script for your platform:

Linux:
```bash
chmod +x run_linux.sh
./run_linux.sh
```

MacOS (Apple Silicon, experimental):
```bash
chmod +x run_mac.zsh
./run_mac.zsh
```

Windows: double-click `run_windows.bat` (or run it from a terminal).

You can also use the manager directly from a terminal (handy on headless
servers):

```bash
python3 -m manager install   # first-time setup
python3 -m manager update    # pull updates + sync dependencies
python3 -m manager launch    # start the UI
python3 -m manager doctor    # diagnose problems
```

### Manual installation

Requirements:
- python >=3.10 (3.12 recommended)
- Nvidia GPU with enough ram to do what you need
- python venv
- git


Linux:
```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python3 -m venv venv
source venv/bin/activate
# install torch first
pip3 install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip3 install -r requirements.txt
```

For devices running **DGX OS** (including DGX Spark), follow [these](dgx_instructions.md) instructions.


Windows:

If you are having issues with Windows. I recommend using the easy install script at [https://github.com/Tavris1/AI-Toolkit-Easy-Install](https://github.com/Tavris1/AI-Toolkit-Easy-Install)

```bash
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python -m venv venv
.\venv\Scripts\activate
pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```


# AI Toolkit UI

<img src="https://ostris.com/wp-content/uploads/2025/02/toolkit-ui.jpg" alt="AI Toolkit UI" width="100%">

The AI Toolkit UI is a web interface for the AI Toolkit. It allows you to easily start, stop, and monitor jobs. It also allows you to easily train models with a few clicks. It also allows you to set a token for the UI to prevent unauthorized access so it is mostly safe to run on an exposed server.

## Running the UI

Requirements:
- Node.js > 20

The UI does not need to be kept running for the jobs to run. It is only needed to start/stop/monitor jobs. The commands below
will install / update the UI and it's dependencies and start the UI. 

```bash
cd ui
npm run build_and_start
```

You can now access the UI at `http://localhost:8675` or `http://<your-ip>:8675` if you are running it on a server.

## Securing the UI

If you are hosting the UI on a cloud provider or any network that is not secure, I highly recommend securing it with an auth token. 
You can do this by setting the environment variable `AI_TOOLKIT_AUTH` to super secure password. This token will be required to access
the UI. You can set this when starting the UI like so:

```bash
# Linux
AI_TOOLKIT_AUTH=super_secure_password npm run build_and_start

# Windows
set AI_TOOLKIT_AUTH=super_secure_password && npm run build_and_start

# Windows Powershell
$env:AI_TOOLKIT_AUTH="super_secure_password"; npm run build_and_start
```


## FLUX.1 Training

### Tutorial

To get started quickly, check out [@araminta_k](https://x.com/araminta_k) tutorial on [Finetuning Flux Dev on a 3090](https://www.youtube.com/watch?v=HzGW_Kyermg) with 24GB VRAM.


### Requirements
You currently need a GPU with **at least 24GB of VRAM** to train FLUX.1. If you are using it as your GPU to control 
your monitors, you probably need to set the flag `low_vram: true` in the config file under `model:`. This will quantize
the model on CPU and should allow it to train with monitors attached. Users have gotten it to work on Windows with WSL,
but there are some reports of a bug when running on windows natively. 
I have only tested on linux for now. This is still extremely experimental
and a lot of quantizing and tricks had to happen to get it to fit on 24GB at all. 

### FLUX.1-dev

FLUX.1-dev has a non-commercial license. Which means anything you train will inherit the
non-commercial license. It is also a gated model, so you need to accept the license on HF before using it.
Otherwise, this will fail. Here are the required steps to setup a license.

1. Sign into HF and accept the model access here [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
2. Make a file named `.env` in the root on this folder
3. [Get a READ key from huggingface](https://huggingface.co/settings/tokens/new?) and add it to the `.env` file like so `HF_TOKEN=your_key_here`

### FLUX.1-schnell

FLUX.1-schnell is Apache 2.0. Anything trained on it can be licensed however you want and it does not require a HF_TOKEN to train.
However, it does require a special adapter to train with it, [ostris/FLUX.1-schnell-training-adapter](https://huggingface.co/ostris/FLUX.1-schnell-training-adapter).
It is also highly experimental. For best overall quality, training on FLUX.1-dev is recommended.

To use it, You just need to add the assistant to the `model` section of your config file like so:

```yaml
      model:
        name_or_path: "black-forest-labs/FLUX.1-schnell"
        assistant_lora_path: "ostris/FLUX.1-schnell-training-adapter"
        is_flux: true
        quantize: true
```

You also need to adjust your sample steps since schnell does not require as many

```yaml
      sample:
        guidance_scale: 1  # schnell does not do guidance
        sample_steps: 4  # 1 - 4 works well
```

### Training
1. Copy the example config file located at `config/examples/train_lora_flux_24gb.yaml` (`config/examples/train_lora_flux_schnell_24gb.yaml` for schnell) to the `config` folder and rename it to `whatever_you_want.yml`
2. Edit the file following the comments in the file
3. Run the file like so `python run.py config/whatever_you_want.yml`

A folder with the name and the training folder from the config file will be created when you start. It will have all 
checkpoints and images in it. You can stop the training at any time using ctrl+c and when you resume, it will pick back up
from the last checkpoint.

IMPORTANT. If you press crtl+c while it is saving, it will likely corrupt that checkpoint. So wait until it is done saving

### Need help?

Please do not open a bug report unless it is a bug in the code. You are welcome to [Join my Discord](https://discord.gg/VXmU2f5WEU)
and ask for help there. However, please refrain from PMing me directly with general question or support. Ask in the discord
and I will answer when I can.

## Ostris Cloud

You can use many cloud providers to rent GPUs. If you want to help support this project in the largest way possible, please consider using [Ostris Cloud](https://cloud.ostris.com). Ostris Cloud is owned and operated by me, Ostris, and every dollar earned goes directly back into funding the development of this project.

<a href="https://cloud.ostris.com" target="_blank"><img src="https://cloud.ostris.com/api/og" alt="Ostris Cloud" style="max-width:100%;width:600px;height:auto;"></a>


## Training in RunPod
If you would like to use Runpod, but have not signed up yet, please consider using [my Runpod affiliate link](https://runpod.io?ref=h0y9jyr2) to help support this project.


I maintain an official Runpod Pod template here which can be accessed [here](https://console.runpod.io/deploy?template=0fqzfjy6f3&ref=h0y9jyr2).

I have also created a short video showing how to get started using AI Toolkit with Runpod [here](https://youtu.be/HBNeS-F6Zz8).

## Training in Modal

### 1. Setup
#### ai-toolkit:
```
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
python -m venv venv
source venv/bin/activate
pip install torch
pip install -r requirements.txt
pip install --upgrade accelerate transformers diffusers huggingface_hub #Optional, run it if you run into issues
```
#### Modal:
- Run `pip install modal` to install the modal Python package.
- Run `modal setup` to authenticate (if this doesn’t work, try `python -m modal setup`).

#### Hugging Face:
- Get a READ token from [here](https://huggingface.co/settings/tokens) and request access to Flux.1-dev model from [here](https://huggingface.co/black-forest-labs/FLUX.1-dev).
- Run `huggingface-cli login` and paste your token.

### 2. Upload your dataset
- Drag and drop your dataset folder containing the .jpg, .jpeg, or .png images and .txt files in `ai-toolkit`.

### 3. Configs
- Copy an example config file located at ```config/examples/modal``` to the `config` folder and rename it to ```whatever_you_want.yml```.
- Edit the config following the comments in the file, **<ins>be careful and follow the example `/root/ai-toolkit` paths</ins>**.

### 4. Edit run_modal.py
- Set your entire local `ai-toolkit` path at `code_mount = modal.Mount.from_local_dir` like:
  
   ```
   code_mount = modal.Mount.from_local_dir("/Users/username/ai-toolkit", remote_path="/root/ai-toolkit")
   ```
- Choose a `GPU` and `Timeout` in `@app.function` _(default is A100 40GB and 2 hour timeout)_.

### 5. Training
- Run the config file in your terminal: `modal run run_modal.py --config-file-list-str=/root/ai-toolkit/config/whatever_you_want.yml`.
- You can monitor your training in your local terminal, or on [modal.com](https://modal.com/).
- Models, samples and optimizer will be stored in `Storage > flux-lora-models`.

### 6. Saving the model
- Check contents of the volume by running `modal volume ls flux-lora-models`. 
- Download the content by running `modal volume get flux-lora-models your-model-name`.
- Example: `modal volume get flux-lora-models my_first_flux_lora_v1`.

### Screenshot from Modal

<img width="1728" alt="Modal Traning Screenshot" src="https://github.com/user-attachments/assets/7497eb38-0090-49d6-8ad9-9c8ea7b5388b">

---

## Dataset Preparation

Datasets generally need to be a folder containing images and associated text files. Currently, the only supported
formats are jpg, jpeg, and png. Webp currently has issues. The text files should be named the same as the images
but with a `.txt` extension. For example `image2.jpg` and `image2.txt`. The text file should contain only the caption.
You can add the word `[trigger]` in the caption file and if you have `trigger_word` in your config, it will be automatically
replaced. 

Images are never upscaled but they are downscaled and placed in buckets for batching. **You do not need to crop/resize your images**.
The loader will automatically resize them and can handle varying aspect ratios. 


## Training Specific Layers

To train specific layers with LoRA, you can use the `only_if_contains` network kwargs. For instance, if you want to train only the 2 layers
used by The Last Ben, [mentioned in this post](https://x.com/__TheBen/status/1829554120270987740), you can adjust your
network kwargs like so:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks.7.proj_out"
            - "transformer.single_transformer_blocks.20.proj_out"
```

The naming conventions of the layers are in diffusers format, so checking the state dict of a model will reveal 
the suffix of the name of the layers you want to train. You can also use this method to only train specific groups of weights.
For instance to only train the `single_transformer` for FLUX.1, you can use the following:

```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          only_if_contains:
            - "transformer.single_transformer_blocks."
```

You can also exclude layers by their names by using `ignore_if_contains` network kwarg. So to exclude all the single transformer blocks,


```yaml
      network:
        type: "lora"
        linear: 128
        linear_alpha: 128
        network_kwargs:
          ignore_if_contains:
            - "transformer.single_transformer_blocks."
```

`ignore_if_contains` takes priority over `only_if_contains`. So if a weight is covered by both,
if will be ignored.

## LoKr Training

To learn more about LoKr, read more about it at [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md). To train a LoKr model, you can adjust the network type in the config file like so:

```yaml
      network:
        type: "lokr"
        lokr_full_rank: true
        lokr_factor: 8
```

Everything else should work the same including layer targeting.


## Support My Work

If you enjoy my projects or use them commercially, please consider sponsoring me. Every bit helps! 💖

<a href="https://ostris.com/sponsors" target="_blank"><img src="https://ostris.com/wp-content/uploads/2025/05/support-banner2.png" alt="Support my work" style="max-width:100%;height:auto;"></a>

### Current Sponsors

All of these people / organizations are the ones who selflessly make this project possible. Thank you!!

<a href="https://ostris.com/sponsors"><img src="https://ostris.com/sponsors.svg" alt="Sponsors" style="width:100%;height:auto;"></a>
