# ChatR Labs Uncensored Omni LoRA v0.1 — for Qwen-Image-2.1

[![Try the Playground](https://img.shields.io/badge/Playground-Try_it_live-2563eb?style=for-the-badge)](http://108.204.92.18:48008/)
[![Download LoRA](https://img.shields.io/badge/GitHub-Download_LoRA-ffcc4d?style=for-the-badge)](https://github.com/chatrlabs/Qwen-Image-2.1-Uncensored-Omni-LoRA/releases/download/v0.1/chatrlabs-omni-lora-v0.1-bf16.safetensors)

**One LoRA for text-to-image generation and image editing. Built with Qwen.**

ChatR Labs Uncensored Omni LoRA v0.1 is an experimental rank-64 adapter fine-tuned from [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1). Create images from text or transform existing images with natural-language instructions, using one adapter for both workflows.

The download contains **LoRA weights only**. Load it on top of the original Qwen-Image-2.1 base model; the transformer, text encoder, processor, and VAE are downloaded separately from the base repository.

## Try it online

**[Open the Playground →](http://108.204.92.18:48008/)**

No account is required. Choose **Text to Image** or **Image Editing**, enter a prompt, and generate. The editing interface accepts one to three reference images. Requests share a queue.

## Download

| File | Format | Size | Use |
| --- | --- | ---: | --- |
| [**chatrlabs-omni-lora-v0.1-bf16.safetensors**](https://github.com/chatrlabs/Qwen-Image-2.1-Uncensored-Omni-LoRA/releases/download/v0.1/chatrlabs-omni-lora-v0.1-bf16.safetensors) | BF16, Diffusers/PEFT keys | 336 MB | Text-to-image and image editing |

Download the weights from the [v0.1 GitHub Release](https://github.com/chatrlabs/Qwen-Image-2.1-Uncensored-Omni-LoRA/releases/tag/v0.1) or the [Hugging Face model repository](https://huggingface.co/chatrlabs/Qwen-Image-2.1-Uncensored-Omni-LoRA-v0.1). The Git repository contains documentation and inference code; the weight file is a release attachment.

File integrity information is in [SHA256SUMS](./SHA256SUMS). This release contains a BF16 adapter, without merged base weights or GGUF/FP8 variants.

## Recommended settings

| Setting | Starting value |
| --- | --- |
| Base model | `Qwen/Qwen-Image-2.1` |
| LoRA strength | `1.0` |
| Sampling steps | `25` |
| True CFG scale | `4.0` |
| Negative prompt | A single space: `" "` |
| Output resolution | Approximately 1 megapixel; e.g. `1024 × 1024` |
| KV cache | Enabled |

These are the settings used to evaluate **this adapter**. In the tested Diffusers pipeline, an omitted or empty negative prompt disables true CFG even when `true_cfg_scale` is greater than 1. Use a single space if you do not need a specific negative prompt.

The adapter is trained for Qwen-Image-2.1. Earlier Qwen-Image, Qwen-Image-2512, and Qwen-Image-Edit-2511 have different architectures and are not compatible bases. This is a regular LoRA; it does not provide few-step distillation.

## Usage with Diffusers

### 1. Install

Install a CUDA-compatible PyTorch build for your system, then install the dependencies in [requirements.txt](./requirements.txt):

```bash
git clone https://github.com/chatrlabs/Qwen-Image-2.1-Uncensored-Omni-LoRA.git
cd Qwen-Image-2.1-Uncensored-Omni-LoRA
git checkout v0.1
pip install -r requirements.txt
```

While the repositories are private, GitHub access is required for cloning and release downloads, and Hugging Face access is required when loading the adapter from the Hub. To use the GitHub download locally:

```bash
gh release download v0.1 --repo chatrlabs/Qwen-Image-2.1-Uncensored-Omni-LoRA --pattern '*.safetensors'
sha256sum --check --ignore-missing SHA256SUMS
```

The requirements pin the Diffusers revision used by the Playground, including `QwenImage21Pipeline`, together with its tested Transformers, PEFT, and Accelerate versions. The deployment uses PyTorch 2.11.0 with CUDA 12.8.

### 2. Load the base model and adapter

```python
import torch
from diffusers import QwenImage21Pipeline

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1",
    torch_dtype=torch.bfloat16,
)
pipe.load_lora_weights(
    ".",  # Directory containing the downloaded release weight file.
    weight_name="chatrlabs-omni-lora-v0.1-bf16.safetensors",
    adapter_name="omni",
)
pipe.set_adapters("omni", adapter_weights=1.0)

# CPU offload reduces GPU residency; it needs sufficient system RAM.
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
# With sufficient VRAM, replace CPU offload with pipe.to("cuda").
```

### 3. Text to image

```python
image = pipe(
    prompt="A blue ceramic mug on a wooden table by a window, soft morning light, product photography",
    negative_prompt=" ",
    width=1024,
    height=1024,
    num_inference_steps=25,
    true_cfg_scale=4.0,
    use_kv_cache=True,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("generated.png")
```

### 4. Image editing

Use the same loaded pipeline and pass reference images alongside the instruction:

```python
from PIL import Image

reference = Image.open("generated.png").convert("RGBA")
image = pipe(
    image=[reference],
    prompt="Change the mug to green, keep everything else unchanged",
    negative_prompt=" ",
    width=1024,
    height=1024,
    output_resolution=1024,
    num_inference_steps=25,
    true_cfg_scale=4.0,
    use_kv_cache=True,
    generator=torch.Generator("cuda").manual_seed(45),
).images[0]
image.save("edited.png")
```

For multiple references, pass a list of PIL images. The Playground has been exercised with one and three references. More references increase memory requirements.

A command-line version of these examples is available in [inference.py](./inference.py):

```bash
python inference.py --lora-file chatrlabs-omni-lora-v0.1-bf16.safetensors --prompt "A blue ceramic mug on a wooden table" --output generated.png
python inference.py --lora-file chatrlabs-omni-lora-v0.1-bf16.safetensors --image generated.png --prompt "Change the mug to green" --output edited.png
```

Omit `--lora-file` to download the adapter from its Hugging Face repository instead.

## Memory and performance

The small adapter download does not reflect the full base model's memory requirements. Use CPU offload or a suitable multi-GPU setup when the complete BF16 pipeline does not fit on one GPU.

Our Playground uses **two RTX 4090 24 GB GPUs** with the BF16 pipeline split across them. Both GPUs serve one request together; simultaneous requests are queued. Observed server-side times after model loading, at 25 steps and CFG 4.0:

| Task | Output size | Time |
| --- | --- | ---: |
| Text to image | 1344 × 768 | 33.2 s |
| Editing with one reference | 1024 × 1024 | 38.4 s |
| Editing with three reference inputs | 1024 × 1024 | 50.5 s |

These are individual deployment checks, not comparative quality benchmarks. The three-reference check used the same source image in all three slots. Timing depends on the prompt, input images, hardware, and execution path; the CPU-offload example above does not reproduce this custom GPU split.

## Checkpoint details

| Property | Value |
| --- | --- |
| Base | Qwen-Image-2.1 |
| Training checkpoint | Epoch 3, step 9,843 |
| Training items recorded in checkpoint | 104,933 |
| Training resolution | 1024 × 1024, aspect-ratio bucketing enabled |
| Rank / original alpha | 64 / 64 |
| Adapted modules | 224 modules across 32 transformer blocks |
| Targets | Attention `to_q`, `to_k`, `to_v`, `to_out.0`; MLP `proj`, `gate_layer`, `out` |
| Published tensors | 448, BF16 |
| Training module | `networks.lora_qwen_image_21` |
| Export | Kohya-style FP32 weights converted to Diffusers/PEFT keys and BF16; alpha/rank folded into `lora_B` |

The training dataset is not included in this release. Checkpoint statistics above come from the embedded training metadata. We are not publishing an independent quality or content-coverage benchmark with v0.1.

## About “uncensored”

The adapter and example inference code do not add a prompt filter or output safety checker. The name describes the intended release, rather than a guarantee that every requested concept will be rendered. Results depend on the base model, adapter, prompt, and the application running them.

### Acknowledgements

Training and evaluation compute for Omni v0.1 was generously provided by [Chatr AI](https://thechatr.ai).

Special thanks to the Chatr AI team for supporting this release with GPU infrastructure.

## Source and license

- **Released by:** [ChatR Labs](https://huggingface.co/chatrlabs)
- **Base model:** [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
- **Base revision used in the Playground:** `790c92633540aa0cb11d9abf19eb46d861714758`
- **Inference:** [Hugging Face Diffusers](https://github.com/huggingface/diffusers), revision `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`
- **Modification:** ChatR Labs fine-tuned a rank-64 adapter and exported the epoch-3 checkpoint to BF16 Diffusers/PEFT format. This repository distributes the adapter separately from the base model.
- **License:** [Qwen Research License Agreement](./LICENSE), with attribution in [Notice](./Notice). Use is limited to non-commercial research and evaluation unless separately licensed by the upstream rights holder.
- **Checksums:** [SHA256SUMS](./SHA256SUMS)

**Built with Qwen.** This is a community fine-tune by ChatR Labs, independent of the upstream Qwen team.
