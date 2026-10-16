"""Text-to-image and reference-image editing with ChatR Labs Omni LoRA v0.1."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=Path("output.png"))
    parser.add_argument("--lora-file", type=Path,
                        help="Use a local release weight file instead of downloading from Hugging Face.")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--negative", default=" ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["offload", "cuda"], default="offload",
                        help="CPU offload by default; cuda keeps the whole pipeline on one GPU.")
    args = parser.parse_args()
    if args.lora_file is not None and not args.lora_file.is_file():
        parser.error(f"LoRA file not found: {args.lora_file}")

    import torch
    from diffusers import QwenImage21Pipeline
    from PIL import Image, ImageOps

    pipe = QwenImage21Pipeline.from_pretrained(
        "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16,
    )
    pipe.load_lora_weights(
        str(args.lora_file.resolve().parent) if args.lora_file else
        "chatrlabs/Qwen-Image-2.1-Uncensored-Omni-LoRA-v0.1",
        weight_name=args.lora_file.name if args.lora_file else
        "chatrlabs-omni-lora-v0.1-bf16.safetensors",
        adapter_name="omni",
    )
    pipe.set_adapters("omni", adapter_weights=args.strength)
    if args.device == "offload":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    # Keep VAE tiling off: tiled decoding adds visible vertical colour streaks.

    references = []
    for path in args.image:
        with Image.open(path) as image:
            references.append(ImageOps.exif_transpose(image).convert("RGBA"))
    kwargs = dict(
        prompt=args.prompt,
        negative_prompt=args.negative if args.negative.strip() else " ",
        width=args.width,
        height=args.height,
        output_resolution=1024,
        num_inference_steps=args.steps,
        true_cfg_scale=args.cfg,
        use_kv_cache=True,
        generator=torch.Generator("cuda").manual_seed(args.seed),
    )
    if references:
        kwargs["image"] = references
    with torch.inference_mode():
        image = pipe(**kwargs).images[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
