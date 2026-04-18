"""
Feed (init_frame, hand_frames) from a rollout manifest to Wan2.2 + robot LoRA,
save the imagined future video per rollout.

Input manifest format (from render_action_condition.py):
  [
    {"init_state": "...", "sample": 0,
     "init_frame": "<rel_path>/init_frame.png",
     "hand_frames": "<rel_path>/sample00/hand_frames.mp4",
     ...},
    ...
  ]

Output:
  For each entry, writes <abs sample_dir>/imagined.mp4 (17 frames @ 15fps)
  and adds key "imagined_video" to the manifest (saved as new file).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/hongyi/PWM/external/DiffSynth-Studio")
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from peft import LoraConfig, inject_adapter_in_model

DIT_SHARDS = [
    "/home/hongyi/PWM/external/DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
    "/home/hongyi/PWM/external/DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
    "/home/hongyi/PWM/external/DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors",
]
T5_PATH = "/home/hongyi/PWM/external/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
VAE_PATH = "/home/hongyi/PWM/external/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
TOKENIZER_ID = "Wan-AI/Wan2.1-T2V-1.3B"

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def load_pipe(device="cuda"):
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device=device,
        model_configs=[
            ModelConfig(path=DIT_SHARDS),
            ModelConfig(path=T5_PATH),
            ModelConfig(path=VAE_PATH),
        ],
        tokenizer_config=ModelConfig(
            model_id=TOKENIZER_ID,
            origin_file_pattern="google/umt5-xxl/",
        ),
    )
    return pipe


def apply_lora(pipe, lora_path, rank=32, alpha=1.0):
    lora_config = LoraConfig(
        r=rank, lora_alpha=rank,
        target_modules=["q", "k", "v", "o", "ffn.0", "ffn.2"],
    )
    pipe.dit = inject_adapter_in_model(lora_config, pipe.dit)
    from diffsynth.core import load_state_dict
    state_dict = load_state_dict(lora_path, torch_dtype=torch.bfloat16, device="cpu")
    mapped = {}
    for k, v in state_dict.items():
        k2 = k.replace("lora_A.weight", "lora_A.default.weight") \
              .replace("lora_B.weight", "lora_B.default.weight")
        mapped[k2] = v * alpha
    missing, unexpected = pipe.dit.load_state_dict(mapped, strict=False)
    print(f"LoRA loaded: missing={len(missing)} unexpected={len(unexpected)}")
    pipe.dit = pipe.dit.to(dtype=torch.bfloat16, device=pipe.device)


def load_hand_frames_video(path, num_frames, width, height):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb).resize((width, height)))
        if len(frames) >= num_frames:
            break
    cap.release()
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Path to manifest.json from rollout stage.")
    p.add_argument("--lora", required=True)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--prompt", default="A humanoid robot opens a drawer with its dexterous hands.")
    p.add_argument("--num_frames", type=int, default=17, help="Including init frame (16 pred + 1 init).")
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--output_manifest", default=None,
                   help="Where to save updated manifest. Default: <manifest>.with_imagined.json")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)
    base_dir = manifest_path.parent
    out_manifest = args.output_manifest or str(manifest_path.with_suffix(".with_imagined.json"))

    print(f"Loading base Wan2.2 pipe ...", flush=True)
    pipe = load_pipe(device="cuda")
    print(f"Applying LoRA: {args.lora}", flush=True)
    apply_lora(pipe, args.lora, rank=args.lora_rank)

    n_hand_frames = args.num_frames - 1
    for i, entry in enumerate(manifest):
        init_path = base_dir / entry["init_frame"]
        hand_path = base_dir / entry["hand_frames"]
        sample_dir = hand_path.parent
        imagined_path = sample_dir / "imagined.mp4"

        t0 = time.time()
        init_img = Image.open(init_path).convert("RGB").resize((args.width, args.height))
        hand_frames = load_hand_frames_video(str(hand_path), n_hand_frames, args.width, args.height)
        if len(hand_frames) < n_hand_frames:
            print(f"[{i+1}/{len(manifest)}] SKIP: hand_frames has only {len(hand_frames)}/{n_hand_frames}", flush=True)
            continue

        with torch.no_grad():
            video = pipe(
                prompt=args.prompt,
                negative_prompt=NEGATIVE_PROMPT,
                input_image=init_img,
                height=args.height, width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                tiled=True,
                pred_frames=hand_frames,
            )
        save_video(video, str(imagined_path), fps=args.fps, quality=5)
        entry["imagined_video"] = str(imagined_path.relative_to(base_dir))
        dt = time.time() - t0
        print(f"[{i+1}/{len(manifest)}] {imagined_path.name} ({dt:.1f}s)", flush=True)

    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. Updated manifest: {out_manifest}")


if __name__ == "__main__":
    main()
