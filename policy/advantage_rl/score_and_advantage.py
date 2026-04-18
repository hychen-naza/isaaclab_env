"""Score sim rollout videos with Robometer, compute per-init advantages, filter.

For each entry in the manifest (from chain_wm_rollout.py):
  - Score rollout.mp4 → per-frame progress + success
  - mean_progress = mean over frames
  - final_success = last frame's success probability

Advantage = sample_mean_progress - baseline,
where baseline per init_i = mean progress across the init's K stochastic samples.
(RISE-style mean-normalization → advantage has zero mean per init.)

Filter keeps entries with advantage > --advantage_threshold.

Usage:
  python policy/advantage_rl/score_and_advantage.py \\
      --manifest <rollout_dir>/manifest.json \\
      --task_prompt "Dexterous robot hand grasp the bottle and lift it ..."
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent.resolve()
_ROBOMETER_DIR = _HERE.parent / "robometer"            # policy/robometer
_ROBOMETER_SCRIPTS = _ROBOMETER_DIR / "scripts"
# Make robometer package + its local inference helper importable.
if str(_ROBOMETER_DIR) not in sys.path:
    sys.path.insert(0, str(_ROBOMETER_DIR))
if str(_ROBOMETER_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_ROBOMETER_SCRIPTS))


DEFAULT_MODEL_PATH = "aliangdw/qwen4b_pref_prog_succ_8_frames_all_part2"
DEFAULT_TASK = (
    "Dexterous robot hand open the fingers and approach the bottle and then "
    "close fingers to grasp the bottle; next the dexterous hand lift it and "
    "move to the bowl and tilt the bottle to pour"
)


def _load_robometer(model_path: str, device):
    """Load robometer model, tokenizer, processor, batch collator — once."""
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator

    exp_config, tokenizer, processor, reward_model = load_model_from_hf(
        model_path=model_path, device=device,
    )
    reward_model.eval()
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
    return {
        "exp_config": exp_config, "tokenizer": tokenizer, "processor": processor,
        "reward_model": reward_model, "batch_collator": batch_collator, "device": device,
    }


def _score_video(video_path: str, task: str, state, fps: float, max_frames: int,
                 max_side: int = 256):
    """Score one video. Returns (progress: (T,), success: (T,)) per-frame arrays."""
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs
    from robometer.evals.eval_viz_utils import extract_frames

    frames = extract_frames(video_path, fps=fps, max_frames=max_frames)
    if frames is None or frames.size == 0:
        raise RuntimeError(f"Could not extract frames from {video_path}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.ndim == 4 and frames.shape[1] in (1, 3) and frames.shape[-1] not in (1, 3):
        frames = frames.transpose(0, 2, 3, 1)

    # Resize so the longest side is at most `max_side` — keeps vision-token count
    # manageable for high-res sim renders (our rollouts are 1280×1920).
    if max_side is not None and max_side > 0:
        import cv2
        H, W = int(frames.shape[1]), int(frames.shape[2])
        longest = max(H, W)
        if longest > max_side:
            scale = max_side / longest
            new_h, new_w = int(round(H * scale)), int(round(W * scale))
            resized = np.empty((frames.shape[0], new_h, new_w, frames.shape[3]),
                               dtype=np.uint8)
            for i in range(frames.shape[0]):
                resized[i] = cv2.resize(frames[i], (new_w, new_h),
                                        interpolation=cv2.INTER_AREA)
            frames = resized

    T = int(frames.shape[0])
    traj = Trajectory(
        frames=frames, frames_shape=tuple(frames.shape), task=task, id="0",
        metadata={"subsequence_length": T}, video_embeddings=None,
    )
    sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = state["batch_collator"]([sample])
    progress_inputs = batch["progress_inputs"]
    for k, v in progress_inputs.items():
        if hasattr(v, "to"):
            progress_inputs[k] = v.to(state["device"])

    exp_config = state["exp_config"]
    loss_cfg = getattr(exp_config, "loss", None)
    is_discrete = (
        loss_cfg is not None
        and str(getattr(loss_cfg, "progress_loss_type", "l2")).lower() == "discrete"
    )
    num_bins = (
        getattr(loss_cfg, "progress_discrete_bins", None) if loss_cfg is not None else None
    ) or getattr(exp_config.model, "progress_discrete_bins", 10)

    results = compute_batch_outputs(
        state["reward_model"], state["tokenizer"], progress_inputs,
        sample_type="progress", is_discrete_mode=is_discrete, num_bins=num_bins,
    )
    progress_pred = results.get("progress_pred", [])
    progress = (np.array(progress_pred[0], dtype=np.float32)
                if progress_pred and len(progress_pred) > 0
                else np.array([], dtype=np.float32))
    outputs_success = results.get("outputs_success", {})
    success_probs = outputs_success.get("success_probs", []) if outputs_success else []
    success = (np.array(success_probs[0], dtype=np.float32)
               if success_probs and len(success_probs) > 0
               else np.array([], dtype=np.float32))
    return progress, success


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True,
                   help="manifest.json from chain_wm_rollout.py")
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH,
                   help="HuggingFace robometer model id or local checkpoint")
    p.add_argument("--task_prompt", default=DEFAULT_TASK)
    p.add_argument("--max_frames", type=int, default=8,
                   help="Frames to sample per video (robometer trained on 8)")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max_side", type=int, default=256,
                   help="Resize frames so longest side is at most this many pixels")
    p.add_argument("--advantage_threshold", type=float, default=0.0)
    p.add_argument("--output_scored", default=None,
                   help="Default: <manifest>.scored.json")
    p.add_argument("--output_filtered", default=None,
                   help="Default: <manifest>.filtered.json")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)
    base_dir = manifest_path.parent

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Robometer: {args.model_path}  (device={device})", flush=True)
    state = _load_robometer(args.model_path, device)
    print(f"Scoring {len(manifest)} videos", flush=True)

    for i, entry in enumerate(manifest):
        vp_rel = entry.get("video")
        if not vp_rel:
            print(f"[{i+1}/{len(manifest)}] SKIP (no 'video' field)")
            continue
        vp = str(base_dir / vp_rel)
        if not os.path.exists(vp):
            print(f"[{i+1}/{len(manifest)}] SKIP (missing file: {vp})")
            continue
        try:
            progress, success = _score_video(
                vp, args.task_prompt, state, fps=args.fps,
                max_frames=args.max_frames, max_side=args.max_side,
            )
        except Exception as e:
            print(f"[{i+1}/{len(manifest)}] SCORE FAIL: {e}", flush=True)
            entry["score_error"] = str(e)
            continue
        entry["progress"] = progress.tolist()
        entry["success"] = success.tolist()
        entry["mean_progress"] = float(progress.mean()) if progress.size else 0.0
        entry["final_success"] = float(success[-1]) if success.size else 0.0
        print(f"[{i+1}/{len(manifest)}] init={entry.get('init_i')} sample={entry.get('sample_k')}  "
              f"mean_progress={entry['mean_progress']:.3f}  final_success={entry['final_success']:.3f}",
              flush=True)

    # Group by init_i, compute per-init baseline + advantage
    def _group_key(e):
        if "init_i" in e:
            return f"init_i={e['init_i']}"
        return e.get("sample_dir", "unknown")

    by_init = {}
    for e in manifest:
        if "mean_progress" not in e:
            continue
        by_init.setdefault(_group_key(e), []).append(e)
    for gk, entries in by_init.items():
        base = float(np.mean([e["mean_progress"] for e in entries]))
        for e in entries:
            e["baseline"] = base
            e["advantage"] = float(e["mean_progress"] - base)

    all_adv = [e["advantage"] for e in manifest if "advantage" in e]
    if all_adv:
        print("Advantage distribution:")
        print(f"  n={len(all_adv)}  mean={np.mean(all_adv):.4f}  std={np.std(all_adv):.4f}  "
              f"min={np.min(all_adv):.4f}  max={np.max(all_adv):.4f}")

    filtered = [e for e in manifest
                if e.get("advantage", -1) > args.advantage_threshold]
    print(f"After filter (advantage > {args.advantage_threshold}): "
          f"{len(filtered)} / {len(manifest)}")

    out_scored = args.output_scored or str(manifest_path.with_suffix(".scored.json"))
    out_filtered = args.output_filtered or str(manifest_path.with_suffix(".filtered.json"))
    with open(out_scored, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(out_filtered, "w") as f:
        json.dump(filtered, f, indent=2)
    print(f"Saved: {out_scored}")
    print(f"Saved: {out_filtered}")


if __name__ == "__main__":
    main()
