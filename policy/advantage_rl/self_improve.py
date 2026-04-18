"""
One iteration of DP self-improvement via video WM imagination.

Stages (each a separate subprocess call to isolate CUDA contexts + env vars):
  1. rollout    : render_action_condition.py (IsaacSim + DP)
  2. imagine    : wm_imagine_from_manifest.py (Wan2.2 + LoRA)
  3. score      : score_and_advantage.py (Robometer)
  4. merge_zarr : merge_rl_zarr.py (build combined zarr)
  5. retrain    : weighted_workspace.py via hydra (advantage-weighted DP)
  6. eval       : eval_in_sim.py (IsaacSim, success rate)

Each iteration writes everything under --iter_dir <out>/iter<N>/.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = "/home/hongyi/PWM/EgoVLA_Release"
DP_LIB = "/home/hongyi/PWM/external/diffusion_policy"
DIFFSYNTH_DIR = "/home/hongyi/PWM/external/DiffSynth-Studio"
DIFFSYNTH_MODELS = f"{DIFFSYNTH_DIR}/models"
PY = "/home/hongyi/miniconda3/envs/pwm/bin/python"
PY_ROBOMETER = "/home/hongyi/miniconda3/envs/robometer/bin/python"


def run(cmd, cwd=None, env=None, description=""):
    print(f"\n{'='*70}\n[self_improve] {description}\n{'='*70}", flush=True)
    print(f"CWD = {cwd or os.getcwd()}", flush=True)
    print(f"CMD = {cmd}", flush=True)
    shell = isinstance(cmd, str)
    r = subprocess.run(cmd, shell=shell, cwd=cwd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Stage failed: {description} (rc={r.returncode})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--out_root", default=f"{ROOT}/dp_results/rl_runs")
    p.add_argument("--init_ckpt", required=True,
                   help="DP checkpoint to seed rollout + retraining.")
    p.add_argument("--lora", default=f"{DIFFSYNTH_DIR}/models/train/Wan2.2-TI2V-5B-Robot_open_drawer_lora/step-3500.safetensors")
    p.add_argument("--num_init_states", type=int, default=5)
    p.add_argument("--n_chunks", type=int, default=4)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--advantage_threshold", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--skip_stages", nargs="*", default=[],
                   help="rollout|imagine|score|merge|train|eval")
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--gpu_dp", default="0", help="GPU for IsaacSim+DP stages")
    p.add_argument("--gpu_wm", default="3", help="GPU for WM imagine")
    p.add_argument("--gpu_score", default="1", help="GPU for Robometer scoring")
    p.add_argument("--gpu_train", default="3", help="GPU for DP retraining")
    args = p.parse_args()

    iter_dir = Path(args.out_root) / f"iter{args.iter:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir = iter_dir / "rollouts"
    manifest = rollout_dir / "manifest.json"
    manifest_img = rollout_dir / "manifest.with_imagined.json"
    manifest_scored = rollout_dir / "manifest.with_imagined.scored.json"
    manifest_filtered = rollout_dir / "manifest.with_imagined.filtered.json"
    zarr_out = iter_dir / "rl.zarr"
    train_run_dir = iter_dir / "train"
    eval_dir = iter_dir / "eval"

    # 1. rollout
    if "rollout" not in args.skip_stages:
        cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu_dp} {PY} "
            f"{ROOT}/dp_results/render_action_condition.py "
            f"--ckpt {args.init_ckpt} "
            f"--num_init_states {args.num_init_states} "
            f"--n_chunks {args.n_chunks} "
            f"--horizon {args.horizon} "
            f"--rand_offset {2000 + args.iter * 100} "
            f"--output_dir {rollout_dir}"
        )
        run(cmd, cwd=ROOT, description="Stage 1: rollout (IsaacSim + DP)")
        assert manifest.exists(), f"Rollout did not produce {manifest}"

    # 2. imagine
    if "imagine" not in args.skip_stages:
        env = os.environ.copy()
        env["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
        env["DIFFSYNTH_MODEL_BASE_PATH"] = DIFFSYNTH_MODELS
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_wm
        cmd = (
            f"{PY} {ROOT}/dp_results/rl/wm_imagine_from_manifest.py "
            f"--manifest {manifest} "
            f"--lora {args.lora} "
            f"--num_frames 17 --height 320 --width 512 --steps 50"
        )
        run(cmd, cwd=ROOT, env=env, description="Stage 2: WM imagination")
        assert manifest_img.exists(), f"Imagine stage did not produce {manifest_img}"

    # 3. score
    if "score" not in args.skip_stages:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_score
        cmd = (
            f"{PY_ROBOMETER} {ROOT}/dp_results/rl/score_and_advantage.py "
            f"--manifest {manifest_img} "
            f"--advantage_threshold {args.advantage_threshold}"
        )
        run(cmd, cwd=ROOT, env=env, description="Stage 3: Robometer scoring + advantage")
        assert manifest_filtered.exists(), f"Score stage did not produce {manifest_filtered}"

    # 4. merge
    if "merge" not in args.skip_stages:
        cmd = (
            f"{PY} {ROOT}/dp_results/rl/merge_rl_zarr.py "
            f"--filtered_manifest {manifest_filtered} "
            f"--output_zarr {zarr_out}"
        )
        run(cmd, cwd=ROOT, description="Stage 4: merge BC+RL zarr")
        assert zarr_out.exists(), f"Merge stage did not produce {zarr_out}"

    # 5. retrain
    if "train" not in args.skip_stages:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT}:{DP_LIB}:{env.get('PYTHONPATH','')}"
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_train
        train_run_dir.mkdir(parents=True, exist_ok=True)
        # Use arg list (shell=False) so no shell-level quoting of '='. Hydra still
        # rejects bare '=' inside override values, so escape with '\='.
        escaped_ckpt = args.init_ckpt.replace("=", r"\=")
        cmd = [
            PY,
            f"{ROOT}/dp_results/rl/weighted_workspace.py",
            "--config-name=train_dp_results_adv",
            f"hydra.run.dir={train_run_dir}",
            f"task.zarr_path={zarr_out}",
            f"task.dataset.zarr_path={zarr_out}",
            f"training.init_ckpt={escaped_ckpt}",
            f"training.num_epochs={args.num_epochs}",
        ]
        run(cmd, cwd=DP_LIB, env=env, description="Stage 5: advantage-weighted DP retrain")

    # 6. eval
    new_ckpt = sorted((train_run_dir / "checkpoints").glob("epoch=*.ckpt"))[-1] \
               if (train_run_dir / "checkpoints").exists() else None
    if "eval" not in args.skip_stages and new_ckpt is not None:
        eval_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_dp
        cmd = (
            f"{PY} {ROOT}/dp_results/eval_in_sim.py "
            f"--ckpt {new_ckpt} "
            f"--num_episodes {args.eval_episodes} "
            f"--save_video "
            f"--output_dir {eval_dir}"
        )
        run(cmd, cwd=ROOT, env=env, description=f"Stage 6: IsaacSim eval ({args.eval_episodes} eps)")

    print(f"\n[self_improve] iter {args.iter} done. iter_dir={iter_dir}", flush=True)
    if new_ckpt:
        print(f"[self_improve] new checkpoint: {new_ckpt}", flush=True)


if __name__ == "__main__":
    main()
