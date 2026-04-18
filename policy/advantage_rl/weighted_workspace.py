"""Advantage-weighted DP3 training entry point.

Uses the AWR weighting  w = exp(beta * advantage) normalized within each batch,
applied to DP3's per-sample diffusion loss (MSE on noise/sample prediction).
Reuses DP3's existing `TrainDP3Workspace` for everything except the training-step
loss computation, which is replaced inline by a duplicated (advantage-aware)
version of DP3.compute_loss.

Zarr format: output of merge_rl_zarr.py (adds a per-step `advantage` column to
the standard DP3 VideoDataset format).

Usage (Hydra; inherits dp3.yaml overrides):
  python policy/advantage_rl/weighted_workspace.py \\
      task=wild_pourtea_rh56e2_rl \\
      training.init_ckpt=/path/to/dp3_v7_epoch1000.ckpt \\
      training.advantage_beta=5.0 \\
      training.num_epochs=200 \\
      hydra.run.dir=data/outputs/wild_pourtea_rh56e2_rl_iter0
"""
from __future__ import annotations

import copy
import os
import pathlib
import sys
import time
from typing import Dict

# Make the DP3 repo importable
_DP3_ROOT = "/home/hongyi/scalevideomanip/3D-Diffusion-Policy/3D-Diffusion-Policy"
if _DP3_ROOT not in sys.path:
    sys.path.insert(0, _DP3_ROOT)

import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from einops import reduce
from omegaconf import OmegaConf

from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.policy.dp3 import DP3

import dp3_train  # registers hydra config search path

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: DP3's VideoDataset + an `advantage` key
# ─────────────────────────────────────────────────────────────────────────────
class WeightedVideoDataset(BaseDataset):
    def __init__(self, zarr_path, horizon=1, pad_before=0, pad_after=0,
                 seed=42, val_ratio=0.0, max_train_episodes=None, task_name=None):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=["agent_pos", "point_cloud", "actions", "advantage"])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=0.0, seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before, pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        vs = copy.copy(self)
        vs.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=self.horizon,
            pad_before=self.pad_before, pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        vs.train_mask = ~self.train_mask
        return vs

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action":      self.replay_buffer["actions"],
            "agent_pos":   self.replay_buffer["agent_pos"],
            "point_cloud": self.replay_buffer["point_cloud"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def _sample_to_data(self, sample):
        return {
            "obs": {
                "point_cloud": sample["point_cloud"][:].astype(np.float32),
                "agent_pos":   sample["agent_pos"][:].astype(np.float32),
            },
            "action":    sample["actions"].astype(np.float32),
            "advantage": sample["advantage"].astype(np.float32),   # (T,)
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self._sample_to_data(self.sampler.sample_sequence(idx))
        return dict_apply(data, torch.from_numpy)


# ─────────────────────────────────────────────────────────────────────────────
# Duplicated compute_loss from DP3 — returns per-sample loss (B,)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_per_sample_loss(model: DP3, batch):
    nobs = model.normalizer.normalize(batch["obs"])
    nactions = model.normalizer["action"].normalize(batch["action"])
    if not model.use_pc_color:
        nobs["point_cloud"] = nobs["point_cloud"][..., :3]

    B = nactions.shape[0]
    horizon = nactions.shape[1]
    trajectory = nactions
    cond_data = trajectory

    if model.obs_as_global_cond:
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :model.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = model.obs_encoder(this_nobs)
        if "cross_attention" in model.condition_type:
            global_cond = nobs_features.reshape(B, model.n_obs_steps, -1)
        else:
            global_cond = nobs_features.reshape(B, -1)
        local_cond = None
    else:
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        nobs_features = model.obs_encoder(this_nobs)
        nobs_features = nobs_features.reshape(B, horizon, -1)
        cond_data = torch.cat([nactions, nobs_features], dim=-1)
        trajectory = cond_data.detach()
        global_cond = None
        local_cond = None

    condition_mask = model.mask_generator(trajectory.shape)
    noise = torch.randn(trajectory.shape, device=trajectory.device)
    timesteps = torch.randint(
        0, model.noise_scheduler.config.num_train_timesteps,
        (trajectory.shape[0],), device=trajectory.device).long()
    noisy_trajectory = model.noise_scheduler.add_noise(trajectory, noise, timesteps)
    loss_mask = ~condition_mask
    noisy_trajectory[condition_mask] = cond_data[condition_mask]
    pred = model.model(sample=noisy_trajectory, timestep=timesteps,
                       local_cond=local_cond, global_cond=global_cond)

    pred_type = model.noise_scheduler.config.prediction_type
    if pred_type == "epsilon":
        target = noise
    elif pred_type == "sample":
        target = trajectory
    elif pred_type == "v_prediction":
        model.noise_scheduler.alpha_t = model.noise_scheduler.alpha_t.to(model.device)
        model.noise_scheduler.sigma_t = model.noise_scheduler.sigma_t.to(model.device)
        alpha_t = model.noise_scheduler.alpha_t[timesteps].unsqueeze(-1).unsqueeze(-1)
        sigma_t = model.noise_scheduler.sigma_t[timesteps].unsqueeze(-1).unsqueeze(-1)
        target = alpha_t * noise - sigma_t * trajectory
    else:
        raise ValueError(f"Unsupported prediction type {pred_type}")

    loss = F.mse_loss(pred, target, reduction="none")
    loss = loss * loss_mask.type(loss.dtype)
    per_sample_loss = reduce(loss, "b ... -> b", "mean")  # (B,)
    return per_sample_loss


def _awr_weights(advantage_per_sample: torch.Tensor, beta: float,
                 clip_max: float = 20.0) -> torch.Tensor:
    """AWR-style exponential weights normalized to mean 1 within the batch."""
    logits = (beta * advantage_per_sample).clamp(max=clip_max)
    w = torch.exp(logits - logits.max())   # stable softmax-style
    w = w / (w.mean() + 1e-8)
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Workspace: subclass DP3's, override train()
# ─────────────────────────────────────────────────────────────────────────────
class WeightedTrainDP3Workspace(dp3_train.TrainDP3Workspace):
    def __init__(self, cfg, output_dir=None, load_ckpt=False, ckpt_path=None):
        super().__init__(cfg, output_dir=output_dir, load_ckpt=load_ckpt, ckpt_path=ckpt_path)
        # Optional: warm-start from an existing DP3 checkpoint
        init_ckpt = getattr(cfg.training, "init_ckpt", None)
        if init_ckpt and os.path.exists(init_ckpt):
            print(f"[weighted] Warm-starting from {init_ckpt}")
            payload = torch.load(init_ckpt, pickle_module=dill, map_location="cpu")
            self.model.load_state_dict(payload["state_dicts"]["model"])
            if self.ema_model is not None and "ema_model" in payload.get("state_dicts", {}):
                self.ema_model.load_state_dict(payload["state_dicts"]["ema_model"])
            if "normalizer" in payload.get("pickles", {}):
                ns = dill.loads(payload["pickles"]["normalizer"])
                self.model.normalizer.load_state_dict(
                    ns.state_dict() if hasattr(ns, "state_dict") else ns)
                if self.ema_model is not None:
                    self.ema_model.normalizer.load_state_dict(
                        ns.state_dict() if hasattr(ns, "state_dict") else ns)

    def train(self):
        cfg = copy.deepcopy(self.cfg)
        beta = float(getattr(cfg.training, "advantage_beta", 5.0))
        print(f"[weighted] AWR weighting, beta={beta}")

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler, optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(self.train_dataloader) * cfg.training.num_epochs)
                // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1)
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        for local_epoch_idx in range(cfg.training.num_epochs):
            train_losses, train_weights = [], []
            with tqdm.tqdm(self.train_dataloader,
                           desc=f"[weighted] epoch {self.epoch}",
                           leave=False,
                           mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    adv = batch["advantage"][:, 0]    # (B,)
                    # Drop the advantage key before forwarding so DP3's compute_* paths
                    # don't see an unexpected obs field. (Also our _compute_per_sample_loss
                    # doesn't read it; we only use it for weighting.)
                    batch_for_model = {k: v for k, v in batch.items() if k != "advantage"}

                    per_sample = _compute_per_sample_loss(self.model, batch_for_model)
                    weights = _awr_weights(adv, beta=beta)            # (B,), mean ~1
                    raw_loss = (weights.detach() * per_sample).mean()
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    if cfg.training.use_ema:
                        ema.step(self.model)

                    train_losses.append(raw_loss.item())
                    train_weights.append(weights.detach().mean().item())
                    tepoch.set_postfix(loss=raw_loss.item(),
                                       w_mean=float(np.mean(train_weights[-1:])),
                                       adv_max=float(adv.max().item()),
                                       refresh=False)
                    self.global_step += 1
                    if (cfg.training.max_train_steps is not None
                        and batch_idx >= cfg.training.max_train_steps - 1):
                        break

            mean_loss = float(np.mean(train_losses)) if train_losses else 0.0
            mean_w    = float(np.mean(train_weights)) if train_weights else 0.0
            print(f"[weighted] epoch {self.epoch}  train_loss={mean_loss:.6f}  w_mean={mean_w:.3f}",
                  flush=True)

            # Periodic checkpoint
            if (self.epoch % cfg.training.checkpoint_every == 0
                and self.epoch > 0):
                output_path = os.path.join(
                    self._output_dir, f"{cfg.task.name}_{self.epoch}.ckpt")
                self.save_checkpoint(path=output_path)
                print(f"[weighted] Saved checkpoint: {output_path}", flush=True)

            self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(_DP3_ROOT) / "diffusion_policy_3d" / "config"),
    config_name="dp3",
)
def main(cfg):
    ws = WeightedTrainDP3Workspace(cfg)
    ws.train()


if __name__ == "__main__":
    main()
