"""Merge rollout trajectories (with per-episode advantage) + optional BC zarr
into a new DP3-format zarr that carries a per-step `advantage` column for
AWR-style advantage-weighted training.

Input:
  --scored_manifest  <rollout_dir>/manifest.scored.json  (from score_and_advantage.py)
  --bc_zarr          (optional) existing BC training zarr, e.g.
                     /home/hongyi/.../data/pourtea_rh56e2_1000.zarr
  --output_zarr      target zarr path

Output (zarr layout, same as DP3's VideoDataset expects, plus `advantage`):
  data/
    point_cloud: (T_total, 1280, 4) float32
    agent_pos:   (T_total, 7)       float32
    actions:     (T_total, 6)       float32
    advantage:   (T_total,)         float32   per-step episode advantage
  meta/
    episode_ends: (n_episodes,)     int64
"""
import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def _load_rollout(sample_dir: Path):
    pcs = np.load(sample_dir / "pcs_traj.npz")["pcs"].astype(np.float32)              # (T, 1280, 4)
    agent = np.load(sample_dir / "agent_pos_traj.npz")["agent_pos"].astype(np.float32)  # (T, 7)
    acts = np.load(sample_dir / "actions.npz")["actions"].astype(np.float32)           # (T, 6)
    T = min(pcs.shape[0], agent.shape[0], acts.shape[0])
    return pcs[:T], agent[:T], acts[:T]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scored_manifest", required=True)
    p.add_argument("--bc_zarr", default=None,
                   help="Original BC training zarr. If given, its episodes are prepended "
                        "with advantage = --bc_advantage.")
    p.add_argument("--bc_advantage", type=float, default=None,
                   help="Weight for BC episodes. Default: max advantage in scored manifest "
                        "(so BC weights as strongly as the best rollout).")
    p.add_argument("--bc_subset_n_episodes", type=int, default=None,
                   help="Randomly subsample BC to this many episodes (for BC:RL ratio control). "
                        "Default: use all BC episodes.")
    p.add_argument("--bc_subset_seed", type=int, default=42,
                   help="Seed for the BC subsampling (reproducible).")
    p.add_argument("--output_zarr", required=True)
    p.add_argument("--only_scored", action="store_true",
                   help="Include only RL entries that have a valid advantage (skip SCORE FAIL).")
    args = p.parse_args()

    manifest_path = Path(args.scored_manifest)
    base_dir = manifest_path.parent
    with open(manifest_path) as f:
        manifest = json.load(f)

    rl_entries = [e for e in manifest if "advantage" in e]
    if args.only_scored:
        rl_entries = [e for e in rl_entries if "score_error" not in e]

    advantages = [e["advantage"] for e in rl_entries]
    max_adv = max(advantages) if advantages else 1.0
    bc_adv = args.bc_advantage if args.bc_advantage is not None else max_adv

    print(f"[merge] RL entries with advantage: {len(rl_entries)}")
    print(f"[merge] RL advantage range: "
          f"min={min(advantages):.4f} max={max(advantages):.4f} "
          f"mean={np.mean(advantages):.4f}")
    print(f"[merge] BC advantage: {bc_adv:.4f}")

    # Collect all episodes: BC first (optional), then RL.
    all_pcs, all_agent, all_act, all_adv = [], [], [], []
    episode_ends = []
    total = 0
    n_bc = 0

    # ── BC episodes ──────────────────────────────────────────────────────────
    if args.bc_zarr is not None:
        print(f"[merge] Loading BC zarr: {args.bc_zarr}")
        bc = zarr.open(args.bc_zarr, mode="r")
        bc_pc    = np.asarray(bc["data/point_cloud"])
        bc_agent = np.asarray(bc["data/agent_pos"])
        bc_act   = np.asarray(bc["data/actions"])
        bc_ends  = np.asarray(bc["meta/episode_ends"])
        bc_starts = np.concatenate([[0], bc_ends[:-1]])
        # Optional subsample to control BC:RL step ratio (RISE Table II → 0.6 offline optimal).
        n_bc_total = len(bc_ends)
        if args.bc_subset_n_episodes is not None and args.bc_subset_n_episodes < n_bc_total:
            rng = np.random.RandomState(args.bc_subset_seed)
            keep_idx = np.sort(rng.choice(n_bc_total, size=args.bc_subset_n_episodes, replace=False))
            bc_starts = bc_starts[keep_idx]
            bc_ends   = bc_ends[keep_idx]
            print(f"[merge] BC subsample: keeping {len(bc_ends)} / {n_bc_total} episodes "
                  f"(seed={args.bc_subset_seed})")
        for s, e in zip(bc_starts, bc_ends):
            T = int(e - s)
            all_pcs.append(bc_pc[s:e].astype(np.float32))
            all_agent.append(bc_agent[s:e].astype(np.float32))
            all_act.append(bc_act[s:e].astype(np.float32))
            all_adv.append(np.full((T,), bc_adv, dtype=np.float32))
            total += T
            episode_ends.append(total)
            n_bc += 1
        print(f"[merge] BC episodes: {n_bc}  total steps: {total}")

    # ── RL episodes ──────────────────────────────────────────────────────────
    n_rl = 0
    n_skipped = 0
    for e in rl_entries:
        sample_dir = base_dir / e["sample_dir"]
        try:
            pcs, agent, acts = _load_rollout(sample_dir)
        except Exception as ex:
            print(f"[merge] skip {e['sample_dir']}: {ex}")
            n_skipped += 1
            continue
        T = pcs.shape[0]
        if T == 0:
            n_skipped += 1
            continue
        all_pcs.append(pcs)
        all_agent.append(agent)
        all_act.append(acts)
        all_adv.append(np.full((T,), float(e["advantage"]), dtype=np.float32))
        total += T
        episode_ends.append(total)
        n_rl += 1
    print(f"[merge] RL episodes added: {n_rl}  skipped: {n_skipped}  total steps now: {total}")

    # ── Write zarr ───────────────────────────────────────────────────────────
    output_zarr = Path(args.output_zarr)
    if output_zarr.exists():
        import shutil
        shutil.rmtree(output_zarr)
    root = zarr.open(str(output_zarr), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")

    pc_arr    = np.concatenate(all_pcs,   axis=0)
    agent_arr = np.concatenate(all_agent, axis=0)
    act_arr   = np.concatenate(all_act,   axis=0)
    adv_arr   = np.concatenate(all_adv,   axis=0)
    ends_arr  = np.asarray(episode_ends,  dtype=np.int64)

    data.create_dataset("point_cloud", data=pc_arr,    chunks=(1024, 1280, 4))
    data.create_dataset("agent_pos",   data=agent_arr, chunks=(4096, agent_arr.shape[1]))
    data.create_dataset("actions",     data=act_arr,   chunks=(4096, act_arr.shape[1]))
    data.create_dataset("advantage",   data=adv_arr,   chunks=(4096,))
    meta.create_dataset("episode_ends", data=ends_arr)

    print(f"[merge] Wrote {output_zarr}")
    print(f"  episodes: {len(ends_arr)}  (BC={n_bc}, RL={n_rl})")
    print(f"  steps:    {total}")
    print(f"  point_cloud: {pc_arr.shape}")
    print(f"  agent_pos:   {agent_arr.shape}")
    print(f"  actions:     {act_arr.shape}")
    print(f"  advantage:   {adv_arr.shape}  (min={adv_arr.min():.4f} max={adv_arr.max():.4f})")


if __name__ == "__main__":
    main()
