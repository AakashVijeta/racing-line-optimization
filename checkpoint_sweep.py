"""
Time-trial every Nth checkpoint of a training run and rank them.

Tracks: the 40 real circuits plus the procedural validation set; add --proc-test
to also report the held-out procedural test set.

    python checkpoint_sweep.py --run sac_v14
    python checkpoint_sweep.py --run sac_v14 --start 2000000 --every 250000 --workers 4

Ranking uses train, val and procedural-val tracks: laps completed first, then
mean lap time on the tracks every candidate finished. Test tracks (Suzuka and
the procedural test set) are reported but never used to rank, so they stay
unbiased generalization results.
"""
import argparse
import csv
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from config import TRACK_IDS, track_split

CKPT_RE = re.compile(r"sac_racing_(\d+)_steps\.zip$")


def find_candidates(run_dir, start, every):
    """(label, steps or None, path) for selected checkpoints plus best/final models."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    candidates = []
    for name in os.listdir(ckpt_dir) if os.path.isdir(ckpt_dir) else []:
        m = CKPT_RE.match(name)
        if m:
            steps = int(m.group(1))
            if steps >= start and steps % every == 0:
                candidates.append((f"{steps:,}", steps, os.path.join(ckpt_dir, name)))
    candidates.sort(key=lambda c: c[1])

    for label, rel in [("best_model", "best_model/best_model.zip"), ("final", "sac_racing_final.zip")]:
        path = os.path.join(run_dir, rel)
        if os.path.exists(path):
            candidates.append((label, None, path))
    return candidates


def build_eval_pool(include_proc_test):
    """Real circuits + procedural validation tracks (+ procedural test tracks): (pool, widths)."""
    from config import TRACK_WIDTHS
    from track_generator import procedural_split
    from track_preprocessing import build_track_pool

    pool, widths = build_track_pool(TRACK_IDS), dict(TRACK_WIDTHS)
    for split in ["val"] + (["test"] if include_proc_test else []):
        tracks, w = procedural_split(split)
        pool.update(tracks)
        widths.update(w)
    return pool, widths


def evaluate(path, obs_version, include_proc_test):
    # Imports and thread limits live here so each worker process sets them up itself
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import SAC
    from agent_eval import run_time_trials
    from config import obs_version_for

    model = SAC.load(path, device="cpu")
    pool, widths = build_eval_pool(include_proc_test)
    results = run_time_trials(model, pool, obs_version_for(path, model, obs_version), widths=widths)
    return {tid: r[0] for tid, r in results.items()}  # lap time or None


def group_of(track_id):
    """'train', 'val' or 'test' for real circuits; 'proc_val' / 'proc_test' for generated ones."""
    split = track_split(track_id)
    return f"proc_{split}" if track_id.startswith("proc-") else split


def main():
    parser = argparse.ArgumentParser(description="Rank a run's checkpoints by time trial on real and generated tracks")
    parser.add_argument("--run", required=True, help="run name under models/")
    parser.add_argument("--start", type=int, default=2_000_000, help="first checkpoint step to include")
    parser.add_argument("--every", type=int, default=250_000, help="checkpoint step interval")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--obs-version", type=int, choices=[1, 2, 3], default=None,
                        help="observation layout (default: inferred from the model's input size)")
    parser.add_argument("--proc-test", action="store_true",
                        help="also report the held-out procedural test tracks (never used for ranking)")
    parser.add_argument("--out", default=None, help="CSV path (default: models/<run>/checkpoint_sweep.csv)")
    args = parser.parse_args()

    run_dir = os.path.join("models", args.run)
    candidates = find_candidates(run_dir, args.start, args.every)
    if not candidates:
        raise SystemExit(f"No checkpoints found in {run_dir}")

    from track_preprocessing import prepare_tracks
    prepare_tracks(TRACK_IDS)  # build once here, not in every worker
    all_ids = list(build_eval_pool(args.proc_test)[0])

    print(f"Evaluating {len(candidates)} models on {len(all_ids)} tracks with {args.workers} workers...")
    laps = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, path, args.obs_version, args.proc_test): label
                   for label, _, path in candidates}
        for i, fut in enumerate(as_completed(futures), 1):
            label = futures[fut]
            laps[label] = fut.result()
            done = sum(t is not None for t in laps[label].values())
            print(f"  [{i}/{len(candidates)}] {label:<10} {done}/{len(all_ids)} laps")

    # Ranking uses train, val and procedural val; test tracks (real and procedural) never count
    ranked_groups = ("train", "val", "proc_val")
    report_groups = ranked_groups + ("test",) + (("proc_test",) if args.proc_test else ())
    ids_in = {g: [t for t in all_ids if group_of(t) == g] for g in report_groups}
    rank_tracks = [t for g in ranked_groups for t in ids_in[g]]
    test_tracks = ids_in["test"]
    # Mean lap time is only comparable on tracks that every candidate finished
    common = [t for t in rank_tracks if all(laps[c[0]][t] is not None for c in candidates)]

    rows = []
    for label, steps, path in candidates:
        lt = laps[label]
        rows.append({
            "model": label,
            "steps": steps if steps is not None else "",
            "path": path,
            **{f"{g}_done": sum(lt[t] is not None for t in ids_in[g]) for g in report_groups},
            "mean_time_common": float(np.mean([lt[t] for t in common])) if common else float("nan"),
            **{f"test_{t}": lt[t] for t in test_tracks},
            **{t: lt[t] for t in all_ids},
        })
    # NaN when no track was finished by every candidate: rank on laps only
    rows.sort(key=lambda r: (-sum(r[f"{g}_done"] for g in ranked_groups),
                             r["mean_time_common"] if r["mean_time_common"] == r["mean_time_common"] else float("inf")))

    fmt = lambda x: "DNF" if x is None else f"{x:.2f}"
    groups_hdr = "  ".join(f"{g:>10}" for g in report_groups if g != "test")
    print(f"\nRanked on {' + '.join(ranked_groups)} laps, then mean lap time "
          f"({len(common)} tracks finished by every model):")
    print(f"{'rank':>4}  {'model':<10} {groups_hdr} {'mean lap (s)':>13}  " +
          "  ".join(f"{t + ' (test)':>16}" for t in test_tracks))
    for i, r in enumerate(rows, 1):
        counts = "  ".join(f"{str(r[g + '_done']) + '/' + str(len(ids_in[g])):>10}"
                           for g in report_groups if g != "test")
        print(f"{i:>4}  {r['model']:<10} {counts} {r['mean_time_common']:>13.2f}  " +
              "  ".join(f"{fmt(r['test_' + t]):>16}" for t in test_tracks))

    out = args.out or os.path.join(run_dir, "checkpoint_sweep.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in r.items()})
    print(f"\nTop pick: {rows[0]['path']}")
    print(f"Per-track lap times saved to {out}")


if __name__ == "__main__":
    main()
