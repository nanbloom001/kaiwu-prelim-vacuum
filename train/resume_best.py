"""Checkpoint management utility for training resume.

Usage:
    python train/resume_best.py list              - List all checkpoints
    python train/resume_best.py best              - Show best checkpoint info
    python train/resume_best.py latest            - Show local resume checkpoint info
    python train/resume_best.py prepare [latest|best] - Prepare checkpoint for resume
    python train/resume_best.py clean [--keep 3]  - Remove old checkpoints, keep best N
"""
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

# 仓库根目录（train/ 的上一级），不再依赖硬编码盘符
BASE = Path(__file__).resolve().parent.parent
BACKUP_DIR = BASE / "train" / "backup_model"
CODE_DIR = BASE / "code"
RESUME_PKL = CODE_DIR / "model.ckpt-resume.pkl"
RESUME_META = CODE_DIR / "model.ckpt-resume.meta.json"
LATEST_PKL = CODE_DIR / "latest_model.pkl"
BEST_PKL = CODE_DIR / "best_model.pkl"
SNAPSHOT_DIR = CODE_DIR / "resume_snapshots"


def load_checkpoints():
    """Load all checkpoint metadata from backup_model/*.zip.json"""
    ckpts = []
    for jf in BACKUP_DIR.glob("*.zip.json"):
        with open(jf, "r", encoding="utf-8") as f:
            meta = json.load(f)
        zip_path = BACKUP_DIR / meta["model_file_name"]
        if zip_path.exists():
            ckpts.append({**meta, "zip_path": str(zip_path), "json_path": str(jf)})
    ckpts.sort(key=lambda x: x.get("train_step", 0))
    return ckpts


def cmd_list():
    """List all checkpoints sorted by train_step"""
    ckpts = load_checkpoints()
    if not ckpts:
        print("No checkpoints found")
        return
    print(f"{'train_step':>10}  {'created_at':<26}  {'train_time':>10}  {'size':>8}  filename")
    print("-" * 90)
    for c in ckpts:
        size_mb = os.path.getsize(c["zip_path"]) / 1024 / 1024
        print(
            f"{c['train_step']:>10}  {c['created_at'][:19]:<26}  "
            f"{c['train_time']:>8}s  {size_mb:>6.1f}MB  {c['model_file_name']}"
        )
    print(f"\nTotal: {len(ckpts)} checkpoints")


def cmd_best():
    """Find and display the best checkpoint"""
    ckpts = load_checkpoints()
    if not ckpts:
        print("No checkpoints found")
        return None

    # Prefer most recent, full-size checkpoints (>1MB).
    # Small checkpoints (<1MB) are likely from different model architectures.
    full_ckpts = [c for c in ckpts if os.path.getsize(c["zip_path"]) > 1_000_000]
    if full_ckpts:
        pool = full_ckpts
        note = " (full-size, most recent)"
    else:
        pool = ckpts
        note = " (most recent)"

    best = max(pool, key=lambda x: x.get("created_at", ""))
    size_mb = os.path.getsize(best["zip_path"]) / 1024 / 1024

    print(f"Best checkpoint{note}:")
    print(f"  train_step : {best['train_step']}")
    print(f"  created_at : {best['created_at'][:19]}")
    print(f"  train_time : {best['train_time']}s ({best['train_time']/3600:.1f}h)")
    print(f"  size       : {size_mb:.1f}MB")
    print(f"  zip        : {best['model_file_name']}")
    print(f"  internal   : {best['model_file_path'][0]}")
    return best


def cmd_latest():
    if not RESUME_PKL.exists():
        print("No local resume checkpoint found")
        return None

    size_mb = RESUME_PKL.stat().st_size / 1024 / 1024
    print("Latest local resume checkpoint:")
    print(f"  path   : {RESUME_PKL}")
    print(f"  size   : {size_mb:.1f}MB")
    if RESUME_META.exists():
        meta = json.loads(RESUME_META.read_text(encoding="utf-8"))
        for key in ("trigger", "episode_cnt", "clean_score", "saved_at", "pid"):
            if key in meta:
                print(f"  {key:<10}: {meta[key]}")
    if SNAPSHOT_DIR.exists():
        count = len(list(SNAPSHOT_DIR.glob("*.pkl")))
        print(f"  snapshots : {count} files in {SNAPSHOT_DIR}")
    return RESUME_PKL


def _copy_ready_checkpoint(source_path, label):
    pkl_size_mb = source_path.stat().st_size / 1024 / 1024
    if source_path.resolve() != RESUME_PKL.resolve():
        shutil.copy2(source_path, RESUME_PKL)
    print(f"Resume from {label}:")
    print(f"  source : {source_path}")
    print(f"  size   : {pkl_size_mb:.1f}MB")
    print(f"  target : {RESUME_PKL}")
    print(f"\nNext: restart training containers. Agent will auto-load this checkpoint.")


def cmd_prepare(mode="auto"):
    """Prepare checkpoint for resume."""
    mode = (mode or "auto").lower()
    if mode not in {"auto", "latest", "best"}:
        print(f"Unsupported prepare mode: {mode}")
        return

    if mode in {"auto", "latest"} and RESUME_PKL.exists():
        _copy_ready_checkpoint(RESUME_PKL, "model.ckpt-resume.pkl")
        return

    if mode in {"auto", "latest"} and LATEST_PKL.exists():
        _copy_ready_checkpoint(LATEST_PKL, "latest_model.pkl")
        return

    if mode in {"auto", "best"} and BEST_PKL.exists():
        _copy_ready_checkpoint(BEST_PKL, "best_model.pkl")
        return

    best = cmd_best()
    if best is None:
        return

    zip_path = best["zip_path"]
    internal_pkl = best["model_file_path"][0]  # e.g. "ckpt/model.ckpt-1701.pkl"
    train_step = best["train_step"]

    # Extract to temp dir
    tmp_dir = BASE / "train" / "_resume_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"\nExtracting {internal_pkl} from zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the pkl file inside the zip
        pkl_name = None
        for name in zf.namelist():
            if name.endswith(".pkl"):
                pkl_name = name
                break
        if pkl_name is None:
            print("ERROR: No .pkl file found in zip")
            return
        zf.extract(pkl_name, tmp_dir)

    extracted = tmp_dir / pkl_name
    if not extracted.exists():
        print(f"ERROR: Extraction failed, {extracted} not found")
        return

    # Copy to code directory as model.ckpt-resume.pkl
    shutil.copy2(extracted, RESUME_PKL)

    # Cleanup temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    pkl_size_mb = os.path.getsize(RESUME_PKL) / 1024 / 1024
    print(f"Resume checkpoint ready:")
    print(f"  source : train_step={train_step} ({best['created_at'][:19]})")
    print(f"  target : {RESUME_PKL}")
    print(f"  size   : {pkl_size_mb:.1f}MB")
    print(f"\nNext: restart training containers. Agent will auto-load this checkpoint.")


def cmd_clean(keep=3):
    """Remove old checkpoints, keep the best N by train_step"""
    ckpts = load_checkpoints()
    if len(ckpts) <= keep:
        print(f"Only {len(ckpts)} checkpoints, nothing to clean (keep={keep})")
        return

    # Sort by train_step desc, keep top N
    ckpts_sorted = sorted(ckpts, key=lambda x: x.get("train_step", 0), reverse=True)
    to_keep = {c["model_file_name"] for c in ckpts_sorted[:keep]}
    to_remove = [c for c in ckpts if c["model_file_name"] not in to_keep]

    print(f"Keeping {keep} best checkpoints:")
    for c in ckpts_sorted[:keep]:
        print(f"  + train_step={c['train_step']}  {c['model_file_name']}")

    print(f"Removing {len(to_remove)} old checkpoints:")
    for c in to_remove:
        zip_p = Path(c["zip_path"])
        json_p = Path(c["json_path"])
        size_mb = zip_p.stat().st_size / 1024 / 1024
        zip_p.unlink(missing_ok=True)
        json_p.unlink(missing_ok=True)
        print(f"  - train_step={c['train_step']}  {size_mb:.1f}MB  {c['model_file_name']}")

    print(f"\nFreed space. {keep} checkpoints remaining.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "list":
        cmd_list()
    elif cmd == "best":
        cmd_best()
    elif cmd == "latest":
        cmd_latest()
    elif cmd == "prepare":
        mode = sys.argv[2].lower() if len(sys.argv) >= 3 else "auto"
        cmd_prepare(mode)
    elif cmd == "clean":
        keep = 3
        if len(sys.argv) >= 3:
            try:
                keep = int(sys.argv[2])
            except ValueError:
                print(f"Invalid keep value: {sys.argv[2]}")
                return
        cmd_clean(keep)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
