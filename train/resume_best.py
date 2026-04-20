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

# Auto-detect base directory (works on both Windows and Linux)
import os
BASE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKUP_DIR = BASE / "train" / "backup_model"
CODE_DIR = BASE / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from agent_ppo.workflow.state_layout import (
    CURRICULUM_STATE_SNAPSHOT_FILE,
    MANUAL_SAVE_MANIFEST_FILE,
    RESUME_CHECKPOINT_FILE,
    RESUME_META_FILE,
    RESUME_STATE_FILE,
    RUN_SESSION_MANIFEST_FILE,
    ensure_runtime_state_dirs,
    legacy_resume_curriculum_snapshot_path,
    legacy_resume_latest_checkpoint_path,
    legacy_resume_latest_meta_path,
    legacy_resume_latest_state_path,
)

LATEST_PKL = CODE_DIR / "latest_model.pkl"
BEST_PKL = CODE_DIR / "best_model.pkl"
STATE_LAYOUT = ensure_runtime_state_dirs(CODE_DIR)
PREPARED_RESUME_DIR = STATE_LAYOUT.current.prepared_resume_dir
RESUME_PKL = PREPARED_RESUME_DIR / RESUME_CHECKPOINT_FILE
RESUME_META = PREPARED_RESUME_DIR / RESUME_META_FILE
RESUME_STATE = PREPARED_RESUME_DIR / RESUME_STATE_FILE
RESUME_CURRICULUM = PREPARED_RESUME_DIR / CURRICULUM_STATE_SNAPSHOT_FILE
SNAPSHOT_DIR = STATE_LAYOUT.current.current_dir


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_sidecars_from_legacy_if_present():
    legacy_meta = legacy_resume_latest_meta_path(CODE_DIR)
    legacy_state = legacy_resume_latest_state_path(CODE_DIR)
    legacy_curriculum = legacy_resume_curriculum_snapshot_path(CODE_DIR)
    if legacy_meta.exists():
        shutil.copy2(legacy_meta, RESUME_META)
    if legacy_state.exists():
        payload = json.loads(legacy_state.read_text(encoding="utf-8"))
        payload["checkpoint_path"] = str(RESUME_PKL)
        if legacy_curriculum.exists():
            shutil.copy2(legacy_curriculum, RESUME_CURRICULUM)
            payload["curriculum_state_snapshot_path"] = str(RESUME_CURRICULUM)
        _write_json(RESUME_STATE, payload)
    elif legacy_curriculum.exists():
        shutil.copy2(legacy_curriculum, RESUME_CURRICULUM)


def _finalize_prepared_bundle(source_path: Path, label: str):
    PREPARED_RESUME_DIR.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != RESUME_PKL.resolve():
        shutil.copy2(source_path, RESUME_PKL)
    _copy_sidecars_from_legacy_if_present()
    if not RESUME_STATE.exists():
        _write_json(
            RESUME_STATE,
            {
                "checkpoint_path": str(RESUME_PKL),
                "global_step": 0,
                "global_step_since_resume": 0,
                "curriculum_state_snapshot_path": str(RESUME_CURRICULUM) if RESUME_CURRICULUM.exists() else None,
            },
        )
    if not RESUME_META.exists():
        _write_json(
            RESUME_META,
            {
                "trigger": "prepare",
                "checkpoint_path": str(RESUME_PKL),
                "resume_state_metadata_path": str(RESUME_STATE),
                "curriculum_state_snapshot_path": str(RESUME_CURRICULUM) if RESUME_CURRICULUM.exists() else None,
                "source_label": label,
            },
        )
    _write_json(
        PREPARED_RESUME_DIR / RUN_SESSION_MANIFEST_FILE,
        {
            "run_session_id": None,
            "source_label": label,
        },
    )
    _write_json(
        PREPARED_RESUME_DIR / MANUAL_SAVE_MANIFEST_FILE,
        {
            "save_name": "prepared_resume",
            "created_at": None,
            "source_label": label,
            "checkpoint_path": str(RESUME_PKL),
            "resume_compatible": True,
            "files": {
                "checkpoint": RESUME_CHECKPOINT_FILE,
                "resume_meta": RESUME_META_FILE,
                "resume_state": RESUME_STATE_FILE,
                "curriculum_snapshot": CURRICULUM_STATE_SNAPSHOT_FILE,
                "run_session": RUN_SESSION_MANIFEST_FILE,
            },
        },
    )


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
        legacy_resume = legacy_resume_latest_checkpoint_path(CODE_DIR)
        if not legacy_resume.exists():
            print("No prepared resume bundle found")
            return None
        print("Prepared bundle missing, but legacy resume checkpoint exists:")
        print(f"  legacy path: {legacy_resume}")
        return None

    size_mb = RESUME_PKL.stat().st_size / 1024 / 1024
    print("Latest prepared resume bundle:")
    print(f"  path   : {RESUME_PKL}")
    print(f"  size   : {size_mb:.1f}MB")
    if RESUME_META.exists():
        meta = json.loads(RESUME_META.read_text(encoding="utf-8"))
        for key in ("trigger", "episode_cnt", "clean_score", "saved_at", "pid"):
            if key in meta:
                print(f"  {key:<10}: {meta[key]}")
    return RESUME_PKL


def _copy_ready_checkpoint(source_path, label):
    pkl_size_mb = source_path.stat().st_size / 1024 / 1024
    _finalize_prepared_bundle(source_path, label)
    print(f"Prepared resume bundle from {label}:")
    print(f"  source : {source_path}")
    print(f"  size   : {pkl_size_mb:.1f}MB")
    print(f"  bundle : {PREPARED_RESUME_DIR}")
    print(f"\nNext: restart training with KAIWU_TRAINING_START_MODE=resume or let prepared bundle be picked by compatibility fallback.")


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

    _finalize_prepared_bundle(extracted, f"train_step={train_step}")

    # Cleanup temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    pkl_size_mb = os.path.getsize(RESUME_PKL) / 1024 / 1024
    print(f"Resume bundle ready:")
    print(f"  source : train_step={train_step} ({best['created_at'][:19]})")
    print(f"  bundle : {PREPARED_RESUME_DIR}")
    print(f"  size   : {pkl_size_mb:.1f}MB")
    print(f"\nNext: restart training with KAIWU_TRAINING_START_MODE=resume.")


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
