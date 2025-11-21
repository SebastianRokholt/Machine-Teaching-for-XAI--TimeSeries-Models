from __future__ import annotations
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid4


PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT_DIR / "Figures"

EXAM_SOURCE_DIR = FIG_DIR / "exam_sets"
TRIAL_EXAM_DIR = EXAM_SOURCE_DIR / "mllm_experiment_sets"

METADATA_DIR = PROJECT_ROOT_DIR / "Data" / "mllm_experiment_metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)
EXAM_CSV = METADATA_DIR / "exam_items.csv"

for directory in [TRIAL_EXAM_DIR, METADATA_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise SystemExit(
            f"Cannot create directory {TRIAL_EXAM_DIR}. "
            f"Check filesystem permissions (chmod/chown) or choose "
            f"another location."
        ) from e

for directory in [FIG_DIR, EXAM_SOURCE_DIR]:
    if not directory.exists():
        msg = (
            f"Cannot find directory {directory}. "
            "This directory is expected to be created by the MT4XAI "
            "notebooks that build the exam sets."
        )
        raise ValueError(msg)

# pattern for overlay filenames
OVERLAY_PATTERN = re.compile(
    r"^ex_(\d+)_" # index
    r"(normal|abnormal)_k(\d+)"  # label, k
    r"__\d+\.png$" # random tail
)

# pattern for raw filenames from group C (no k)
RAW_PATTERN = re.compile(
    r"^ex_(\d+)_"  
    r"(normal|abnormal)__" # (no 'k')
    r"\d+\.png$"            
)

@dataclass(slots=True)
class ExamExample:
    """Represent one exam example before anonymisation.

    Attributes:
        exam_set_id: Identifier for the exam set such as 'set1' or 'set2'.
        index: Integer index within the exam set (1-based).
        ai_class: AI-predicted class ('normal' or 'abnormal').
        simplicity_k: Integer k value extracted from the filename.
        overlay_src: Source path of the overlay image.
        raw_src: Source path of the raw image.
    """

    exam_set_id: str
    index: int
    ai_class: str
    simplicity_k: int
    overlay_src: Path
    raw_src: Path



def _parse_exam_filenames(exam_set_id: str) -> list[ExamExample]:
    """Parse existing exam filenames for a single set.

    Expected source layout (created by MT4XAI notebook):

        Figures/exam_sets/<set_id>/A/overlay/*.png   # overlay modality
        Figures/exam_sets/<set_id>/C/raw/*.png       # raw modality

    Overlay filenames follow:

        ex_<index>_<label>_k<kvalue>__<id>.png

    Raw filenames follow:

        ex_<index>_<label>__<id>.png

    Args:
        exam_set_id: Name of the exam set, such as "set1" or "set2".

    Returns:
        List of ExamExample objects with paired overlay and raw source paths.
    """
    overlay_dir = EXAM_SOURCE_DIR / exam_set_id / "A" / "overlay"
    raw_dir = EXAM_SOURCE_DIR / exam_set_id / "C" / "raw"

    if not overlay_dir.is_dir():
        raise FileNotFoundError(
            f"Overlay directory not found for {exam_set_id}: {overlay_dir}"
        )
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw directory not found for {exam_set_id}: {raw_dir}"
        )

    overlay_by_idx: dict[int, tuple[str, int, Path]] = {}
    raw_by_idx: dict[int, Path] = {}

    # parse overlay filenames (label + k)
    for path in sorted(overlay_dir.glob("*.png")):
        match = OVERLAY_PATTERN.match(path.name)
        if not match:
            raise ValueError(
                f"Could not parse overlay filename for {exam_set_id}: {path.name}"
            )
        idx_str, label, k_str = match.groups()
        idx = int(idx_str)
        k_val = int(k_str)
        overlay_by_idx[idx] = (label, k_val, path)

    # parse raw filenames (no k, only index + label)
    for path in sorted(raw_dir.glob("*.png")):
        match = RAW_PATTERN.match(path.name)
        if not match:
            raise ValueError(
                f"Could not parse raw filename for {exam_set_id}: {path.name}"
            )
        idx_str, _label = match.groups()
        idx = int(idx_str)
        raw_by_idx[idx] = path

    missing_in_raw = sorted(set(overlay_by_idx) - set(raw_by_idx))
    missing_in_overlay = sorted(set(raw_by_idx) - set(overlay_by_idx))

    if missing_in_raw:
        raise ValueError(
            f"Raw images missing for indices {missing_in_raw} in {exam_set_id}"
        )
    if missing_in_overlay:
        raise ValueError(
            f"Overlay images missing for indices {missing_in_overlay} in {exam_set_id}"
        )

    examples: list[ExamExample] = []
    for idx in sorted(overlay_by_idx):
        label, k_val, overlay_path = overlay_by_idx[idx]
        raw_path = raw_by_idx[idx]
        examples.append(
            ExamExample(
                exam_set_id=exam_set_id,
                index=idx,
                ai_class=label,
                simplicity_k=k_val,
                overlay_src=overlay_path,
                raw_src=raw_path,
            )
        )

    return examples


def _anonymise_exam_set(examples: Iterable[ExamExample]) -> list[dict]:
    """Copy exam images into anonymised sets and build metadata rows.
    For each example, this function creates two anonymised copies:
        Figures/exam_sets/mllm_experiment_sets/<set_id>/overlay/<fname>
        Figures/exam_sets/mllm_experiment_sets/<set_id>/raw/<fname>
    The anonymised filenames do not contain labels or k values and have
    the format: ex_<index:03d>_<rand>.png

    Args:
        examples: Iterable of ExamExample objects for a single set.

    Returns:
        List of dictionaries representing rows for exam_items.csv.
    """
    examples = list(examples)
    if not examples:
        return []

    set_id = examples[0].exam_set_id
    dst_overlay_dir = TRIAL_EXAM_DIR / set_id / "overlay"
    dst_raw_dir = TRIAL_EXAM_DIR / set_id / "raw"
    dst_overlay_dir.mkdir(parents=True, exist_ok=True)
    dst_raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for ex in sorted(examples, key=lambda e: e.index):
        rand_id = uuid4().hex[:7]
        filename = f"ex_{ex.index:03d}_{rand_id}.png"
        overlay_dst = dst_overlay_dir / filename
        raw_dst = dst_raw_dir / filename
        shutil.copy2(ex.overlay_src, overlay_dst)
        shutil.copy2(ex.raw_src, raw_dst)
        item_id = f"{set_id}_ex_{ex.index:03d}"

        # margin is unknown here; you can overwrite this column later if needed
        margin = 0.0

        rows.append(
            {
                "exam_set_id": set_id,
                "item_id": item_id,
                "filename": filename,
                "AI_class": ex.ai_class,
                "simplicity_k": ex.simplicity_k,
                "margin": margin,
                # keep original filenames for cross-checking with pool metadata
                "source_overlay_filename": ex.overlay_src.name,
                "source_raw_filename": ex.raw_src.name,
            }
        )

    return rows


def build_exam_metadata(exam_sets: list[str]) -> None:
    """Create anonymised exam sets and a metadata CSV.

    Args:
        exam_sets: List of exam set identifiers, typically ["set1", "set2"].
    """
    all_rows: list[dict] = []

    for set_id in exam_sets:
        print(f"[build_exam_sets] processing {set_id}")
        examples = _parse_exam_filenames(set_id)
        print(f"  found {len(examples)} examples")
        rows = _anonymise_exam_set(examples)
        print(
            f"  copied {len(rows)} examples to "
            f"{TRIAL_EXAM_DIR / set_id / 'overlay'} and 'raw'"
        )
        all_rows.extend(rows)

    if not all_rows:
        print("[build_exam_sets] no examples found, nothing to write")
        return

    fieldnames = [
        "exam_set_id",
        "item_id",
        "filename",
        "AI_class",
        "simplicity_k",
        "margin",
        "source_overlay_filename",
        "source_raw_filename",
    ]

    with EXAM_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[build_exam_sets] wrote {len(all_rows)} rows to {EXAM_CSV}")


def main() -> None:
    """Entry point when the script is run from the command line."""
    exam_sets = ["set1", "set2"]
    build_exam_metadata(exam_sets)


if __name__ == "__main__":
    main()