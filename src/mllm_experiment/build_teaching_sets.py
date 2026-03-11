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
TEACHING_SET_DIR = FIG_DIR / "teaching_sets" / "--archive"
TRIAL_TEACHING_SET_DIR = FIG_DIR / "teaching_sets" / "mllm_experiment_sets"
METADATA_DIR = PROJECT_ROOT_DIR / "Data" / "mllm_experiment_metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)
TEACHING_CSV = METADATA_DIR / "teaching_items.csv"

for directory in [TRIAL_TEACHING_SET_DIR, METADATA_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise SystemExit(
            f"Cannot create directory {TRIAL_TEACHING_SET_DIR}. "
            f"Check filesystem permissions (chmod/chown) or choose "
            f"another location."
        ) from e

for directory in [FIG_DIR, TEACHING_SET_DIR]: 
    if not Path(directory).exists():
        raise ValueError(f"Cannot find directory {directory}."
                         f"This should be set in the teaching set construction pipeline (see `06__MT4XAI.ipynb`)."
                         f"You can set the location of required directories/files in `build_teaching_sets.py`")

@dataclass(slots=True)
class TeachingExample:
    """Represent one labelled teaching example before anonymisation.

    Attributes:
        group: Teaching group the example belongs to ("A", "B", "C" or "D").
        src_path: Original path of the labelled image.
        ai_class: AI-predicted class ("normal" or "abnormal").
        simplicity_k: Integer k value extracted from the filename or set
            as a placeholder for group C.
        index: Original index in the group sequence (1-based).
    """

    group: str
    src_path: Path
    ai_class: str
    simplicity_k: int
    index: int


def find_labelled_examples(group: str) -> list[TeachingExample]:
    """Collect labelled examples for one group from the archive.

    This function scans the "--archive/<group> labelled" directory for
    png files and parses label and simplicity_k from the filename.

    Expected patterns:

        A/B/D: <GROUP>_ex_<index>_<label>_k<kvalue>__<id>.png
             e.g. A_ex_1_normal_k1__9489647.png

        C:   C_ex_<index>_<label>__<id>.png
             e.g. C_ex_1_normal__6823722.png

    For group C, filenames do not contain k, so this function sets
    simplicity_k = 0 as a placeholder. You can later overwrite these
    values in metadata/teaching_items.csv if you have the true k values
    in a separate table.

    Args:
        group: Group identifier, such as "A", "B", "C" or "D".

    Returns:
        List of TeachingExample instances for the group.
    """
    labelled_dir = TEACHING_SET_DIR / f"{group} labelled"
    if not labelled_dir.is_dir():
        msg = f"labelled directory not found for group {group}: {labelled_dir}"
        raise FileNotFoundError(msg)

    if group in {"A", "B", "D"}:
        pattern = re.compile(
            rf"^{group}_ex_(\d+)_"           # index
            r"(normal|abnormal)_k(\d+)__"    # label, k
            r"(\d+)\.png$"                   # random id
        )
    elif group == "C":
        pattern = re.compile(
            r"^C_ex_(\d+)_"                   # index
            r"(normal|abnormal)__"            # label (no k part)
            r"(\d+)\.png$"                    # random id
        )
    else:
        msg = f"unsupported group: {group}"
        raise ValueError(msg)

    examples: list[TeachingExample] = []

    for path in sorted(labelled_dir.glob("*.png")):
        match = pattern.match(path.name)
        if not match:
            raise ValueError(
                f"could not parse filename for group {group}: {path.name}"
            )

        if group in {"A", "B", "D"}:
            idx_str, label, k_str, _random_tail = match.groups()
            simplicity_k = int(k_str)
        else:  # group C
            idx_str, label, _random_tail = match.groups()
            # placeholder: update later from your own metadata if needed
            simplicity_k = 0

        examples.append(
            TeachingExample(
                group=group,
                src_path=path,
                ai_class=label,
                simplicity_k=simplicity_k,
                index=int(idx_str),
            )
        )

    return examples


def anonymise_group_examples(examples: Iterable[TeachingExample]) -> list[dict]:
    """Copy labelled examples into anonymised teaching sets and build metadata.

    This function creates a clean directory Figures/teaching_sets/<group>
    for each group and copies the original images using filenames that
    do not contain class labels or k values:

        <GROUP>_ex_<seq:03d>_<rand>.png

    Args:
        examples: Iterable of TeachingExample instances that belong to
            the same group.

    Returns:
        List of metadata rows to write into teaching_items.csv.
    """
    examples = list(examples)
    if not examples:
        return []

    group = examples[0].group
    dst_dir = TRIAL_TEACHING_SET_DIR / group
    dst_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for seq, ex in enumerate(sorted(examples, key=lambda e: e.index), start=1):
        rand_id = uuid4().hex[:7]
        filename = f"{group}_ex_{seq:03d}_{rand_id}.png"
        dst_path = dst_dir / filename

        shutil.copy2(ex.src_path, dst_path)

        item_id = f"{group}_{seq:03d}"

        # margin is unknown here; you can later overwrite this column if needed
        margin = 0.0

        # order_index is set to seq by default. for group A you can later
        # replace this with curriculum order based on k and margin.
        order_index = seq

        rows.append(
            {
                "item_id": item_id,
                "group": group,
                "filename": filename,
                "AI_class": ex.ai_class,
                "simplicity_k": ex.simplicity_k,
                "margin": margin,
                "order_index": order_index,
            }
        )

    return rows


def build_teaching_metadata(groups: list[str]) -> None:
    """Create anonymised teaching sets and a metadata CSV.

    Args:
        groups: List of group identifiers to process, typically ["A","B","C"].
    """
    all_rows: list[dict] = []

    for group in groups:
        print(f"[build_teaching_sets] processing group {group}")
        examples = find_labelled_examples(group)
        print(f"  found {len(examples)} labelled examples")
        rows = anonymise_group_examples(examples)
        print(f"  copied {len(rows)} examples to {TRIAL_TEACHING_SET_DIR / group}")
        all_rows.extend(rows)

    if not all_rows:
        print("[build_teaching_sets] no examples found, nothing to write")
        return

    fieldnames = [
        "item_id",
        "group",
        "filename",
        "AI_class",
        "simplicity_k",
        "margin",
        "order_index",
    ]

    with TEACHING_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[build_teaching_sets] wrote {len(all_rows)} rows to {TEACHING_CSV}")


def main() -> None:
    """Entry point for the script when run from the command line."""
    groups = ["A", "B", "C", "D"]
    build_teaching_metadata(groups)


if __name__ == "__main__":
    main()
