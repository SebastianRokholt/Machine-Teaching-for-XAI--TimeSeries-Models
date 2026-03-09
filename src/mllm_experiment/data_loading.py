# src/mllm_experiment/data_loading.py
# loads images + metadata for exam sets and teaching sets
from __future__ import annotations
import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .config import (
    ExperimentConfig,
    TEACHING_METADATA_FILENAME,
    EXAM_METADATA_FILENAME,
)


class Group(str, Enum):
    """Identify the teaching condition for a participant."""

    A = "A"  # overlayed raw power, simplified power, simplified SOC with curriculum
    B = "B"  # As A but no curriculum (unordered)
    C = "C"  # As B (unordered) but raw-only, no simplifications
    D = "D"  # As A (curriculum) but simplifications only
    # TODO: Add group E
    # E = "E"  # As A but with enforced rule-of-thumb updating in teaching session
    # TODO: Add group F
    # F = "F"  # No teaching at all, just pre and post exam (for baseline)


class Phase(str, Enum):
    """Identify the experimental phase."""
    PRE = "pre"
    TEACHING = "teaching"
    POST = "post"


@dataclass(slots=True)
class ExampleItem:
    """Represent a single example item used in exams or teaching.

    Attributes:
        item_id: Unique identifier of the item.
        filename: File name of the image (no directory part).
        ai_class: AI-predicted class ("normal" or "abnormal").
        simplicity_k: Simplicity measure k used in curriculum ordering.
        margin: Margin of the classifier for this example.
        group: Group for which this teaching item is defined, or None for exam items.
        exam_set_id: Exam set identifier (e.g. "set1" or "set2") for exam items.
        order_index: Order index for curriculum or teaching ordering.
    """

    item_id: str
    filename: str
    ai_class: str
    simplicity_k: int
    margin: float
    group: Group | None = None
    exam_set_id: str | None = None
    order_index: int | None = None


def load_teaching_metadata(config: ExperimentConfig) -> dict[Group, list[ExampleItem]]:
    """Load teaching metadata and validate the image files.

    This function reads teaching_items.csv, constructs ExampleItem
    instances and groups them by teaching condition (A, B, C). It
    validates that the referenced image file exists under the expected
    teaching subdirectory.

    Args:
        config: Experiment configuration instance.

    Returns:
        Dictionary that maps each Group to a list of ExampleItem objects.
    """
    teaching_csv = config.metadata_root / TEACHING_METADATA_FILENAME
    if not teaching_csv.is_file():
        msg = f"Teaching metadata file not found: {teaching_csv}"
        raise FileNotFoundError(msg)

    items_by_group: dict[Group, list[ExampleItem]] = {g: [] for g in Group}

    with teaching_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group_value = row["group"].strip()
            group = Group(group_value)

            filename = row["filename"].strip()
            image_path = config.teaching_root / group.value / filename
            if not image_path.is_file():
                msg = f"Teaching image not found for group {group.value}: {image_path}"
                raise FileNotFoundError(msg)

            item = ExampleItem(
                item_id=row["item_id"].strip(),
                filename=filename,
                ai_class=row["AI_class"].strip(),
                simplicity_k=int(row["simplicity_k"]),
                margin=float(row["margin"]),
                group=group,
                exam_set_id=None,
                order_index=int(row["order_index"]) if row.get("order_index") else None,
            )
            items_by_group[group].append(item)

    # sort by order_index if available (useful for curriculum in group A)
    for group, items in items_by_group.items():
        items.sort(key=lambda it: it.order_index if it.order_index is not None else 10**9)

    return items_by_group


def load_exam_metadata(config: ExperimentConfig) -> dict[str, list[ExampleItem]]:
    """Load exam metadata and validate that images exist in both modalities.

    This function reads exam_items.csv, constructs ExampleItem instances
    for each exam set and validates that, for every filename, both the
    simplified and raw versions exist under the respective exam set
    subdirectories.

    Args:
        config: Experiment configuration instance.

    Returns:
        Dictionary that maps exam_set_id (e.g. "set1") to a list of
        ExampleItem objects.
    """
    exam_csv = config.metadata_root / EXAM_METADATA_FILENAME
    if not exam_csv.is_file():
        msg = f"Exam metadata file not found: {exam_csv}"
        raise FileNotFoundError(msg)

    items_by_set: dict[str, list[ExampleItem]] = {}

    with exam_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exam_set_id = row["exam_set_id"].strip()
            filename = row["filename"].strip()

            simplified_path = config.exam_root / exam_set_id / "overlay" / filename
            raw_path = config.exam_root / exam_set_id / "raw" / filename

            if not simplified_path.is_file():
                msg = f"Simplified exam image missing: {simplified_path}"
                raise FileNotFoundError(msg)
            if not raw_path.is_file():
                msg = f"Raw exam image missing: {raw_path}"
                raise FileNotFoundError(msg)

            item = ExampleItem(
                item_id=row["item_id"].strip(),
                filename=filename,
                ai_class=row["AI_class"].strip(),
                simplicity_k=int(row["simplicity_k"]),
                margin=float(row["margin"]),
                group=None,
                exam_set_id=exam_set_id,
                order_index=None,
            )
            items_by_set.setdefault(exam_set_id, []).append(item)

    return items_by_set


def resolve_exam_image_path(
    exam_root: Path,
    item: ExampleItem,
    group: Group,
    phase: Phase,
) -> Path:
    """Resolve the correct exam image path for a given item, group and phase.

    In the pre-teaching exam (Phase.PRE), all groups see the raw-only
    modality under 'raw'. In the post-teaching exam (Phase.POST), groups
    A and B see the overlay modality under 'overlay', while group C
    still sees the raw-only modality.

    Args:
        exam_root: Root directory for exam sets.
        item: ExampleItem instance describing the exam example.
        group: Participant group (A, B or C).
        phase: Experimental phase (PRE or POST).

    Returns:
        Path to the image file to present to the participant.
    """
    if item.exam_set_id is None:
        msg = f"Exam item {item.item_id} has no exam_set_id."
        raise ValueError(msg)

    if phase is Phase.PRE:
        subdir = "raw"
    elif phase is Phase.POST:
        subdir = "overlay" if group in (Group.A, Group.B) else "raw"
    else:
        msg = f"resolve_exam_image_path called for unsupported phase: {phase}"
        raise ValueError(msg)

    path = exam_root / item.exam_set_id / subdir / item.filename
    if not path.is_file():
        msg = f"Expected exam image not found: {path}"
        raise FileNotFoundError(msg)
    return path


def resolve_teaching_image_path(
    teaching_root: Path,
    item: ExampleItem,
) -> Path:
    """Resolve the teaching image path for a given teaching item.

    Args:
        teaching_root: Root directory for teaching sets.
        item: ExampleItem instance with a group attribute.

    Returns:
        Path to the teaching image file.
    """
    if item.group is None:
        msg = f"Teaching item {item.item_id} has no group assigned."
        raise ValueError(msg)

    path = teaching_root / item.group.value / item.filename
    if not path.is_file():
        msg = f"Expected teaching image not found: {path}"
        raise FileNotFoundError(msg)
    return path
