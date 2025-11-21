# src/mllm_experiment/prompts.py
# prompt templates + builders per phase / group
# prompts.py
from __future__ import annotations
import base64
from pathlib import Path
from typing import Any
from .data_loading import Group, Phase, ExampleItem


def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file as a base64 data URL string.

    Args:
        image_path: Path to the image file on disk.

    Returns:
        Base64-encoded data URL string suitable for the OpenAI API.
    """
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


INTRO_OVERLAY = (
    "There is an AI that has been trained on a large dataset of electric vehicle "
    "(EV) charging sessions. The AI classifies each charging session as either "
    "'normal' or 'abnormal'. It bases its decision only on patterns in the power "
    "transfer between charger and battery (in kW) over time, the battery's state "
    "of charge (SOC) over time, and a few technical features such as temperature "
    "and charger specifications. You do not know the exact rule the AI uses. "
    "You will be shown line charts of individual charging sessions. In each chart, "
    "the dashed orange line shows the original power in kW over minutes since the "
    "start of charging, the solid red line shows a simplified version of the same "
    "power curve, and the solid blue line shows the battery's SOC in percent over "
    "time. Each of these charts was shown to the AI. When we showed the AI only "
    "the simplified red power curve (without the orange line), it produced exactly "
    "the same classification as when it saw the original power curve. This means "
    "that, for this AI, the red simplified power curve together with the blue SOC "
    "curve contains all the information it needs to decide between 'normal' and "
    "'abnormal'. Your goal is to infer how this AI tends to behave and to simulate "
    "its decisions on new examples."
)

INTRO_RAW_ONLY = (
    "There is an AI that has been trained on a large dataset of electric vehicle "
    "(EV) charging sessions. The AI classifies each charging session as either "
    "'normal' or 'abnormal'. It bases its decision only on patterns in the power "
    "transfer between charger and battery (in kW) over time, the battery's state "
    "of charge (SOC) over time, and a few technical features such as temperature "
    "and charger specifications. You do not know the exact rule the AI uses. "
    "You will be shown line charts of individual charging sessions. In each chart, "
    "the orange line shows the power in kW over minutes since the start of "
    "charging and the blue line shows the battery's SOC in percent over time. "
    "Each of these charts was shown to the AI. Your goal is to infer how this AI "
    "tends to behave and to simulate its 'normal'/'abnormal' decisions on new "
    "examples."
)

TEACHING_INTRO = (
    "We will now show you labelled examples that reveal how the AI behaved on "
    "specific charging sessions. For each example you will see one charging-chart "
    "image together with the AI's classification ('normal' or 'abnormal'). Study "
    "the relationship between the power curve(s) and the SOC curve carefully and "
    "update your internal rule for how the AI seems to decide. After you have "
    "looked at the image and read the label, respond with a JSON object of the "
    "form {'acknowledged': true} and nothing else if you saw and understood the "
    "image. If the image is missing, unreadable, or otherwise not understandable, "
    "respond with {'acknowledged': false} instead."
)

# Should we include the 'decision rule' hint here?
POST_EXAM_INTRO = (
    "Let us test what you have learned about this AI. You will now see new, "
    "unlabelled charging session examples. For each example, you must guess how "
    "the AI would classify it ('normal' or 'abnormal') based on the patterns you "
    "observed during the teaching examples." 
    # "Try to apply a single, consistent "
    # "decision rule across all examples in the batch, even if you feel uncertain "
    # "about individual cases."
)


def exam_system_message() -> dict[str, Any]:
    """Build the system message used for the whole experiment.

    Returns:
        Dictionary representing the system message for the chat API.
    """
    content = (
        "You are participating in an Explainable AI study where you must infer the "
        "behaviour of a fixed black-box AI model from examples and then simulate "
        "its predictions. Your task is not to judge whether a charging session is "
        "good or bad in the real world, but to mimic how this AI tends to label "
        "sessions. For every task in this study you must respond with a single JSON "
        "object that exactly matches the schema described in the user's "
        "instructions. Do not include any additional explanations, commentary, code "
        "fences, or extra keys. Return only machine-readable JSON."
    )
    return {"role": "system", "content": content}


def build_exam_user_content(
    group: Group,
    phase: Phase,
    exam_items: list[tuple[ExampleItem, Path]],
) -> list[dict[str, Any]]:
    """Build multimodal user content for a pre- or post-teaching exam.

    The returned content is suitable for a single user message in the
    OpenAI chat API and contains the textual instructions as well as
    the exam images.

    Args:
        group: Participant group (A, B or C).
        phase: Experimental phase (PRE or POST).
        exam_items: List of (ExampleItem, image_path) pairs.

    Returns:
        List of content parts for the user message.
    """
    if group is Group.C or phase is Phase.PRE:
        intro = INTRO_RAW_ONLY
        modality_shown = "raw_only"
    elif phase is Phase.POST and group in (Group.A, Group.B):
            intro = INTRO_OVERLAY
            modality_shown = "overlay"
    else: 
        raise ValueError(f"Invalid phase: {phase} and/or group: {group}."
                         f"Select group from {list(Group)} and phase from {list(Phase)}.")
    
    if phase is Phase.POST:
        intro = POST_EXAM_INTRO + " " + intro

    content: list[dict[str, Any]] = [
        {"type": "text", "text": intro},
        {
            "type": "text",
            "text": (
                "You will be shown a *batch* of exam examples. For each example you must "
                "predict whether the charging session is 'normal' or 'abnormal'."
                "Important: "
                " - For each batch, respond with a **single JSON object**. "
                " - Do not include any extra keys. "
                " - Do not add explanations, comments, or prose outside the JSON. "
                " - The JSON must have exactly this structure: "
                "{"
                "'answers': ["
                    "{'item_id': '<ITEM_ID_1>', 'guess': 'your_guess_label'},"
                    "{'item_id': '<ITEM_ID_2>', 'guess': 'your_guess_label'},"
                    "..."
                "]"
                "}"
                "  where you must include exactly one entry object per seen example/item_id. The value of "
                "'guess' must be either 'normal' or 'abnormal' in lower case. Do not use any other labels."
            ),
        },
    ]

    for idx, (item, image_path) in enumerate(exam_items, start=1):
        data_url = encode_image_to_base64(image_path)
        content.append(
            {
                "type": "text",
                "text": f"Example {idx} (item_id={item.item_id}).",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url, "detail": "high"},
            }
        )

    return content, modality_shown


def build_teaching_user_content(
    group: Group,
    item: ExampleItem,
    image_path: Path,
    index: int,
    total: int,
) -> list[dict[str, Any]]:
    """Build multimodal user content for a single teaching example.

    Args:
        group: Participant group (A, B or C).
        item: ExampleItem describing the teaching example.
        image_path: Path to the teaching image.
        index: Position of this example in the teaching sequence (1-based).
        total: Total number of teaching examples in the sequence.

    Returns:
        List of content parts for the user message.
    """
    if index == 1:
        header = TEACHING_INTRO + " "
    else:
        header = ""

    label_text = (
        f"This is teaching example {index} of {total} "
        f"(item_id={item.item_id}). The AI classified this example as "
        f'"{item.ai_class}".'
    )

    data_url = encode_image_to_base64(image_path)

    content: list[dict[str, Any]] = []
    if header:
        content.append({"type": "text", "text": header})

    content.append({"type": "text", "text": label_text})
    content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})
    content.append(
        {
            "type": "text",
            "text": (
                'Once you have studied this example, respond with {"acknowledged": true} '
                "and nothing else."
            ),
        }
    )

    return content
