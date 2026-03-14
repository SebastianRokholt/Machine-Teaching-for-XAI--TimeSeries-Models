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

INTRO_SIMPLIFIED_ONLY = (
    "There is an AI that has been trained on a large dataset of electric vehicle "
    "(EV) charging sessions. The AI classifies each charging session as either "
    "'normal' or 'abnormal'. It bases its decision only on patterns in the power "
    "transfer between charger and battery (in kW) over time, the battery's state "
    "of charge (SOC) over time, and a few technical features such as temperature "
    "and charger specifications. You do not know the exact rule the AI uses. "
    "You will be shown line charts of individual charging sessions. In each chart, "
    "the solid red line shows a simplified version of the charging power in kW over "
    "minutes since the start of charging, and the solid blue line shows the battery's "
    "SOC in percent over time. Each chart was shown to the AI as simplified power and "
    "SOC only. Your goal is to infer how this AI tends to behave and to simulate its "
    "decisions on new examples."
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

TEACHING_INTRO_SIMPLIFIED_ONLY = (
    "We will now show you labelled examples that reveal how the AI behaved on "
    "specific charging sessions. For each example you will see one charging-chart "
    "image together with the AI's classification ('normal' or 'abnormal'). The chart "
    "shows only simplified power and SOC. Study the relationship between the "
    "simplified power curve and the SOC curve carefully and update your internal rule "
    "for how the AI seems to decide. After you have looked at the image and read the "
    "label, respond with a JSON object of the form {'acknowledged': true} and nothing "
    "else if you saw and understood the image. If the image is missing, unreadable, or "
    "otherwise not understandable, respond with {'acknowledged': false} instead."
)

TEACHING_INTRO_RULE_UPDATE = (
    "We will now show you labelled examples that reveal how the AI behaved on "
    "specific charging sessions. For each example you will see one charging-chart "
    "image together with the AI's classification ('normal' or 'abnormal'). The chart "
    "shows only simplified power and SOC. After each example, respond with a single "
    "JSON object that includes exactly three keys: "
    "{'description_sentence': '<one sentence>', 'rule_action': '<write|retain|rephrase>', "
    "'rule_of_thumb': '<current rule>'}. The first example must use "
    "'rule_action': 'write' to establish an initial rule."
)

POST_EXAM_RULE_CARRYOVER_TEMPLATE = (
    "You finished the teaching phase with this rule-of-thumb: \"{rule}\". "
    "Do not update, rewrite, refine, or shorten this rule during the exam. "
    "Use this fixed rule-of-thumb to label every example in the batch."
)

GROUP_E_POST_EXAM_TIE_BREAKER_TEXT = {
    "teaching_majority_label": (
        "If the fixed rule is ambiguous for one item, use this tie-break rule. "
        "Choose the label that appears most consistent with the dominant pattern in the "
        "teaching examples that shaped your fixed rule. Keep this tie-break behaviour "
        "consistent for all ambiguous items in the batch."
    ),
}

POST_EXAM_INTRO = (
    "Let us test what you have learned about this AI. You will now see new, "
    "unlabelled charging session examples. For each example, you must guess how "
    "the AI would classify it ('normal' or 'abnormal') based on the patterns you "
    "observed during the teaching examples."
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
    fixed_rule_of_thumb: str | None = None,
    group_e_post_exam_tie_breaker: str = "teaching_majority_label",
) -> tuple[list[dict[str, Any]], str]:
    """Build multimodal user content for a pre- or post-teaching exam.

    The returned content is suitable for a single user message in the
    OpenAI chat API and contains the textual instructions as well as
    the exam images.

    Args:
        group: Participant group (A, B, C, D, E or F).
        phase: Experimental phase (PRE or POST).
        exam_items: List of (ExampleItem, image_path) pairs.
        fixed_rule_of_thumb: Optional fixed rule-of-thumb for group E
            in the post-exam phase.
        group_e_post_exam_tie_breaker: Tie-break strategy for group E
            post-exam prompts.

    Returns:
        Tuple containing:
            - list of content parts for the user message,
            - modality identifier shown to the participant.
    """
    if phase is Phase.PRE:
        intro = INTRO_RAW_ONLY
        modality_shown = "raw_only"
    elif phase is Phase.POST and group in (Group.A, Group.B):
        intro = INTRO_OVERLAY
        modality_shown = "overlay"
    elif phase is Phase.POST and group in (Group.D, Group.E):
        intro = INTRO_SIMPLIFIED_ONLY
        modality_shown = "simplified_only"
    elif phase is Phase.POST and group in (Group.C, Group.F):
        intro = INTRO_RAW_ONLY
        modality_shown = "raw_only"
    else: 
        raise ValueError(f"Invalid phase: {phase} and/or group: {group}."
                         f"Select group from {list(Group)} and phase from {list(Phase)}.")
    
    if phase is Phase.POST:
        intro = POST_EXAM_INTRO + " " + intro

    content: list[dict[str, Any]] = [{"type": "text", "text": intro}]

    if phase is Phase.POST and group is Group.E:
        rule_text = (fixed_rule_of_thumb or "").strip()
        if not rule_text:
            msg = (
                "Group E post-exam prompts require a non-empty fixed_rule_of_thumb."
            )
            raise ValueError(msg)
        carryover_text = POST_EXAM_RULE_CARRYOVER_TEMPLATE.format(rule=rule_text)
        tie_breaker_text = GROUP_E_POST_EXAM_TIE_BREAKER_TEXT.get(
            group_e_post_exam_tie_breaker,
        )
        if tie_breaker_text is None:
            msg = (
                "Group E post-exam prompts require a supported "
                "group_e_post_exam_tie_breaker value."
            )
            raise ValueError(msg)
        decision_scaffolding_text = (
            "Use this decision checklist for every item in the batch. "
            "1. Do not rewrite, refine, or shorten the fixed rule. "
            "2. Apply one consistent fixed rule across all examples in the batch. "
            f"3. {tie_breaker_text}"
        )
        content.append({"type": "text", "text": carryover_text})
        content.append({"type": "text", "text": decision_scaffolding_text})

    content.append(
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
    )

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


def build_exam_missing_answers_repair_content(
    missing_item_ids: list[str],
) -> list[dict[str, Any]]:
    """Build a repair prompt for unresolved exam item IDs.

    Args:
        missing_item_ids: Item IDs that still require guesses.

    Returns:
        Content parts for one follow-up user message.
    """
    missing_ids_csv = ", ".join(missing_item_ids)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Your previous response does not include valid answers for all required "
                "item IDs in this batch. Return exactly one JSON object in this shape: "
                "{'answers': [{'item_id': '<ITEM_ID>', 'guess': '<normal|abnormal>'}, ...]}. "
                "Include entries only for these missing item IDs: "
                f"{missing_ids_csv}. "
                "Do not include already answered IDs. Do not include any text outside JSON."
            ),
        },
    ]
    return content


def build_teaching_user_content(
    group: Group,
    item: ExampleItem,
    image_path: Path,
    index: int,
    total: int,
    current_rule_of_thumb: str | None = None,
) -> list[dict[str, Any]]:
    """Build multimodal user content for a single teaching example.

    Args:
        group: Participant group (A, B, C, D, E or F).
        item: ExampleItem describing the teaching example.
        image_path: Path to the teaching image.
        index: Position of this example in the teaching sequence (1-based).
        total: Total number of teaching examples in the sequence.
        current_rule_of_thumb: Current rule-of-thumb state for group E.

    Returns:
        List of content parts for the user message.
    """
    if index == 1:
        if group is Group.E:
            header = TEACHING_INTRO_RULE_UPDATE + " "
        elif group is Group.D:
            header = TEACHING_INTRO_SIMPLIFIED_ONLY + " "
        else:
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
    if group is Group.E:
        if index == 1:
            instruction = (
                "Respond with exactly one JSON object and no other text: "
                "{"
                "'description_sentence': '<one sentence description of this example>', "
                "'rule_action': 'write', "
                "'rule_of_thumb': '<your initial rule>'"
                "}."
            )
        else:
            prior_rule = (current_rule_of_thumb or "").strip()
            instruction = (
                "Your current rule-of-thumb before this example is: "
                f"\"{prior_rule}\". "
                "Respond with exactly one JSON object and no other text: "
                "{"
                "'description_sentence': '<one sentence description of this example>', "
                "'rule_action': '<write|retain|rephrase>', "
                "'rule_of_thumb': '<your updated or retained rule>'"
                "}."
            )
        content.append({"type": "text", "text": instruction})
    else:
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


def build_group_e_retry_correction_text(
    current_rule_of_thumb: str,
    error_message: str,
) -> str:
    """Build correction guidance for one group E teaching retry.

    Args:
        current_rule_of_thumb: Current rule-of-thumb before the retry.
        error_message: Protocol violation from the previous attempt.

    Returns:
        Retry instruction text for the next response.
    """
    prior_rule = current_rule_of_thumb.strip()
    return (
        "Retry this same teaching example. "
        f'The previous response violates the protocol: "{error_message}". '
        "If rule_action is 'retain', the returned 'rule_of_thumb' must match "
        "the current rule-of-thumb exactly. "
        f'The current rule-of-thumb is: "{prior_rule}". '
        "Respond again with exactly one JSON object and no other text."
    )
