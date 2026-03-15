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
    "update your internal rule for how the AI seems to decide. Most examples require "
    "an acknowledgement response. Every tenth example requires a structured checkpoint "
    "JSON summary with normal cues, abnormal cues, exceptions, confidence and a "
    "rule-of-thumb."
)

TEACHING_INTRO_SIMPLIFIED_ONLY = (
    "We will now show you labelled examples that reveal how the AI behaved on "
    "specific charging sessions. For each example you will see one charging-chart "
    "image together with the AI's classification ('normal' or 'abnormal'). The chart "
    "shows only simplified power and SOC. Study the relationship between the "
    "simplified power curve and the SOC curve carefully and update your internal rule "
    "for how the AI seems to decide. Most examples require an acknowledgement response. "
    "Every tenth example requires a structured checkpoint JSON summary with normal cues, "
    "abnormal cues, exceptions, confidence and a rule-of-thumb."
)

TEACHING_INTRO_RULE_UPDATE = (
    "We will now show you labelled examples that reveal how the AI behaved on "
    "specific charging sessions. For each example you will see one charging-chart "
    "image together with the AI's classification ('normal' or 'abnormal'). The chart "
    "shows only simplified power and SOC. After each example, respond with a single "
    "JSON object that includes exactly five keys: "
    "{'description_sentence': '<one sentence>', 'rule_action': '<write|retain|rephrase>', "
    "'normal_cues': ['<cue>', '...'], 'abnormal_cues': ['<cue>', '...'], "
    "'rule_of_thumb': '<current rule>'}. The first example must use "
    "'rule_action': 'write' to establish an initial rule. Keep both cue arrays "
    "non-empty. If one side has no clear cue, include a placeholder cue string instead "
    "of leaving that array empty."
)

POST_EXAM_RULE_CARRYOVER_TEMPLATE = (
    "You finished the teaching phase with this locked rule snapshot: \"{rule}\". "
    "Do not update, rewrite, refine, or shorten this locked rule during the exam. "
    "Use this fixed rule to label every example in the batch."
)

GROUP_E_POST_EXAM_TIE_BREAKER_TEXT = {
    "closest_matching_pattern": (
        "If the fixed rule is ambiguous for one item, use this tie-break rule. "
        "Choose the label whose locked cue pattern is the closest match to the current "
        "example. Keep this tie-break behaviour consistent for all ambiguous items in "
        "the batch."
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
    group_e_post_exam_tie_breaker: str = "closest_matching_pattern",
) -> tuple[list[dict[str, Any]], str]:
    """Build multimodal user content for a pre- or post-teaching exam.

    The returned content is suitable for a single user message in the
    OpenAI chat API and contains the textual instructions as well as
    the exam images.

    Args:
        group: Participant group (A, B, C, D, E or F).
        phase: Experimental phase (PRE or POST).
        exam_items: List of (ExampleItem, image_path) pairs.
        fixed_rule_of_thumb: Optional fixed locked rule text for
            groups A-E in the post-exam phase.
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

    if phase is Phase.POST and group in (Group.A, Group.B, Group.C, Group.D, Group.E):
        rule_text = (fixed_rule_of_thumb or "").strip()
        if not rule_text:
            msg = (
                "Post-exam prompts for groups A-E require a non-empty fixed_rule_of_thumb."
            )
            raise ValueError(msg)
        carryover_text = POST_EXAM_RULE_CARRYOVER_TEMPLATE.format(rule=rule_text)
        if group is Group.E:
            tie_breaker_text = GROUP_E_POST_EXAM_TIE_BREAKER_TEXT.get(
                group_e_post_exam_tie_breaker,
            )
            if tie_breaker_text is None:
                msg = (
                    "Group E post-exam prompts require a supported "
                    "group_e_post_exam_tie_breaker value."
                )
                raise ValueError(msg)
        else:
            tie_breaker_text = (
                "If the fixed rule is ambiguous for one item, choose the label "
                "whose locked cue pattern is the closest match."
            )
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
    teaching_step_type: str = "example",
    current_rule_of_thumb: str | None = None,
) -> list[dict[str, Any]]:
    """Build multimodal user content for a single teaching example.

    Args:
        group: Participant group (A, B, C, D, E or F).
        item: ExampleItem describing the teaching example.
        image_path: Path to the teaching image.
        index: Position of this example in the teaching sequence (1-based).
        total: Total number of teaching examples in the sequence.
        teaching_step_type: Teaching step type.
        current_rule_of_thumb: Current rule-of-thumb state for the
            current group.

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
                "Respond with exactly one JSON object and no other text. "
                "Use this copyable template exactly: "
                "{\"description_sentence\":\"<one sentence description of this example>\","
                "\"rule_action\":\"write\","
                "\"normal_cues\":[\"<normal cue>\",\"...\"],"
                "\"abnormal_cues\":[\"<abnormal cue>\",\"...\"],"
                "\"rule_of_thumb\":\"<your initial rule>\"}. "
                "Both cue arrays must be non-empty. If this single example gives no clear "
                "cue for one side, include a placeholder such as "
                "\"no_clear_normal_cue_from_this_example\" or "
                "\"no_clear_abnormal_cue_from_this_example\"."
            )
        else:
            prior_rule = (current_rule_of_thumb or "").strip()
            instruction = (
                "Your current rule-of-thumb before this example is: "
                f"\"{prior_rule}\". "
                "Respond with exactly one JSON object and no other text. "
                "Use this copyable template exactly: "
                "{\"description_sentence\":\"<one sentence description of this example>\","
                "\"rule_action\":\"<write|retain|rephrase>\","
                "\"normal_cues\":[\"<normal cue>\",\"...\"],"
                "\"abnormal_cues\":[\"<abnormal cue>\",\"...\"],"
                "\"rule_of_thumb\":\"<your updated or retained rule>\"}. "
                "Both cue arrays must be non-empty. If this single example gives no clear "
                "cue for one side, include a placeholder such as "
                "\"no_clear_normal_cue_from_this_example\" or "
                "\"no_clear_abnormal_cue_from_this_example\"."
            )
        content.append({"type": "text", "text": instruction})
    else:
        if teaching_step_type == "checkpoint":
            prior_rule = (current_rule_of_thumb or "").strip()
            checkpoint_instruction = (
                "This example is a checkpoint. Respond with exactly one JSON object and "
                "no other text. Use this copyable template exactly: "
                "{\"acknowledged\":true,"
                "\"normal_cues\":[\"<normal cue>\",\"...\"],"
                "\"abnormal_cues\":[\"<abnormal cue>\",\"...\"],"
                "\"exceptions\":[\"<exception cue>\",\"...\"],"
                "\"confidence\":<float between 0 and 1>,"
                "\"rule_of_thumb\":\"<current rule>\"}. "
                "Both cue arrays must be non-empty. If one side has no clear cue yet, "
                "add a placeholder such as \"no_clear_normal_cue_at_checkpoint\" or "
                "\"no_clear_abnormal_cue_at_checkpoint\"."
            )
            if prior_rule:
                checkpoint_instruction += (
                    " The previous checkpoint rule-of-thumb is: "
                    f"\"{prior_rule}\"."
                )
            content.append({"type": "text", "text": checkpoint_instruction})
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
        "the current rule-of-thumb exactly. Keep non-empty normal_cues and "
        "abnormal_cues arrays in the JSON. If one side has no clear cue in this "
        "single example, include a placeholder cue string instead of leaving the "
        "array empty. Use this template exactly: "
        "{\"description_sentence\":\"<one sentence>\","
        "\"rule_action\":\"<write|retain|rephrase>\","
        "\"normal_cues\":[\"<normal cue or placeholder>\",\"...\"],"
        "\"abnormal_cues\":[\"<abnormal cue or placeholder>\",\"...\"],"
        "\"rule_of_thumb\":\"<rule>\"}. "
        f'The current rule-of-thumb is: "{prior_rule}". '
        "Respond again with exactly one JSON object and no other text."
    )


def build_checkpoint_retry_correction_text(error_message: str) -> str:
    """Build correction guidance for one non-E teaching checkpoint retry.

    Args:
        error_message: Protocol violation from the previous attempt.

    Returns:
        Retry instruction text for the next response.
    """
    return (
        "Retry this same teaching checkpoint example. "
        f'The previous response violates the protocol: "{error_message}". '
        "Respond again with exactly one JSON object and no other text. "
        "Use this template exactly: "
        "{\"acknowledged\":true,"
        "\"normal_cues\":[\"<normal cue or placeholder>\",\"...\"],"
        "\"abnormal_cues\":[\"<abnormal cue or placeholder>\",\"...\"],"
        "\"exceptions\":[\"<exception cue>\",\"...\"],"
        "\"confidence\":<float between 0 and 1>,"
        "\"rule_of_thumb\":\"<current rule>\"}. "
        "Both cue arrays must be non-empty. If one side has no clear cue, include "
        "a placeholder cue string."
    )


def build_rule_lock_user_content(
    group: Group,
    teaching_examples_seen: int,
    current_rule_of_thumb: str | None = None,
) -> list[dict[str, Any]]:
    """Build user content for the final locked-rule snapshot.

    Args:
        group: Participant group (A-E).
        teaching_examples_seen: Number of teaching examples shown.
        current_rule_of_thumb: Optional current rule-of-thumb before lock.

    Returns:
        List of content parts for the user message.
    """
    prior_rule = (current_rule_of_thumb or "").strip()
    group_text = (
        "overlay charts (raw, simplified and SOC)"
        if group in (Group.A, Group.B)
        else (
            "raw-only charts"
            if group is Group.C
            else "simplified-only charts"
        )
    )
    base_text = (
        "Teaching phase is complete. You have now seen "
        f"{teaching_examples_seen} labelled examples using {group_text}. "
        "Create one final locked rule snapshot that will be reused unchanged during the "
        "post-exam. This lock is strict, so do not leave out any required keys."
    )
    if prior_rule:
        base_text += f' Current rule-of-thumb before lock: "{prior_rule}".'

    schema_text = (
        "Respond with exactly one JSON object and no other text: "
        "{"
        "'locked_rule_of_thumb': '<final fixed rule>', "
        "'normal_cues': ['<normal cue>', '...'], "
        "'abnormal_cues': ['<abnormal cue>', '...'], "
        "'exceptions': ['<exception cue>', '...'], "
        "'confidence': <float between 0 and 1>"
        "}."
    )
    return [
        {"type": "text", "text": base_text},
        {"type": "text", "text": schema_text},
    ]


def build_rule_lock_retry_correction_text(error_message: str) -> str:
    """Build correction guidance for one final rule-lock retry.

    Args:
        error_message: Protocol violation from the previous attempt.

    Returns:
        Retry instruction text for the next response.
    """
    return (
        "Retry the final rule-lock response. "
        f'The previous response violates the protocol: "{error_message}". '
        "Respond again with exactly one JSON object with the required keys "
        "locked_rule_of_thumb, normal_cues, abnormal_cues, exceptions and confidence, "
        "and no other text."
    )
