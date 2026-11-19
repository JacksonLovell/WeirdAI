
"""
Distillation dataset utilities for Weird AI.

Chapter 8 introduces model distillation:

    teacher model -> synthetic training examples -> student model

For Weird AI, this means preparing teacher-generated parody lyrics and teacher
notes as reusable supervised training data. This module does not train a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import random
from pathlib import Path
from typing import Any, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]:
        ...


@dataclass
class TeacherExample:
    prompt: str
    teacher_lyrics: str
    teacher_notes: str
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillationRecord:
    prompt: str
    target_text: str
    full_text: str
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_teacher_example(example: dict) -> bool:
    """
    Validate that a raw teacher example contains usable fields.

    Required fields:
        prompt, teacher_lyrics, teacher_notes, quality_score
    """
    # TODO:
    # 1. Check that all required fields are present.
    # 2. Check that prompt, teacher_lyrics, and teacher_notes are non-empty strings.
    # 3. Check that quality_score can be converted to float.
    # 4. Return True only if all checks pass.
    required_fields = ["prompt", "teacher_lyrics", "teacher_notes", "quality_score"]
    for field in required_fields:
        if field not in example:
            return False
        if field in ["prompt", "teacher_lyrics", "teacher_notes"]:
            if not isinstance(example[field], str) or not example[field].strip():
                return False
        if field == "quality_score":
            try:
                float(example[field])
            except (ValueError, TypeError):
                return False
    return True

def teacher_example_from_dict(example: dict) -> TeacherExample:
    """Convert a valid dictionary into a TeacherExample."""
    # TODO:
    # 1. Validate the example.
    # 2. Raise ValueError if invalid.
    # 3. Convert quality_score to float.
    # 4. Put any extra keys into metadata.
    # 5. Return a TeacherExample.
    valid_keys = {"prompt", "teacher_lyrics", "teacher_notes", "quality_score"}
    if not validate_teacher_example(example):
        raise ValueError("Invalid teacher example dictionary.")
    metadata = {k: v for k, v in example.items() if k not in valid_keys}
    return TeacherExample(
        prompt=example["prompt"],
        teacher_lyrics=example["teacher_lyrics"],
        teacher_notes=example["teacher_notes"],
        quality_score=float(example["quality_score"]),
        metadata=metadata,
    )
    raise NotImplementedError("Implement teacher_example_from_dict.")


def format_teacher_response(
    teacher_lyrics: str,
    teacher_notes: str,
    *,
    include_thinking: bool = True,
) -> str:
    """Format teacher notes and lyrics as a distillation target."""
    # TODO:
    # 1. Strip teacher_lyrics and teacher_notes.
    # 2. If include_thinking=True, return notes inside <think>...</think> tags,
    #    followed by a blank line and the lyrics.
    # 3. If include_thinking=False, return only the lyrics.
    # 4. Avoid extra leading/trailing whitespace.
    teacher_lyrics = teacher_lyrics.strip()
    teacher_notes = teacher_notes.strip()
    if include_thinking:
        return f"<think>{teacher_notes}</think>\n\n{teacher_lyrics}"
    else:
        return teacher_lyrics
    
    raise NotImplementedError("Implement format_teacher_response.")


def format_training_prompt(prompt: str) -> str:
    """Format a Weird AI prompt for distillation training."""
    # TODO:
    # 1. Strip the prompt.
    # 2. Return a consistent format such as:
    #    "User: Write a parody...\\nAssistant:"
    prompt = prompt.strip()
    return f"User: {prompt}\nAssistant:"

    raise NotImplementedError("Implement format_training_prompt.")


def build_full_text(prompt: str, target_text: str) -> str:
    """Combine formatted prompt and target text."""
    # TODO:
    # 1. Use format_training_prompt(prompt).
    # 2. Append target_text after the assistant marker.
    # 3. Return the full training string.
    formatted_prompt = format_training_prompt(prompt)
    return f"{formatted_prompt}{target_text}"

    raise NotImplementedError("Implement build_full_text.")


def tokenize_text(tokenizer: TokenizerLike, text: str) -> list[int]:
    """Tokenize text using an object with encode(text) -> list[int]."""
    # TODO:
    # 1. Call tokenizer.encode(text).
    # 2. Validate that the result is a list of integers.
    # 3. Return the token IDs.
    tokenized = tokenizer.encode(text)
    if not isinstance(tokenized, list) or not all(isinstance(i, int) for i in tokenized):
        raise ValueError("Tokenizer did not return a list of integers.")
    return tokenized

    raise NotImplementedError("Implement tokenize_text.")


def create_labels(
    input_ids: list[int],
    prompt_length: int,
    *,
    mask_prompt: bool = True,
    ignore_index: int = -100,
) -> list[int]:
    """
    Create training labels.

    If mask_prompt=True, prompt token labels should be replaced with ignore_index.
    """
    # TODO:
    # 1. Copy input_ids into labels.
    # 2. If mask_prompt=True, replace labels before prompt_length with ignore_index.
    # 3. Return labels.
    labels = input_ids.copy()
    if mask_prompt:
        for i in range(prompt_length):
            labels[i] = ignore_index
    return labels

    raise NotImplementedError("Implement create_labels.")


def build_distillation_record(
    example: TeacherExample,
    tokenizer: TokenizerLike,
    *,
    include_thinking: bool = True,
    mask_prompt: bool = True,
) -> DistillationRecord:
    """Build one processed distillation record."""
    # TODO:
    # 1. Format the teacher response.
    # 2. Format the prompt.
    # 3. Build the full text.
    # 4. Tokenize the prompt and full text.
    # 5. Create labels.
    # 6. Return a DistillationRecord.
    formatted_response = format_teacher_response(
        example.teacher_lyrics,
        example.teacher_notes,
        include_thinking=include_thinking,
    )
    formatted_prompt = format_training_prompt(example.prompt)
    full_text = build_full_text(example.prompt, formatted_response)
    input_ids = tokenize_text(tokenizer, full_text)
    prompt_length = len(tokenize_text(tokenizer, formatted_prompt))
    labels = create_labels(input_ids, prompt_length, mask_prompt=mask_prompt)
    return DistillationRecord(
        prompt=example.prompt,
        target_text=formatted_response,
        full_text=full_text,
        input_ids=input_ids,
        labels=labels,
        prompt_length=prompt_length,
        metadata=example.metadata,  
    )   
    raise NotImplementedError("Implement build_distillation_record.")


def filter_teacher_examples(
    examples: list[TeacherExample],
    *,
    min_quality_score: float = 0.75,
    min_lyric_lines: int = 4,
) -> list[TeacherExample]:
    """Keep only high-quality teacher examples."""
    # TODO:
    # 1. Keep examples with quality_score >= min_quality_score.
    # 2. Keep examples with at least min_lyric_lines non-empty lyric lines.
    # 3. Return the filtered list.
    filtered = []
    for example in examples:
        if example.quality_score >= min_quality_score:
            normalized_lyrics = example.teacher_lyrics.replace("\\n", "\n")
            lyric_lines = [line for line in normalized_lyrics.splitlines() if line.strip()]
            if len(lyric_lines) >= min_lyric_lines:
                filtered.append(example)
    return filtered


def split_records(
    records: list[DistillationRecord],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[DistillationRecord], list[DistillationRecord]]:
    """Split records into train and validation sets."""
    # TODO:
    # 1. Validate validation_fraction is between 0 and 1.
    # 2. Shuffle a copy of records using seed.
    # 3. Split into train and validation records.
    # 4. Return both lists.
    if not (0 <= validation_fraction <= 1):
        raise ValueError("validation_fraction must be between 0 and 1.")
    shuffled_records = records.copy()
    random.Random(seed).shuffle(shuffled_records)
    split_index = int(len(shuffled_records) * (1 - validation_fraction))
    train_records = shuffled_records[:split_index]
    validation_records = shuffled_records[split_index:]
    
    return train_records, validation_records
    raise NotImplementedError("Implement split_records.")


def record_to_json_dict(record: DistillationRecord) -> dict:
    """Convert a DistillationRecord into a JSON-serializable dictionary."""
    # TODO:
    # 1. Convert the dataclass to a dictionary.
    # 2. Return the dictionary.
    DistillationRecord_dict = asdict(record)


    return DistillationRecord_dict

    raise NotImplementedError("Implement record_to_json_dict.")


def export_jsonl(records: list[DistillationRecord], path: str | Path) -> None:
    """Export records to a JSONL file."""
    # TODO:
    # 1. Open path for writing with UTF-8 encoding.
    # 2. Write one JSON object per line.
    # 3. Use record_to_json_dict for each record.
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            json_line = json.dumps(record_to_json_dict(record), ensure_ascii=False)
            f.write(json_line + "\n")



def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file into dictionaries."""
    # TODO:
    # 1. Open path with UTF-8 encoding.
    # 2. Read each non-empty line.
    # 3. Parse JSON and append to a list.
    # 4. Return the list.
    with open(path, "r", encoding="utf-8") as f:
        records = []
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
    raise NotImplementedError("Implement load_jsonl.")


def summarize_distillation_records(records: list[DistillationRecord]) -> str:
    """Create a readable summary of distillation records."""
    # TODO:
    # 1. Handle empty records.
    # 2. Include number of records.
    # 3. Include average input length.
    # 4. Include average prompt length.
    # 5. Include average target length. 
    empty_summary = "No records to summarize."
    if not records:
        return empty_summary

    num_records = len(records)
    avg_input_length = sum(len(r.input_ids) for r in records) / num_records
    avg_prompt_length = sum(r.prompt_length for r in records) / num_records
    avg_target_length = sum(len(r.labels) for r in records) / num_records

    summary = f"Summary of {num_records} distillation records:\n"
    summary += f"  Average input length: {avg_input_length:.2f}\n"
    summary += f"  Average prompt length: {avg_prompt_length:.2f}\n"
    summary += f"  Average target length: {avg_target_length:.2f}\n"

    return summary
    raise NotImplementedError("Implement summarize_distillation_records.")
