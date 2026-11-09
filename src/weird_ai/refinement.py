"""
Self-refinement utilities for Weird AI.

This module connects Chapter 5 of Build a Reasoning Model from Scratch to the
Weird AI project.

Chapter 4 added best-of-N generation:

    generate several candidates -> evaluate each -> select the best

Chapter 5 adds self-refinement:

    generate one candidate -> evaluate -> ask for a revision -> evaluate again
    -> keep the better version -> repeat

The goal is to improve output quality during inference without retraining the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


GenerationFunction = Callable[..., str]
EvaluationFunction = Callable[[str], dict]


@dataclass
class RefinementStep:
    """Store information about one refinement attempt."""

    iteration: int
    prompt: str
    previous_text: str
    revised_text: str
    previous_score: float
    revised_score: float
    accepted: bool
    evaluation: dict = field(default_factory=dict)


def get_score(evaluation: dict, score_key: str = "overall_score") -> float:
    """Extract a numeric score from an evaluation dictionary."""

    # TODO:
    # 1. Read score_key from the evaluation dictionary.
    # 2. Convert it to a float.
    # 3. If the key is missing or cannot be converted, return 0.0.
    score = evaluation.get(score_key)
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0
    


def build_refinement_prompt(
    original_prompt: str,
    current_text: str,
    evaluation: dict,
    *,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    score_key: str = "overall_score",
) -> str:
    """Build a prompt asking the model to improve a generated parody."""

    # TODO:
    # 1. Include the original prompt or topic.
    # 2. Include the current lyrics.
    # 3. Include the current overall score.
    # 4. Include any available evaluation details, such as rhyme, syllables, and structure.
    # 5. Include focus items, such as stronger rhymes or more consistent syllables.
    # 6. Include preserve items, such as topic, tone, and approximate length.
    # 7. Ask the model to return only the revised lyrics.
    #
    # Keep the prompt specific. "Make it better" is too vague.
    
    score = get_score(evaluation, score_key)
    evaluation_details = "\n".join(
        f"- {key}: {value}"
        for key, value in evaluation.items()
        if key != score_key
    )
    focus_items = "\n".join(f"- {item}" for item in focus or [])
    preserve_items = "\n".join(f"- {item}" for item in preserve or [])

    return (
        f"Original prompt: {original_prompt}\n\n"
        f"Current lyrics:\n{current_text}\n\n"
        f"Current score: {score}\n"
        f"Evaluation details:\n{evaluation_details or '- None'}\n\n"
        f"Focus on:\n{focus_items or '- Improve the overall quality of the lyrics.'}\n"
        f"Preserve:\n{preserve_items or '- The original topic, tone, and approximate length.'}\n\n"
        "Return only the revised lyrics."
    )


def refine_once(
    original_prompt: str,
    current_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> RefinementStep:
    """Run one self-refinement step."""

    # TODO:
    # 1. Evaluate current_text.
    # 2. Build a refinement prompt using build_refinement_prompt.
    # 3. Generate revised text with generation_func.
    # 4. Evaluate the revised text.
    # 5. Compare the old score and revised score.
    # 6. Return a RefinementStep.
    #
    # This function should not decide whether to continue the full loop.
    # It only performs one attempt.

    evaluation = evaluation_func(current_text)
    refinement_prompt = build_refinement_prompt(
        original_prompt,
        current_text,
        evaluation,
        focus=focus,
        preserve=preserve,
        score_key=score_key,
    )
    revised_text = generation_func(
        refinement_prompt,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens
    )
    revised_evaluation = evaluation_func(revised_text)
    return RefinementStep(
        iteration=1,
        prompt=refinement_prompt,
        previous_text=current_text,
        revised_text=revised_text,
        evaluation=evaluation,
        previous_score=get_score(evaluation, score_key),
        revised_score=get_score(revised_evaluation, score_key),
        accepted=get_score(revised_evaluation, score_key) > get_score(evaluation, score_key),
    )

def choose_better_text(
    current_text: str,
    revised_text: str,
    current_evaluation: dict,
    revised_evaluation: dict,
    *,
    score_key: str = "overall_score",
    require_improvement: bool = True,
) -> tuple[str, dict, bool]:
    """Decide whether to accept a revised text."""

    # TODO:
    # 1. Compare the current score and revised score.
    # 2. If require_improvement is True, accept only strictly higher scores.
    # 3. If require_improvement is False, accept equal scores as well.
    # 4. Return the selected text, selected evaluation, and whether the revision was accepted.

    current_score = get_score(current_evaluation, score_key)
    revised_score = get_score(revised_evaluation, score_key)
    improved = (
        revised_score > current_score
        if require_improvement
        else revised_score >= current_score
    )
    if improved:
        return revised_text, revised_evaluation, True
    return current_text, current_evaluation, False


def should_continue(
    *,
    iteration: int,
    max_iterations: int,
    current_score: float,
    target_score: Optional[float] = None,
    last_improved: bool = True,
    stop_on_no_improvement: bool = True,
) -> bool:
    """Decide whether the refinement loop should continue."""

    # TODO:
    # 1. Stop if iteration has reached max_iterations.
    # 2. Stop if target_score is set and current_score is high enough.
    # 3. Stop if stop_on_no_improvement is True and last_improved is False.
    # 4. Otherwise continue.

    if iteration >= max_iterations:
        return False
    if target_score is not None and current_score >= target_score:
        return False
    if stop_on_no_improvement and not last_improved:
        return False
    return True


def iterative_refinement(
    original_prompt: str,
    initial_text: str,
    generation_func: GenerationFunction,
    evaluation_func: EvaluationFunction,
    *,
    max_iterations: int = 3,
    target_score: Optional[float] = None,
    temperature: float = 0.7,
    top_p: Optional[float] = 0.9,
    max_new_tokens: int = 160,
    score_key: str = "overall_score",
    stop_on_no_improvement: bool = True,
    require_improvement: bool = True,
    focus: Optional[list[str]] = None,
    preserve: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict:
    """Run the full self-refinement loop."""

    # TODO:
    # 1. Evaluate initial_text.
    # 2. Repeatedly call refine_once until should_continue says to stop.
    # 3. After each step, keep the better text using choose_better_text.
    # 4. Track every RefinementStep in a list.
    # 5. Return a result dictionary with the initial and final information.
    #
    # Be careful to avoid infinite loops.

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    current_text = initial_text
    current_evaluation = evaluation_func(current_text)
    current_score = get_score(current_evaluation, score_key)
    initial_score = current_score

    steps: list[RefinementStep] = []
    stopped_reason = "max_iterations"

    if target_score is not None and current_score >= target_score:
        return {
            "initial_text": initial_text,
            "initial_score": initial_score,
            "final_text": current_text,
            "final_score": current_score,
            "iterations": steps,
            "stopped_reason": "target_score",
        }

    for iteration in range(1, max_iterations + 1):
        step = refine_once(
            original_prompt,
            current_text,
            generation_func,
            evaluation_func,
            iteration=iteration,
            current_evaluation=current_evaluation,
            require_improvement=require_improvement,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            score_key=score_key,
            focus=focus,
            preserve=preserve,
            **kwargs,
        )
        steps.append(step)

        if step.accepted:
            current_text = step.revised_text
            current_evaluation = step.evaluation
            current_score = step.revised_score

        if not should_continue(
            iteration=iteration,
            max_iterations=max_iterations,
            current_score=current_score,
            target_score=target_score,
            last_improved=step.accepted,
            stop_on_no_improvement=stop_on_no_improvement,
        ):
            if target_score is not None and current_score >= target_score:
                stopped_reason = "target_score"
            elif stop_on_no_improvement and not step.accepted:
                stopped_reason = "no_improvement"
            else:
                stopped_reason = "max_iterations"
            break

    return {
        "initial_text": initial_text,
        "initial_score": initial_score,
        "final_text": current_text,
        "final_score": current_score,
        "iterations": steps,
        "stopped_reason": stopped_reason,
    }


def summarize_refinement(result: dict) -> str:
    """Create a readable summary of a refinement run."""

    # TODO:
    # 1. Include the initial score and final score.
    # 2. Include the number of iterations.
    # 3. Include whether each iteration was accepted or rejected.
    # 4. Include the stopped reason.
    #
    # This is useful for debugging and for showing the tradeoff between extra
    # inference compute and output improvement.

   
    lines = "\n".join(
        f"  {s.iteration}. {'accepted' if s.accepted else 'rejected':<8} "
        f"{s.previous_score:.2f} -> {s.revised_score:.2f}"
        for s in result["iterations"]
    )

    return (
        "Refinement summary\n"
        "------------------\n"
        f"Initial score:  {result['initial_score']:.2f}\n"
        f"Final score:    {result['final_score']:.2f} "
        f"({result['final_score'] - result['initial_score']:+.2f})\n"
        f"Iterations:     {len(result['iterations'])}\n"
        f"Stopped reason: {result['stopped_reason']}\n\n"
        f"Steps:\n{lines}"
    )
