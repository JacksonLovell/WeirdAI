"""
Instruction data formatting utilities for Weird AI.
"""





def format_input(entry):
    """
    Format one instruction example using an Alpaca-style prompt.
    This function does NOT include the response.
    """

    # TODO:
    # 1. Build the standard instruction text.
    # 2. Add the ### Instruction section.
    # 3. Add the ### Input section only when entry["input"] is not empty.
    # 4. Return the complete prompt.
    instruction = entry["instruction"]
    input_text = entry["input"]

    prompt = f"### Instruction:\n{instruction}\n\n"
    if input_text:
        prompt += f"### Input:\n{input_text}\n\n"

    return prompt


def format_response(entry):
    """
    Format the expected response section.
    """

    # TODO:
    # Return a string like:
    # "\n\n### Response:\n..."
    return f"\n\n### Response:\n{entry['output']}"
    raise NotImplementedError("Implement format_response.")


def format_full_example(entry):
    """
    Format an entire instruction-response example.
    """

    # TODO:
    # Combine format_input(entry) and format_response(entry).
    return format_input(entry) + format_response(entry)

    raise NotImplementedError("Implement format_full_example.")


def validate_instruction_entry(entry):
    """
    Validate that an instruction dataset entry has instruction, input, and output fields.
    """

    # TODO:
    # Check for instruction, input, and output keys.
    # Verify that instruction and output are not empty.
    if not all(key in entry for key in ["instruction", "input", "output"]):
        return False

    if not entry["instruction"] or not entry["output"]:
       return False

    return True 
