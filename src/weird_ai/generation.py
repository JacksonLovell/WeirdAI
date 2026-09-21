"""
Text generation utilities for Weird AI.

This module contains helper functions that convert between text and token IDs
and generate new text from a trained or partially trained model.
"""

import torch


def text_to_token_ids(text, tokenizer):
    """
    Convert text into a tensor of token IDs.

    Args:
        text: The input prompt as a string.
        tokenizer: The tokenizer object.

    Returns:
        A tensor of shape (1, num_tokens).
    """

    # TODO:
    # 1. Use the tokenizer to encode the text.
    # 2. Convert the encoded list into a torch tensor.
    # 3. Add a batch dimension using unsqueeze(0).
    token_ids = tokenizer.encode(text)
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)
    batch_dimension = token_ids_tensor.unsqueeze(0)
    return torch.tensor(batch_dimension, dtype=torch.long)




def token_ids_to_text(token_ids, tokenizer):
    """
    Convert token IDs back into text.

    Args:
        token_ids: Tensor of token IDs.
        tokenizer: The tokenizer object.

    Returns:
        The decoded text as a string.
    """

    # TODO:
    # 1. Remove the batch dimension if present.
    # 2. Convert the tensor to a Python list.
    # 3. Use the tokenizer to decode the list.
    batch_removed = token_ids.squeeze(0)
    token_ids_list = batch_removed.tolist()
    token_ids_text = tokenizer.decode(token_ids_list)
    return token_ids_text


    raise NotImplementedError("Implement token_ids_to_text.")


def generate_text_simple(model, input_ids, max_new_tokens, context_size):
    """
    Generate text one token at a time using greedy decoding.

    Args:
        model: The Weird AI model.
        input_ids: Tensor of shape (batch_size, num_tokens).
        max_new_tokens: Number of new tokens to generate.
        context_size: Maximum number of tokens the model can consider.

    Returns:
        Tensor containing the original input IDs plus generated token IDs.
    """

    # TODO:
    # Repeat max_new_tokens times:
    # 1. Crop input_ids to the most recent context_size tokens.
    # 2. Pass the cropped input into the model to get logits.
    # 3. Select only the logits for the last time step.
    # 4. Use argmax to choose the most likely next token.
    # 5. Append that token to input_ids.
    input_ids = input_ids.to(model.device)
    for _ in range(max_new_tokens):
        cropped_input = input_ids[:, -context_size:]
        logits = model(cropped_input)
        last_logits = logits[:, -1, :]
        next_token_id = torch.argmax(last_logits, dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token_id), dim=1)

    

    raise NotImplementedError("Implement generate_text_simple.")


def generate_and_print_sample(model, tokenizer, device, start_context, context_size, max_new_tokens=50):
    """
    Generate and print a sample text output.

    This is useful during training so students can visually inspect whether
    the model is improving.

    Args:
        model: The Weird AI model.
        tokenizer: The tokenizer object.
        device: CPU or CUDA device.
        start_context: Prompt text.
        context_size: Maximum number of tokens the model can consider.
        max_new_tokens: Number of tokens to generate.
    """

    model.eval()

    # TODO:
    # 1. Convert start_context to token IDs.
    # 2. Move token IDs to the selected device.
    # 3. Generate new token IDs.
    # 4. Convert generated token IDs back to text.
    # 5. Print the generated text.
    start_context_ids = text_to_token_ids(start_context, tokenizer).to(device)
    generated_ids = generate_text_simple(model, start_context_ids, max_new_tokens, context_size
    )
    generated_text = token_ids_to_text(generated_ids, tokenizer)
    print(f"Generated text:\n{generated_text}")

 