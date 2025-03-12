import torch
import math

def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    """Process long sequences by splitting them into chunks."""
    
    # Constants
    max_len = 512
    stride = 256
    
    # Get sequence length
    n = input_ids.size(0)
    length = input_ids.size(1)
    
    # If sequence is short enough, process normally
    if length <= max_len:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        return output.last_hidden_state, output.attentions

    # Process long sequence in chunks
    stride = min(stride, max_len)
    
    # Calculate number of chunks needed
    chunks = math.ceil((length - max_len) / stride) + 1
    
    # Initialize output tensors
    hidden_states = torch.zeros(
        n, length, model.config.hidden_size, 
        dtype=torch.float, 
        device=input_ids.device
    )
    
    # Process each chunk
    for i in range(chunks):
        start = min(i * stride, length - max_len)
        end = start + max_len
        
        # Get chunk of input
        chunk_input_ids = input_ids[:, start:end]
        chunk_attention_mask = attention_mask[:, start:end]
        
        # Process chunk
        outputs = model(
            input_ids=chunk_input_ids,
            attention_mask=chunk_attention_mask,
            output_attentions=True,
        )
        
        # Store chunk outputs
        chunk_start = i * stride
        chunk_end = min(chunk_start + max_len, length)
        hidden_states[:, chunk_start:chunk_end] = outputs.last_hidden_state[:, :chunk_end-chunk_start]
    
    return hidden_states, None  # Return None for attentions as they're not needed for long sequences 