# Save as complete_flash_patch.py
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_complete_flash_attention_patch():
    """Create a complete patch that replaces Flash Attention with standard attention"""
    # Get the absolute path to ensure we find the file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    attention_file = os.path.join(base_dir, "Wan2.1", "wan", "modules", "attention.py")
    
    if not os.path.exists(attention_file):
        logging.error(f"Attention file not found at {attention_file}")
        logging.info(f"Current directory is: {os.getcwd()}")
        logging.info(f"Checking if Wan2.1 directory exists: {os.path.exists(os.path.join(base_dir, 'Wan2.1'))}")
        return False
    
    logging.info(f"Found attention file at: {attention_file}")
    
    # Read the current file content
    try:
        with open(attention_file, 'r') as f:
            content = f.read()
        logging.info("Successfully read the original attention.py file")
    except Exception as e:
        logging.error(f"Error reading attention file: {str(e)}")
        return False
    
    # Create backup
    backup_file = attention_file + ".bak"
    if not os.path.exists(backup_file):
        try:
            with open(backup_file, 'w') as f:
                f.write(content)
            logging.info(f"Created backup at {backup_file}")
        except Exception as e:
            logging.error(f"Error creating backup file: {str(e)}")
            return False
    
    # Create a completely new implementation of flash_attention
    new_content = """
# Modified attention.py with standard attention fallback

import os
import warnings
import torch
import torch.nn.functional as F
from typing import Optional

# Define constants
FLASH_ATTN_2_AVAILABLE = False  # Set to False to use fallback
XFORMERS_AVAILABLE = False  # We're not using xformers

def flash_attention(q, k, v, mask=None):
    """
    Standard attention implementation as fallback for Flash Attention
    """
    # Compute attention scores
    attn_weights = torch.matmul(q, k.transpose(-2, -1))
    
    # Scale attention scores
    scale = 1.0 / (q.size(-1) ** 0.5)
    attn_weights = attn_weights * scale
    
    # Apply mask if provided
    if mask is not None:
        attn_weights = attn_weights + mask
    
    # Apply softmax to get attention probabilities
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    # Get output by applying attention weights to values
    output = torch.matmul(attn_weights, v)
    
    return output

def flash_attn_func(q, k, v, mask=None, dropout_p=0.0, causal=False):
    """Compatibility wrapper for flash_attention"""
    return flash_attention(q, k, v, mask)

def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, causal=False, return_attn_probs=False):
    """
    Fallback implementation for variable length flash attention
    """
    # For simplicity in the fallback, we'll just reshape to 3D batched attention
    # This is a simplification and may not work for all cases
    batch_size = cu_seqlens_q.size(0) - 1
    device = q.device
    
    # Create an output tensor to fill
    out = torch.zeros_like(q)
    
    # Process each batch item separately
    for i in range(batch_size):
        # Get the start and end indices for this batch
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i+1].item()
        k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i+1].item()
        
        # Extract the sequences for this batch
        q_seq = q[q_start:q_end]
        k_seq = k[k_start:k_end]
        v_seq = v[k_start:k_end]
        
        # Create causal mask if needed
        mask_seq = None
        if causal:
            q_len = q_end - q_start
            k_len = k_end - k_start
            mask_seq = torch.triu(
                torch.ones(q_len, k_len, device=device) * float("-inf"), 
                diagonal=1
            )
        
        # Compute attention and store in output tensor
        out_seq = flash_attention(q_seq, k_seq, v_seq, mask_seq)
        out[q_start:q_end] = out_seq
    
    return out
"""
    
    # Write the new content
    try:
        with open(attention_file, 'w') as f:
            f.write(new_content)
        logging.info("Complete Flash Attention replacement patch applied successfully")
    except Exception as e:
        logging.error(f"Error writing new attention file: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    print("Starting to apply Flash Attention patch...")
    if apply_complete_flash_attention_patch():
        print("✅ Complete Flash Attention replacement patch applied successfully")
    else:
        print("❌ Failed to apply Flash Attention patch")
        sys.exit(1)  # Exit with error code
    print("Patch application complete. Please restart your Flask app.")