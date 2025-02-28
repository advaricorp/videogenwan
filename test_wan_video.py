#!/usr/bin/env python3
# test_wan_video.py - Standalone script to test Wan2.1 video generation

import os
import sys
import logging
import subprocess
import time
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("wan_test.log")
    ]
)

# Configuration
MODEL_SIZE = "1.3B"  # Use smaller model for first test
MODEL_REPO = f"Wan-AI/Wan2.1-T2V-{MODEL_SIZE}"
MODEL_DIR = os.path.abspath(f"./models/Wan2.1-T2V-{MODEL_SIZE}")
OUTPUT_DIR = os.path.abspath("./output/videos")
RESOLUTION = "832*480"  # Default for 1.3B model
WAN_DIR = os.path.abspath("./Wan2.1")

def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WAN_DIR, exist_ok=True)

def install_dependencies():
    """Install required dependencies for Wan2.1."""
    logging.info("Installing required dependencies...")
    
    try:
        # First install PyTorch - required for flash-attn
        logging.info("Installing PyTorch...")
        subprocess.run([
            "pip", "install", "torch>=2.0.0", "torchvision"
        ], check=True)
        
        # Now install other dependencies
        dependencies = [
            "transformers>=4.36.0",
            "diffusers>=0.25.0",
            "accelerate>=0.25.0",
            "safetensors>=0.4.0",
            "einops>=0.7.0",
            "tqdm",
            "pillow",
            "numpy",
            "av",
            "moviepy",
            "sentencepiece",
            "protobuf"
        ]
        
        for dep in dependencies:
            logging.info(f"Installing {dep}...")
            subprocess.run(["pip", "install", dep], check=True)
        
        # Install flash-attn
        logging.info("Installing flash-attn...")
        subprocess.run(["pip", "install", "flash-attn"], check=True)
        
        logging.info("All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error installing dependencies: {e}")
        return False

def download_model():
    """Download the Wan2.1 model if not already present."""
    if not os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        logging.info(f"Downloading model {MODEL_REPO}...")
        try:
            subprocess.run([
                "huggingface-cli", "download",
                MODEL_REPO,
                "--local-dir", MODEL_DIR
            ], check=True)
            logging.info("Model downloaded successfully")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error downloading model: {e}")
            return False
    else:
        logging.info(f"Model already exists at {MODEL_DIR}")
    return True

def setup_wan_repo():
    """Clone the Wan2.1 repository if not already present."""
    if not os.path.exists(os.path.join(WAN_DIR, "generate.py")):
        logging.info("Cloning Wan2.1 repository...")
        try:
            subprocess.run([
                "git", "clone",
                "https://github.com/Wan-Video/Wan2.1.git",
                WAN_DIR
            ], check=True)
            
            # Install dependencies
            logging.info("Installing Wan2.1 dependencies...")
            subprocess.run([
                "pip", "install", "-r",
                os.path.join(WAN_DIR, "requirements.txt")
            ], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error setting up Wan2.1 repository: {e}")
            return False
    else:
        logging.info("Wan2.1 repository already exists")
    return True

def apply_dashscope_patch():
    """Apply patch to bypass dashscope requirement."""
    prompt_extend_file = os.path.join(WAN_DIR, "wan", "utils", "prompt_extend.py")
    if os.path.exists(prompt_extend_file):
        logging.info("Applying dashscope bypass patch...")
        try:
            # Backup original file
            if not os.path.exists(prompt_extend_file + ".bak"):
                import shutil
                shutil.copy2(prompt_extend_file, prompt_extend_file + ".bak")
            
            # Replace with simplified version
            with open(prompt_extend_file, 'w') as f:
                f.write("""
# Simple prompt expander that doesn't require dashscope
import re

class DashScopePromptExpander:
    def __init__(self, api_key=None):
        pass
        
    def expand(self, prompt, max_length=150):
        # Just return the original prompt since we don't have dashscope
        return prompt

class QwenPromptExpander:
    def __init__(self, api_key=None, model="qwen-max"):
        pass
        
    def expand(self, prompt, max_length=150):
        # Just return the original prompt since we don't have QwenVL
        return prompt

def main():
    # Sample usage
    expander = DashScopePromptExpander()
    expanded = expander.expand("A cat")
    print(expanded)

if __name__ == "__main__":
    main()
""")
            logging.info("Patch applied successfully")
        except Exception as e:
            logging.error(f"Error applying patch: {e}")
            return False
    return True

def apply_flash_attn_patch():
    """Apply patch to handle flash-attn import issues."""
    attention_file = os.path.join(WAN_DIR, "wan", "modules", "attention.py")
    if os.path.exists(attention_file):
        logging.info("Checking flash-attn integration...")
        try:
            # Verify flash-attn is installed
            import flash_attn
            logging.info("flash-attn is properly installed, no patch needed")
            return True
        except ImportError:
            logging.warning("flash-attn not found, applying fallback patch...")
            
            # Backup original file
            if not os.path.exists(attention_file + ".bak"):
                import shutil
                shutil.copy2(attention_file, attention_file + ".bak")
            
            # Replace with simplified version
            with open(attention_file, 'w') as f:
                f.write("""
# Modified attention.py that doesn't require flash-attn
import torch
import torch.nn.functional as F

def flash_attention(q, k, v, mask=None, causal=False, window_size=(-1, -1)):
    # Fallback to standard attention
    q = q / (q.shape[-1] ** 0.5)
    attn = torch.matmul(q, k.transpose(-2, -1))
    
    if causal:
        # Apply causal mask
        batch_size, num_heads, seq_len, _ = attn.shape
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attn.device), diagonal=1
        ).bool()
        attn = attn.masked_fill(causal_mask, -float('inf'))
    
    if mask is not None:
        attn = attn.masked_fill(mask, -float('inf'))
    
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, v)

def flash_attention_varlen(q, k, v, mask=None, causal=False, window_size=(-1, -1)):
    return flash_attention(q, k, v, mask, causal, window_size)
""")
            logging.info("Flash-attn fallback patch applied successfully")
            return True
    return True

def generate_test_video():
    """Generate a test video using Wan2.1."""
    timestamp = int(time.time())
    save_file = f"test_video_{timestamp}"
    
    prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    
    logging.info(f"Generating test video with prompt: '{prompt}'")
    
    try:
        # Change to Wan2.1 directory to run the command
        original_dir = os.getcwd()
        os.chdir(WAN_DIR)
        
        # Make sure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        
        # Add memory optimization flags
        cmd = [
            "python", "generate.py",
            "--task", f"t2v-{MODEL_SIZE}",
            "--size", RESOLUTION,
            "--ckpt_dir", MODEL_DIR,
            "--offload_model", "True",
            "--t5_cpu",
            "--sample_shift", "8",
            "--sample_guide_scale", "6",
            "--sample_steps", "30",  # Reduce steps to save memory
            "--prompt", prompt,
            "--save_file", save_file
        ]
        
        # Set environment variables to optimize CUDA memory usage
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
        
        logging.info(f"Running command from {os.getcwd()}: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=env
        )
        
        logging.info("Command completed successfully")
        
        # Find the generated video file
        generated_file = None
        for file in os.listdir("outputs"):
            if file.startswith(save_file) and file.endswith(".mp4"):
                generated_file = os.path.join("outputs", file)
                break
        
        # Change back to original directory
        os.chdir(original_dir)
        
        if generated_file:
            # Get the full path to the generated file
            full_generated_path = os.path.join(WAN_DIR, generated_file)
            
            # Move the file to our output directory
            output_path = os.path.join(OUTPUT_DIR, os.path.basename(generated_file))
            shutil.move(full_generated_path, output_path)
            logging.info(f"Video generated successfully: {output_path}")
            return output_path
        else:
            logging.error("Could not find generated video file")
            return None
        
    except subprocess.CalledProcessError as e:
        # Change back to original directory in case of error
        os.chdir(original_dir)
        
        logging.error(f"Error generating video: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            logging.error(f"Command output: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            logging.error(f"Command error: {e.stderr}")
        logging.error("Failed to generate test video. This may be due to insufficient GPU memory in WSL. Consider running in a Docker container on a machine with more GPU memory.")
        return None
    except Exception as e:
        # Change back to original directory in case of error
        os.chdir(original_dir)
        
        logging.error(f"Unexpected error: {str(e)}")
        return None

def setup_only():
    """Setup environment without running the test."""
    logging.info("Setting up environment only...")
    
    # Setup environment
    ensure_directories()
    
    # Install dependencies
    if not install_dependencies():
        logging.error("Failed to install dependencies. Exiting.")
        return False
    
    # Download model
    if not download_model():
        logging.error("Failed to download model. Exiting.")
        return False
    
    # Setup Wan2.1 repository
    if not setup_wan_repo():
        logging.error("Failed to setup Wan2.1 repository. Exiting.")
        return False
    
    # Apply dashscope patch
    if not apply_dashscope_patch():
        logging.warning("Failed to apply dashscope patch. Continuing anyway.")
    
    # Apply flash-attn patch
    if not apply_flash_attn_patch():
        logging.warning("Failed to apply flash-attn patch. Continuing anyway.")
    
    logging.info("Setup completed successfully")
    return True

def main():
    logging.info("Starting Wan2.1 test script")
    
    # Setup environment
    ensure_directories()
    
    # Install dependencies
    if not install_dependencies():
        logging.error("Failed to install dependencies. Exiting.")
        return False
    
    # Download model
    if not download_model():
        logging.error("Failed to download model. Exiting.")
        return False
    
    # Setup Wan2.1 repository
    if not setup_wan_repo():
        logging.error("Failed to setup Wan2.1 repository. Exiting.")
        return False
    
    # Apply dashscope patch
    if not apply_dashscope_patch():
        logging.warning("Failed to apply dashscope patch. Continuing anyway.")
    
    # Apply flash-attn patch
    if not apply_flash_attn_patch():
        logging.warning("Failed to apply flash-attn patch. Continuing anyway.")
    
    # Generate test video
    output_path = generate_test_video()
    
    if output_path and os.path.exists(output_path):
        logging.info(f"SUCCESS! Test video generated at: {output_path}")
        return True
    else:
        logging.error("Failed to generate test video.")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        success = setup_only()
    else:
        success = main()
    sys.exit(0 if success else 1)