import os
import logging
import subprocess

def clean_wsl_memory():
    """Clean WSL memory by dropping caches."""
    logging.info("Cleaning WSL memory cache...")
    try:
        # This requires sudo access
        subprocess.run("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'", 
                      shell=True, check=False)
        logging.info("Memory cache cleared successfully")
    except Exception as e:
        logging.error(f"Failed to clear memory: {str(e)}")

def get_memory_info():
    """Get memory information in WSL."""
    free_output = subprocess.check_output(["free", "-h"]).decode()
    logging.info(f"Memory status:\n{free_output}")
    return free_output

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Memory status before cleaning:")
    get_memory_info()
    clean_wsl_memory()
    print("Memory status after cleaning:")
    get_memory_info() 