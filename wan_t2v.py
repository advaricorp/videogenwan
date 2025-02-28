import os
import sys
import logging
import torch
import subprocess
import requests
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4
import time
import random

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuración del modelo
MODEL_REPO = "Wan-AI/Wan2.1-T2V-14B"
MODEL_DIR = "./models/Wan2.1-T2V-14B"
VIDEO_OUTPUT_DIR = "output/videos"
DEFAULT_RESOLUTION = "832*480"  # Resolución por defecto para T2V-1.3B

# Variable global para rastrear si se ha aplicado el parche
_patch_applied = False

def apply_dashscope_patch(wan_repo_dir):
    """
    Aplica el parche para bypass de dashscope una sola vez.
    """
    global _patch_applied
    
    # Si ya aplicamos el parche antes, no hacerlo nuevamente
    if _patch_applied:
        return
    
    prompt_extend_file = os.path.join(wan_repo_dir, "wan", "utils", "prompt_extend.py")
    if os.path.exists(prompt_extend_file):
        # Verificar si ya hemos aplicado el parche
        try:
            with open(prompt_extend_file, 'r') as f:
                content = f.read()
            
            # Solo aplicar el parche si no parece estar ya aplicado
            if "DashScopePromptExpander" not in content or "# Simple prompt expander that doesn't require dashscope" not in content:
                logging.info("Aplicando parche para bypass dashscope...")
                try:
                    # Crear un backup del archivo original si no existe
                    backup_file = prompt_extend_file + ".bak"
                    if not os.path.exists(backup_file):
                        import shutil
                        shutil.copy2(prompt_extend_file, backup_file)
                    
                    # Reemplazar el contenido con una versión simplificada que no use dashscope
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
                    logging.info("Parche aplicado exitosamente.")
                    _patch_applied = True
                except Exception as e:
                    logging.error(f"Error aplicando parche: {e}")
            else:
                logging.info("El parche para dashscope ya está aplicado.")
                _patch_applied = True
        except Exception as e:
            logging.error(f"Error verificando archivo prompt_extend.py: {e}")
    else:
        logging.warning(f"Archivo prompt_extend.py no encontrado en {prompt_extend_file}")

def ensure_model_exists():
    """
    Asegura que el modelo esté descargado y disponible.
    """
    if not os.path.exists(MODEL_DIR):
        logging.info(f"El modelo no existe en {MODEL_DIR}, creando directorio...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        logging.info(f"Descargando modelo desde Hugging Face: {MODEL_REPO}")
        try:
            subprocess.run([
                "huggingface-cli", "download", 
                MODEL_REPO, 
                "--local-dir", MODEL_DIR
            ], check=True)
            logging.info("Modelo descargado exitosamente")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error descargando el modelo: {e}")
            raise ValueError(f"No se pudo descargar el modelo: {e}")

def setup_wan_environment():
    """
    Configura el entorno para ejecutar Wan2.1.
    """
    # Verificar si ya existe el directorio del repositorio
    wan_repo_dir = os.path.abspath("./Wan2.1")
    if not os.path.exists(wan_repo_dir) or not os.path.exists(os.path.join(wan_repo_dir, "generate.py")):
        logging.info(f"Repositorio Wan2.1 no encontrado o incompleto en {wan_repo_dir}, clonando...")
        
        # Si existe pero está incompleto, eliminarlo
        if os.path.exists(wan_repo_dir):
            logging.info("Eliminando directorio incompleto...")
            import shutil
            shutil.rmtree(wan_repo_dir)
        
        try:
            subprocess.run([
                "git", "clone", 
                "https://github.com/Wan-Video/Wan2.1.git", 
                wan_repo_dir
            ], check=True)
            
            # Instalar dependencias
            logging.info("Instalando dependencias del repositorio...")
            subprocess.run([
                "pip", "install", "-r", 
                os.path.join(wan_repo_dir, "requirements.txt")
            ], check=True)
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Error configurando el entorno Wan2.1: {e}")
            raise ValueError(f"No se pudo configurar el entorno Wan2.1: {e}")
    else:
        logging.info(f"Usando repositorio Wan2.1 existente en {wan_repo_dir}")
    
    # Asegurar que las dependencias críticas están instaladas
    try:
        # Lista de dependencias críticas que podrían no estar en requirements.txt del repo
        critical_deps = [
            "easydict",
            "omegaconf", 
            "ftfy", 
            "regex", 
            "timm", 
            "decord", 
            "kornia"
        ]
        
        for dep in critical_deps:
            try:
                # Intentar importar para verificar si está instalado
                __import__(dep)
                logging.info(f"Dependencia {dep} ya está instalada")
            except ImportError:
                logging.info(f"Instalando dependencia faltante: {dep}")
                subprocess.run([
                    "pip", "install", dep
                ], check=True)
        
    except Exception as e:
        logging.warning(f"Error verificando/instalando dependencias: {e}")
        # Intentar continuar a pesar del error
    
    # Aplicamos el parche para dashscope una sola vez (función separada)
    apply_dashscope_patch(wan_repo_dir)
    
    # Asegurar que el modelo existe
    ensure_model_exists()
    
    # Crear directorio de salida de videos
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    
    return wan_repo_dir

# Aplicar el parche al cargar el módulo
wan_repo_dir = os.path.abspath("./Wan2.1")
if os.path.exists(wan_repo_dir):
    apply_dashscope_patch(wan_repo_dir)

def model_files_exist(repo_dir, model_size="1.3B"):
    """Check if all required model files exist."""
    model_dir = os.path.join(repo_dir, f"Wan2.1-T2V-{model_size}")
    
    # Check for essential files
    required_files = [
        "models_t5_umt5-xxl-enc-bf16.pth",
        "models_vae_vae-enc-bf16.pth",
        "models_vae_vae-dec-bf16.pth"
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            return False
    
    return True

def download_model_using_cli(model_size="1.3B"):
    """Download model using the huggingface-cli."""
    try:
        # Install dependencies if needed
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"], check=True)
        
        # Determine target directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "models", f"Wan2.1-T2V-{model_size}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Run the command from README
        cmd = [
            "huggingface-cli", "download", 
            f"Wan-AI/Wan2.1-T2V-{model_size}", 
            "--local-dir", model_dir
        ]
        
        logging.info(f"Ejecutando comando de descarga recomendado: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logging.info(f"Descarga completada usando huggingface-cli")
        
        return model_dir
    except Exception as e:
        logging.error(f"Error usando huggingface-cli: {str(e)}")
        return None

def check_model_files(model_dir, model_size="1.3B"):
    """Check if all required model files exist based on the actual file structure."""
    if not os.path.exists(model_dir):
        logging.warning(f"Model directory {model_dir} does not exist")
        return False
    
    # List files in directory
    files = os.listdir(model_dir)
    logging.info(f"Found {len(files)} files in {model_dir}")
    
    # Required files for Wan2.1 models
    t5_found = False
    vae_found = False
    diffusion_found = False
    config_found = False
    
    # Check for T5 encoder file
    if "models_t5_umt5-xxl-enc-bf16.pth" in files:
        t5_found = True
    
    # Check for VAE file
    if "Wan2.1_VAE.pth" in files:
        vae_found = True
    
    # Check for diffusion model file
    if "diffusion_pytorch_model.safetensors" in files:
        diffusion_found = True
    
    # Check for config file
    if "config.json" in files:
        config_found = True
    
    # Log what we found and what's missing
    if not t5_found:
        logging.warning(f"Missing T5 encoder file in {model_dir}")
    if not vae_found:
        logging.warning(f"Missing VAE file in {model_dir}")
    if not diffusion_found:
        logging.warning(f"Missing diffusion model file in {model_dir}")
    if not config_found:
        logging.warning(f"Missing config.json in {model_dir}")
    
    # Model is complete if we have all required files
    is_complete = t5_found and vae_found and diffusion_found and config_found
    
    if is_complete:
        logging.info(f"Model {model_size} is complete at {model_dir}")
    else:
        logging.warning(f"Model {model_size} is incomplete at {model_dir}")
    
    return is_complete

def get_available_models():
    """Get a list of available models and their paths."""
    result = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check models in the standard directories
    model_dirs = [
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "Wan2.1")
    ]
    
    # Check for 1.3B model
    for base_path in model_dirs:
        path = os.path.join(base_path, "Wan2.1-T2V-1.3B")
        if os.path.exists(path) and check_model_files(path, "1.3B"):
            result["1.3B"] = path
            logging.info(f"Model 1.3B found at {path}")
            break
    
    # Check for 14B model
    for base_path in model_dirs:
        path = os.path.join(base_path, "Wan2.1-T2V-14B")
        if os.path.exists(path) and check_model_files(path, "14B"):
            result["14B"] = path
            logging.info(f"Model 14B found at {path}")
            break
    
    # Log if models are not found
    if "1.3B" not in result:
        logging.warning("Model 1.3B not found")
    if "14B" not in result:
        logging.warning("Model 14B not found")
    
    return result

def generate_video_with_wan(prompt, model_size="1.3B", resolution="832*480", duration=5, output_path=None):
    """Generate a video using the Wan2.1 model."""
    try:
        logging.info(f"Generating video with prompt: {prompt}")
        logging.info(f"Requested model size: {model_size}, Resolution: {resolution}, Duration: {duration} sec")
        
        # Ensure repository exists
        ensure_wan_repo_exists()
        
        # Get available models
        available_models = get_available_models()
        
        # Check if requested model is available
        if model_size not in available_models:
            logging.warning(f"Model {model_size} not found. Attempting to download...")
            download_model_using_cli(model_size)
            
            # Refresh available models
            available_models = get_available_models()
        
        # If requested model is still not available, use alternative
        if model_size not in available_models:
            alternative_size = "14B" if model_size == "1.3B" else "1.3B"
            if alternative_size in available_models:
                logging.warning(f"Using alternative model {alternative_size} instead of requested {model_size}")
                model_size = alternative_size
            else:
                logging.warning("No Wan2.1 models found. Creating placeholder video.")
                if output_path is None:
                    output_path = generate_unique_output_path()
                return generate_placeholder_video(
                    output_path, 
                    f"No Wan2.1 models available.\nPlease download model files from https://huggingface.co/Wan-AI/Wan2.1-T2V-{model_size}"
                )
        
        # Set model path
        model_path = available_models[model_size]
        
        # Generate unique output path if not provided
        if output_path is None:
            output_path = generate_unique_output_path()
        
        # Get generate.py path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        generate_script = os.path.join(base_dir, "Wan2.1", "generate.py")
        
        if not os.path.exists(generate_script):
            logging.error(f"Generate script not found at {generate_script}")
            return generate_placeholder_video(output_path, "Generate script not found.")
        
        # Prepare command with CORRECT arguments format and values
        command = [
            "python", generate_script,
            "--task", f"t2v-{model_size}",
            "--size", resolution,
            "--ckpt_dir", model_path,
            "--prompt", prompt,
            "--save_file", output_path,
            "--t5_cpu"
        ]
        
        # Add specific options for model size using CORRECT values from README
        if model_size == "1.3B":
            command.extend([
                "--offload_model", "True",
                "--sample_shift", "8",  # FIXED: Using numeric value 8 as per README
                "--sample_guide_scale", "6",  # ADDED: Recommended value from README
                "--sample_steps", "30"
            ])
        else:  # For 14B model
            command.extend([
                "--offload_model", "True"
            ])
        
        # Execute command
        logging.info(f"Executing command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        
        if result.returncode == 0:
            logging.info(f"Video generated successfully: {output_path}")
            return output_path
        else:
            logging.error(f"Error generating video: {result.stderr}")
            return generate_placeholder_video(
                output_path, 
                f"Error generating video:\n{result.stderr[:200]}..."
            )
    
    except Exception as e:
        logging.exception(f"Exception in generate_video_with_wan: {str(e)}")
        if output_path is None:
            output_path = generate_unique_output_path()
        return generate_placeholder_video(
            output_path, 
            f"Exception in video generation:\n{str(e)}"
        )

def generate_placeholder_video(output_path, message="Video no disponible"):
    """Genera un video de placeholder con un mensaje."""
    try:
        from moviepy.editor import TextClip, ColorClip, CompositeVideoClip
        
        # Crear un clip de fondo negro
        bg_clip = ColorClip(size=(832, 480), color=(0, 0, 0), duration=5)
        
        # Crear el texto con el mensaje
        text_clip = TextClip(message, fontsize=30, color='white', bg_color='transparent', 
                            size=(700, 400), method='caption').set_position('center').set_duration(5)
        
        # Combinar los clips correctamente
        final_clip = CompositeVideoClip([bg_clip, text_clip])
        
        # Guardar el video
        final_clip.write_videofile(output_path, fps=24, codec='libx264', audio=False)
        
        return output_path
    except Exception as e:
        logging.exception(f"Error al crear video placeholder: {e}")
        try:
            # Método alternativo simple sin moviepy
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Crear un archivo vacío
            with open(output_path, 'wb') as f:
                # Escribir un archivo MP4 mínimo (bytes de cabecera)
                f.write(b'\x00\x00\x00\x20\x66\x74\x79\x70\x6d\x70\x34\x32')
            return output_path
        except:
            # Si todo falla, intentar crear un archivo vacío
            try:
                with open(output_path, 'w') as f:
                    pass
                return output_path
            except:
                return None

def generate_unique_output_path():
    """Generate a unique output path for videos."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = int(time.time())
    return os.path.join(base_dir, "static", "videos", f"output_{timestamp}.mp4")

def ensure_wan_repo_exists():
    """Ensure Wan2.1 repository exists and is properly set up."""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        repo_dir = os.path.join(base_dir, "Wan2.1")
        
        # Check if repo directory exists
        if not os.path.exists(repo_dir):
            logging.info(f"Creating Wan2.1 repository directory: {repo_dir}")
            os.makedirs(repo_dir, exist_ok=True)
            
            # Clone the repository
            logging.info("Cloning Wan2.1 repository...")
            subprocess.run(["git", "clone", "https://github.com/Wan-Video/Wan2.1.git", repo_dir], check=True)
        
        # Check if generate.py exists
        if not os.path.exists(os.path.join(repo_dir, "generate.py")):
            raise Exception("Wan2.1 repository missing generate.py script")
        
        logging.info("Wan2.1 repository is properly set up")
        return True
    except Exception as e:
        logging.exception(f"Error ensuring Wan2.1 repository: {str(e)}")
        return False

def batch_generate_videos(prompts, model_size="1.3B", resolution="832*480", duration=5):
    """
    Genera múltiples videos a partir de una lista de prompts.
    
    Args:
        prompts (list): Lista de textos para generar videos.
        model_size (str): Tamaño del modelo a usar ('1.3B' o '14B').
        resolution (str): Resolución del video ('832*480' o '1280*720').
        duration (int): Duración aproximada de cada video en segundos.
        
    Returns:
        list: Lista de rutas de los videos generados.
    """
    videos = []
    logging.info(f"Iniciando generación de {len(prompts)} videos con Wan2.1 {model_size}")
    
    # Verificar disponibilidad de modelos
    available_models = get_available_models()
    
    # Si el modelo solicitado no está disponible, intentar usar alternativas
    if model_size not in available_models:
        alternative_size = "14B" if model_size == "1.3B" else "1.3B"
        if alternative_size in available_models:
            logging.warning(f"Modelo {model_size} no disponible. Usando alternativa: {alternative_size}")
            model_size = alternative_size
        else:
            logging.error("No hay modelos Wan2.1 disponibles")
            # Generar videos de placeholder en caso de no tener modelos
            return [generate_placeholder_video(
                generate_unique_output_path(),
                f"No hay modelos Wan2.1 disponibles.\nPor favor descarga los archivos desde HuggingFace."
            ) for _ in prompts]
    
    # Generar cada video
    for i, prompt in enumerate(prompts):
        try:
            logging.info(f"Generando video {i+1}/{len(prompts)}")
            output_path = generate_video_with_wan(
                prompt=prompt,
                model_size=model_size,
                resolution=resolution,
                duration=duration
            )
            videos.append(output_path)
            logging.info(f"Video {i+1} generado: {output_path}")
        except Exception as e:
            logging.error(f"Error generando video {i+1}: {str(e)}")
            # Crear un video de placeholder en caso de error
            placeholder = generate_placeholder_video(
                generate_unique_output_path(),
                f"Error al generar video: {str(e)}"
            )
            videos.append(placeholder)
    
    return videos

if __name__ == "__main__":
    # Ejemplo de uso
    prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    video_path = generate_video_with_wan(prompt)
    print(f"Video generado: {video_path}") 