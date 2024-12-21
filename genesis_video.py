import os
import re
import asyncio
import logging
import requests
import time
import json
from textwrap import wrap
from uuid import uuid4
from edge_tts import Communicate
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    TextClip, ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips, VideoFileClip, vfx
)
import random

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuraciones globales
OUTPUT_DIR = "output"
TEMP_DIR = "temp_files"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "final_video.mp4")

# Configuración de voces disponibles
VOICES = [
    {"id": "es-ES-AlvaroNeural", "name": "Alvaro (ES)", "gender": "Male"},
    {"id": "es-MX-JorgeNeural", "name": "Jorge (MX)", "gender": "Male"},
    {"id": "es-ES-ElviraNeural", "name": "Elvira (ES)", "gender": "Female"},
    {"id": "en-US-ChristopherNeural", "name": "Christopher (US)", "gender": "Male"},
    {"id": "en-GB-RyanNeural", "name": "Ryan (UK)", "gender": "Male"},
    {"id": "en-US-JennyNeural", "name": "Jenny (US)", "gender": "Female"},
    {"id": "fr-FR-HenriNeural", "name": "Henri (FR)", "gender": "Male"},
    {"id": "fr-CA-AntoineNeural", "name": "Antoine (CA)", "gender": "Male"},
    {"id": "fr-FR-DeniseNeural", "name": "Denise (FR)", "gender": "Female"}
]

# Configuración de efectos visuales disponibles
EFFECTS = [
    'zoom_in',
    'zoom_out',
    'pan_left_to_right',
    'pan_right_to_left',
    'ken_burns',
    'random'
]

# Otras configuraciones
MAX_TOKENS_SD = 75
MAX_CHARS_PROMPT = 200
MAX_CHARS_SUBTITLE = 50
NEGATIVE_PROMPT = "realistic human, cartoon, anime, low quality, blurry, text, watermark, bad anatomy, bad proportions, deformed"
FPS = 24
DEFAULT_MASTER_PROMPT = "Imagen fotorrealista, alta calidad, iluminación cinematográfica de:"

# Estilos de subtítulos
SUBTITLE_STYLES = {
    'default': {
        'font': 'Arial',
        'fontsize': 40,
        'color': 'white',
        'stroke_color': 'black',
        'stroke_width': 2,
        'bg_color': 'rgba(0,0,0,0.5)',
        'method': 'caption'
    }
}

# Crear carpetas necesarias
for directory in [OUTPUT_DIR, TEMP_DIR, AUDIO_DIR, IMAGE_DIR]:
    os.makedirs(directory, exist_ok=True)

def clean_output_dirs():
    """Limpia los directorios de salida antes de generar nuevo contenido"""
    logging.info("Limpiando directorios de salida...")
    try:
        for dir_path in [IMAGE_DIR, AUDIO_DIR]:
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logging.error(f"Error al eliminar {file_path}: {e}")
        
        if os.path.exists(OUTPUT_VIDEO_PATH):
            os.unlink(OUTPUT_VIDEO_PATH)
        logging.info("Directorios limpiados correctamente")
    except Exception as e:
        logging.error(f"Error al limpiar directorios: {e}")
        raise

def generate_random_name(extension="jpg"):
    return f"{uuid4().hex}.{extension}"

def split_text_into_sentences(text):
    """
    Divide el texto en oraciones usando puntos y punto y coma como separadores.
    Maneja diálogos y puntuación especial.
    """
    # Normalizar puntuación
    text = text.replace(';', '.')
    text = text.replace('...', '***ELLIPSIS***')
    text = text.replace('".', '"***POINT***.')
    text = text.replace('?.', '?***POINT***.')
    text = text.replace('!.', '!***POINT***.')
    
    # Dividir por puntos
    sentences = []
    for part in text.split('.'):
        part = part.strip()
        if part:
            # Restaurar puntuación especial
            part = part.replace('***ELLIPSIS***', '...')
            part = part.replace('***POINT***.', '.')
            sentences.append(part)
    
    # Filtrar y limpiar
    sentences = [s.strip() for s in sentences if s.strip()]
    
    logging.info(f"Número de oraciones detectadas: {len(sentences)}")
    for i, s in enumerate(sentences):
        logging.info(f"Oración {i+1}: {s[:100]}...")
    
    return sentences

def clean_and_trim_sentence(sentence, max_tokens):
    """
    Limpia y optimiza una oración para los subtítulos y prompts.
    Reduce el texto si es muy largo pero mantiene la coherencia.
    """
    # Limpiar espacios extras
    sentence = ' '.join(sentence.split())
    
    # Si la oración es más larga que el máximo de tokens
    if len(sentence.split()) > max_tokens:
        words = sentence.split()
        # Reducir al 70% manteniendo coherencia
        reduced_length = int(len(words) * 0.7)
        final_length = min(reduced_length, max_tokens)
        
        # Intentar cortar en un punto lógico (coma, punto y coma, etc.)
        punctuation_marks = [',', ';', ':', '-']
        best_cut = final_length
        
        for i in range(final_length - 5, final_length + 5):
            if i >= len(words):
                break
            if any(mark in words[i] for mark in punctuation_marks):
                best_cut = i + 1
                break
        
        sentence = ' '.join(words[:best_cut])
        if not any(sentence.endswith(p) for p in '.!?'):
            sentence += '...'
    
    return sentence.strip()

def generate_prompts_from_text(sentences, master_prompt=None):
    """
    Genera prompts optimizados para Stable Diffusion.
    Balancea el prompt maestro con el contenido específico.
    """
    logging.info("Generando prompts...")
    master_prompt = master_prompt or DEFAULT_MASTER_PROMPT
    prompts = []
    
    # Calcular tokens disponibles
    master_tokens = len(master_prompt.split())
    available_tokens = MAX_TOKENS_SD - master_tokens - 2  # Margen de seguridad
    
    for i, sentence in enumerate(sentences):
        try:
            # Optimizar contenido para tokens disponibles
            words = sentence.split()
            content = ' '.join(words[:available_tokens])
            
            # Crear prompt final
            prompt = f"{master_prompt} {content}"
            
            # Verificar longitud total
            if len(prompt.split()) > MAX_TOKENS_SD:
                prompt = ' '.join(prompt.split()[:MAX_TOKENS_SD])
            
            logging.info(f"Prompt [{i+1}]: {prompt}")
            prompts.append(prompt)
            
        except Exception as e:
            logging.error(f"Error procesando prompt {i+1}: {e}")
            # Usar un prompt simplificado en caso de error
            prompts.append(f"{master_prompt} escena {i+1}")
    
    return prompts

def generate_images_with_local_sd(prompts):
    """
    Genera imágenes usando el servidor local de Stable Diffusion.
    Incluye reintentos y manejo de errores mejorado.
    """
    logging.info("\n=== Generación de Imágenes ===")
    logging.info(f"Prompt Negativo Global: {NEGATIVE_PROMPT}")
    
    images = []
    max_retries = 3
    
    for i, prompt in enumerate(prompts):
        retry_count = 0
        while retry_count < max_retries:
            try:
                logging.info(f"\nGenerando imagen {i+1} (intento {retry_count + 1}):")
                logging.info(f"Prompt: {prompt}")
                
                response = requests.post(
                    "http://127.0.0.1:8000/generate",
                    json={
                        "prompt": prompt,
                        "negative_prompt": NEGATIVE_PROMPT,
                        "steps": 30,
                        "width": 1024,
                        "height": 576
                    },
                    timeout=60
                )
                response.raise_for_status()
                
                image_name = generate_random_name()
                image_path = os.path.join(IMAGE_DIR, image_name)
                
                with open(image_path, "wb") as f:
                    f.write(response.content)
                
                images.append(image_path)
                logging.info(f"✓ Imagen guardada: {image_path}")
                
                # Pausa entre generaciones para evitar sobrecarga
                time.sleep(1)
                break  # Salir del bucle si la generación fue exitosa
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                logging.error(f"Error de conexión (intento {retry_count}): {str(e)}")
                if retry_count == max_retries:
                    raise ValueError(f"No se pudo conectar al servidor SD después de {max_retries} intentos")
                time.sleep(2)  # Esperar antes de reintentar
                
            except Exception as e:
                logging.error(f"Error inesperado generando imagen {i+1}: {e}")
                raise
    
    return images

async def generate_voiceover(sentences, voice_id=None, rate="+0%", volume="+0%"):
    """
    Genera voiceovers usando Edge TTS.
    Soporta múltiples voces y ajustes de velocidad/volumen.
    """
    logging.info("Generando audios con Edge TTS...")
    
    # Usar voz por defecto si no se especifica
    voice_id = voice_id or "es-MX-JorgeNeural"
    
    # Validar que la voz existe
    valid_voice = any(voice["id"] == voice_id for voice in VOICES)
    if not valid_voice:
        logging.warning(f"Voz {voice_id} no encontrada, usando voz por defecto")
        voice_id = "es-MX-JorgeNeural"
    
    audio_files = []
    audio_durations = []
    
    for i, sentence in enumerate(sentences):
        try:
            audio_name = generate_random_name("mp3")
            audio_path = os.path.join(AUDIO_DIR, audio_name)
            
            # Configurar el comunicador de TTS
            communicate = Communicate(
                text=sentence,
                voice=voice_id,
                rate=rate,
                volume=volume
            )
            
            # Generar el audio
            await communicate.save(audio_path)
            
            # Obtener duración
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration
            audio_clip.close()
            
            audio_files.append(audio_path)
            audio_durations.append(duration)
            
            logging.info(f"Audio [{i+1}] generado: {audio_path}")
            logging.info(f"Duración: {duration:.2f} segundos")
            
        except Exception as e:
            logging.error(f"Error generando audio {i+1}: {e}")
            raise ValueError(f"Error en la generación de audio: {str(e)}")
    
    return audio_files, audio_durations

def apply_effect(clip, effect_type='zoom_in', duration=None):
    """Aplica efectos visuales al clip de video"""
    duration = duration or clip.duration
    w, h = clip.size

    if effect_type == 'zoom_in':
        return clip.resize(lambda t: 1 + 0.1 * t/duration)
    elif effect_type == 'zoom_out':
        return clip.resize(lambda t: 1.1 - 0.1 * t/duration)
    elif effect_type == 'pan_left_to_right':
        return clip.resize(width=w*1.3).set_position(
            lambda t: ('left', 0) if t==0 else ('right', 0))
    elif effect_type == 'pan_right_to_left':
        return clip.resize(width=w*1.3).set_position(
            lambda t: ('right', 0) if t==0 else ('left', 0))
    elif effect_type == 'ken_burns':
        zoom = clip.resize(lambda t: 1 + 0.2 * t/duration)
        return zoom.set_position(
            lambda t: ('center', 'center' if t==0 else 'top'))
    return clip

def create_video_with_audio(images, sentences, audio_files, audio_durations, output_path):
    logging.info("Creando el video final...")
    video_clips = []
    
    for i, (image_path, audio_path, duration) in enumerate(zip(images, audio_files, audio_durations)):
        try:
            # Preparar el texto del subtítulo
            sentence = sentences[i]
            wrapped_text = "\n".join(wrap(sentence, width=MAX_CHARS_SUBTITLE))
            font_size = 40 if len(sentence) < 100 else 30

            # Crear clip de imagen con efecto
            image_clip = ImageClip(image_path).set_duration(duration)
            effect = random.choice(['zoom_in', 'zoom_out', 'ken_burns'])
            image_clip = apply_effect(image_clip, effect, duration)

            # Crear subtítulos con estilo mejorado
            subtitle = TextClip(
                wrapped_text, 
                fontsize=font_size,
                font='Arial',
                color='white',
                stroke_color='black',
                stroke_width=2,
                bg_color='rgba(0,0,0,0.5)',
                size=(1920, None),
                method='caption'
            )
            subtitle = subtitle.set_duration(duration).set_position(('center', 'bottom'))

            # Crear clip de audio
            audio_clip = AudioFileClip(audio_path).set_duration(duration)
            
            # Combinar todo
            video_clip = CompositeVideoClip([image_clip, subtitle]).set_audio(audio_clip)
            video_clips.append(video_clip)
            
            logging.info(f"Clip [{i+1}] creado con duración: {duration} seg y efecto: {effect}")
        except Exception as e:
            logging.error(f"Error creando clip [{i+1}]: {e}")

    final_video = concatenate_videoclips(video_clips, method="compose")
    final_video.write_videofile(
        output_path,
        fps=FPS,
        codec='libx264',
        audio_codec='aac',
        threads=4,
        preset='medium'
    )
    logging.info(f"Video guardado exitosamente: {output_path}")
    return output_path

async def process_video(
    text, 
    progress_callback=None, 
    effect_type='random', 
    transition='fade',
    master_prompt=None,
    negative_prompt=None,
    voice=None
):
    """
    Procesa el video con callbacks de progreso y opciones personalizables.
    """
    try:
        if progress_callback:
            progress_callback(0, "Iniciando proceso...")

        # Actualizar configuraciones globales si se proporcionan
        global DEFAULT_MASTER_PROMPT, NEGATIVE_PROMPT
        if master_prompt:
            DEFAULT_MASTER_PROMPT = master_prompt
        if negative_prompt:
            NEGATIVE_PROMPT = negative_prompt

        # Limpiar directorios
        clean_output_dirs()
        if progress_callback:
            progress_callback(10, "Directorios preparados")

        # Procesar texto
        sentences = split_text_into_sentences(text)
        sentences = [clean_and_trim_sentence(sentence, MAX_TOKENS_SD) for sentence in sentences]
        if progress_callback:
            progress_callback(20, "Texto procesado")

        # Generar prompts
        prompts = generate_prompts_from_text(sentences)
        if progress_callback:
            progress_callback(30, "Prompts generados")

        # Generar imágenes
        images = generate_images_with_local_sd(prompts)
        if progress_callback:
            progress_callback(60, "Imágenes generadas")

        # Generar audio
        audio_files, audio_durations = await generate_voiceover(sentences, voice or "es-MX-JorgeNeural")
        if progress_callback:
            progress_callback(80, "Audio generado")

        # Crear video final
        final_video_path = create_video_with_audio(
            images, 
            sentences, 
            audio_files, 
            audio_durations, 
            OUTPUT_VIDEO_PATH
        )
        
        if progress_callback:
            progress_callback(100, "Video completado")

        return final_video_path

    except Exception as e:
        logging.error(f"Error en process_video: {e}")
        if progress_callback:
            progress_callback(0, f"Error: {str(e)}")
        raise e

def main(text=None, voice_id="es-MX-JorgeNeural"):
    """
    Función principal que puede ser llamada directamente o desde la interfaz web.
    """
    if text is None:
        text = """La fábula de la luciérnaga y la serpiente..."""  # Texto por defecto
        
    clean_output_dirs()
    sentences = split_text_into_sentences(text)
    sentences = [clean_and_trim_sentence(sentence, MAX_TOKENS_SD) for sentence in sentences]
    prompts = generate_prompts_from_text(sentences)
    images = generate_images_with_local_sd(prompts)
    audio_files, audio_durations = asyncio.run(generate_voiceover(sentences, voice_id))
    final_video_path = create_video_with_audio(images, sentences, audio_files, audio_durations, OUTPUT_VIDEO_PATH)
    logging.info(f"Proceso completado. Video final: {final_video_path}")
    return final_video_path

if __name__ == "__main__":
    main()
