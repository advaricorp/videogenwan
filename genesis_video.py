import os
import re
import asyncio
import logging
import requests
from textwrap import wrap
from uuid import uuid4
from edge_tts import Communicate
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    TextClip, ImageClip, AudioFileClip, CompositeVideoClip, concatenate_videoclips
)

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuraciones globales
OUTPUT_DIR = "output"
TEMP_DIR = "temp_files"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, "final_video.mp4")

MAX_TOKENS_SD = 75
MAX_CHARS_PROMPT = 200
MAX_CHARS_SUBTITLE = 50
NEGATIVE_PROMPT = "realistic human, cartoon, anime, low quality, blurry, text, watermark"
FPS = 24

# Crear carpetas necesarias
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

def generate_random_name(extension="jpg"):
    return f"{uuid4().hex}.{extension}"

def split_text_into_sentences(text):
    # Dividir por puntos seguidos de espacio y mayúscula
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    # Limpiar espacios y filtrar líneas vacías
    sentences = [s.strip() for s in sentences if s.strip()]
    logging.info(f"Número de oraciones detectadas: {len(sentences)}")
    for i, s in enumerate(sentences):
        logging.info(f"Oración {i+1}: {s[:100]}...")  # Mostrar primeros 100 caracteres
    return sentences

def clean_and_trim_sentence(sentence, max_tokens):
    if len(sentence.split()) > max_tokens:
        sentence = " ".join(sentence.split()[:max_tokens])
    return sentence.strip()

def generate_prompts_from_text(sentences):
    logging.info("Generando prompts...")
    prompts = []
    for i, sentence in enumerate(sentences):
        prompt = f"Dibuja una impresionante y epica en base a : {sentence}. {NEGATIVE_PROMPT}"
        if len(prompt) > MAX_CHARS_PROMPT:
            prompt = prompt[:MAX_CHARS_PROMPT].rsplit(" ", 1)[0]
        logging.info(f"Prompt [{i+1}]: {prompt}")
        prompts.append(prompt)
    return prompts

def generate_images_with_local_sd(prompts):
    logging.info("Generando imágenes con Stable Diffusion...")
    images = []
    for i, prompt in enumerate(prompts):
        try:
            logging.info(f"Enviando prompt a Stable Diffusion [{i+1}]: {prompt}")
            response = requests.post("http://127.0.0.1:8000/generate", json={"prompt": prompt})
            response.raise_for_status()
            image_name = generate_random_name()
            image_path = os.path.join(IMAGE_DIR, image_name)
            with open(image_path, "wb") as f:
                f.write(response.content)
            images.append(image_path)
            logging.info(f"Imagen [{i+1}] guardada: {image_path}")
        except Exception as e:
            logging.error(f"Error al generar la imagen [{i+1}]: {e}")
            raise ValueError("Error en la generación de imágenes.")
    return images

async def generate_voiceover(sentences):
    logging.info("Generando audios con Edge TTS...")
    audio_files = []
    audio_durations = []
    for i, sentence in enumerate(sentences):
        try:
            audio_name = generate_random_name("mp3")
            audio_path = os.path.join(AUDIO_DIR, audio_name)
            communicate = Communicate(text=sentence, voice="es-MX-JorgeNeural")
            await communicate.save(audio_path)

            # Obtener duración del audio
            audio_clip = AudioFileClip(audio_path)
            audio_durations.append(audio_clip.duration)
            audio_clip.close()

            audio_files.append(audio_path)
            logging.info(f"Audio [{i+1}] guardado: {audio_path}, Duración: {audio_durations[-1]} seg")
        except Exception as e:
            logging.error(f"Error al generar el audio [{i+1}]: {e}")
            raise ValueError("Error en la generación de audios.")
    return audio_files, audio_durations

def create_video_with_audio(images, sentences, audio_files, audio_durations, output_path):
    logging.info("Creando el video final...")
    video_clips = []
    
    for i, (image_path, audio_path, duration) in enumerate(zip(images, audio_files, audio_durations)):
        try:
            sentence = sentences[i]
            wrapped_text = "\n".join(wrap(sentence, width=MAX_CHARS_SUBTITLE))
            font_size = 40 if len(sentence) < 100 else 30

            # Crear subtítulos
            subtitle = TextClip(
                wrapped_text, 
                fontsize=font_size,
                color='white',
                bg_color='black',
                size=(1920, None),
                method='caption'
            )
            subtitle = subtitle.set_duration(duration).set_position(('center', 'bottom'))

            # Crear clip de imagen
            image_clip = ImageClip(image_path).set_duration(duration)
            audio_clip = AudioFileClip(audio_path).set_duration(duration)
            
            # Combinar imagen, subtítulo y audio
            video_clip = CompositeVideoClip([image_clip, subtitle]).set_audio(audio_clip)
            video_clips.append(video_clip)
            
            logging.info(f"Clip [{i+1}] creado con duración: {duration} seg")
        except Exception as e:
            logging.error(f"Error creando clip [{i+1}]: {e}")

    final_video = concatenate_videoclips(video_clips, method="compose")
    final_video.write_videofile(output_path, fps=FPS, threads=4)
    logging.info(f"Video guardado exitosamente: {output_path}")
    return output_path

def main():
    TEXT = """ La fábula de la luciérnaga y la serpiente narra cómo una serpiente perseguía incansablemente a una luciérnaga. Tras varios días de huir, la luciérnaga, agotada, se detuvo y le dijo a la serpiente: “¿Puedo hacerte tres preguntas antes de que me devores?”. La serpiente aceptó. La luciérnaga preguntó: “¿Pertenezco a tu cadena alimenticia?”. La serpiente respondió: “No”. La luciérnaga continuó: “¿Te hice algún daño?”. La serpiente admitió: “No”. Finalmente, la luciérnaga preguntó: “Entonces, ¿por qué quieres acabar conmigo?”. La serpiente contestó: “Porque no soporto verte brillar”.

    La luciérnaga, con valentía y determinación, respondió: “Pues si esa es la razón, voy a brillar más fuerte y volar más alto”. La moraleja de la fábula es clara: no permitas que la envidia apague tu luz; al contrario, enfréntala brillando con más intensidad y alcanzando nuevas alturas."""
    sentences = split_text_into_sentences(TEXT)
    sentences = [clean_and_trim_sentence(sentence, MAX_TOKENS_SD) for sentence in sentences]
    prompts = generate_prompts_from_text(sentences)
    images = generate_images_with_local_sd(prompts)
    audio_files, audio_durations = asyncio.run(generate_voiceover(sentences))
    final_video_path = create_video_with_audio(images, sentences, audio_files, audio_durations, OUTPUT_VIDEO_PATH)
    logging.info(f"Proceso completado. Video final: {final_video_path}")

if __name__ == "__main__":
    main()
