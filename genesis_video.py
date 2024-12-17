import openai
import requests
import os
from gtts import gTTS
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from pydub import AudioSegment

# Configuraciones iniciales
openai.api_key = "***REMOVED***"  # Reemplaza con tu clave de OpenAI

TEXT = """
En el principio creó Dios los cielos y la tierra. Y la tierra estaba desordenada y vacía, 
y las tinieblas estaban sobre la faz del abismo, y el Espíritu de Dios se movía sobre la faz de las aguas.
Y dijo Dios: Sea la luz; y fue la luz. Y vio Dios que la luz era buena; y separó Dios la luz de las tinieblas.
Llamó Dios a la luz Día, y a las tinieblas llamó Noche. Y fue la tarde y la mañana un día.
"""
OUTPUT_DIR = "output"
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "genesis_video.mp4")
TEMP_DIR = "temp_files"

# Crear carpetas temporales y de salida
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def generate_keywords_from_text(text):
    """Genera palabras clave relevantes usando la API de OpenAI ChatCompletion."""
    print("Generando palabras clave con OpenAI...")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Usa 'gpt-4' si tienes acceso, de lo contrario usa 'gpt-3.5-turbo'
        messages=[
            {"role": "system", "content": "Eres un asistente que genera palabras clave visuales relevantes."},
            {"role": "user", "content": f"Extrae palabras clave visuales descriptivas del siguiente texto:\n{text}"}
        ],
        max_tokens=50,
        temperature=0.5
    )
    # Acceder a la respuesta generada
    content = response.choices[0].message.content
    keywords = content.strip().split("\n")
    return [kw.strip() for kw in keywords if kw.strip()]



def generate_images_with_openai(keywords):
    """Genera imágenes usando DALL·E a partir de palabras clave."""
    print("Generando imágenes con OpenAI DALL·E...")
    image_paths = []
    for idx, keyword in enumerate(keywords):
        response = openai.Image.create(
            prompt=keyword,
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']
        image_path = os.path.join(TEMP_DIR, f"image_{idx}.jpg")
        with open(image_path, "wb") as img:
            img.write(requests.get(image_url).content)
        image_paths.append(image_path)
    return image_paths

def text_to_speech(text, output_path):
    """Convierte el texto en voz usando gTTS."""
    print("Generando voz...")
    tts = gTTS(text, lang="es")
    tts.save(output_path)

def create_video_with_audio(images, audio_path, output_path):
    """Crea un video combinando imágenes y audio."""
    print("Creando video...")
    audio = AudioFileClip(audio_path)
    duration_per_image = audio.duration / len(images)

    clips = []
    for image in images:
        clip = ImageClip(image).set_duration(duration_per_image)
        clips.append(clip)
    video = concatenate_videoclips(clips)
    video = video.set_audio(audio)
    video.write_videofile(output_path, fps=24)

def main():
    # Generar palabras clave desde el texto
    keywords = generate_keywords_from_text(TEXT)
    print(f"Palabras clave generadas: {keywords}")

    # Generar imágenes con OpenAI DALL·E
    images = generate_images_with_openai(keywords)

    # Generar voz desde el texto
    audio_path = os.path.join(TEMP_DIR, "voice.mp3")
    text_to_speech(TEXT, audio_path)

    # Crear video
    create_video_with_audio(images, audio_path, OUTPUT_VIDEO)
    print(f"Video creado exitosamente: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
