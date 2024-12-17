import os
import requests
from openai import OpenAI
from gtts import gTTS
from moviepy.editor import ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OUTPUT_DIR = "output"
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "genesis_video.mp4")
TEMP_DIR = "temp_files"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_text_into_sentences(text):
    """Divide el texto en oraciones basadas en saltos de línea o puntos."""
    sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]
    return sentences

def clean_sentence(sentence):
    """Filtra palabras sensibles que podrían activar el filtro de contenido."""
    forbidden_words = ["Dios", "abismo", "religiosas", "espíritu"]
    for word in forbidden_words:
        sentence = sentence.replace(word, "paisaje")
    return sentence

def generate_prompts_from_text(sentences):
    """Genera prompts usando OpenAI GPT-3.5 a partir de oraciones."""
    print("Generando prompts con OpenAI...")
    prompts = []
    for sentence in sentences:
        # Clean the sentence to remove forbidden words
        cleaned_sentence = clean_sentence(sentence)
        
        # Generate the prompt
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Genera descripciones visuales detalladas de escenas basadas en texto biblico. Las imágenes deben representar solo paisajes o elementos de la naturaleza, como cielos, montañas, ríos o árboles. Evita cualquier mención de personas, figuras religiosas, animales o elementos controvertidos."},
                {"role": "user", "content": f"Describe un paisaje visual basado en la siguiente oración:\n{cleaned_sentence}"}
            ],
            max_tokens=100,
            temperature=0.5
        )
        prompt = response.choices[0].message.content.strip()
        final_prompt = f"{prompt}. La imagen debe representar únicamente un paisaje natural con elementos como montañas, cielos o ríos. No debe incluir personas, animales ni figuras religiosas."
        print(f"Prompt generado: {final_prompt}")
        prompts.append(final_prompt)
    return prompts

def generate_images_with_openai(prompts):
    """Genera imágenes usando DALL·E a partir de los prompts con caché."""
    print("Generando imágenes con OpenAI DALL·E (usando caché si existe)...")
    image_paths = []
    for idx, prompt in enumerate(prompts):
        image_path = os.path.join(TEMP_DIR, f"image_{idx}.jpg")

        # Check if the image already exists in the cache
        if os.path.exists(image_path):
            print(f"[CACHÉ] Usando imagen existente: {image_path}")
            image_paths.append(image_path)
            continue
        
        # Generate a new image if not cached
        print(f"[API] Generando imagen {idx+1}/{len(prompts)}: {prompt}")
        try:
            response = client.images.generate(
                prompt=prompt,
                n=1,
                size="512x512"
            )
            image_url = response.data[0].url
            with open(image_path, "wb") as img:
                img.write(requests.get(image_url).content)
            print(f"[API] Imagen guardada: {image_path}")
            image_paths.append(image_path)
        except Exception as e:
            print(f"[ERROR] No se pudo generar la imagen {idx+1}: {e}")
            raise e
    return image_paths

def text_to_speech(text, output_path):
    """Convierte el texto en voz usando gTTS."""
    print("Generando voz...")
    tts = gTTS(text, lang="es")
    tts.save(output_path)

def split_sentence_into_chunks(sentence, chunk_size=4):
    """Divide una oración en trozos de ~4 palabras cada uno."""
    words = sentence.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def create_video_with_audio(images, sentences, audio_path, output_path):
    """
    Crea un video combinando imágenes y audio con subtítulos, transiciones y zoom.
    Cada oración se divide en trozos más pequeños de ~4 palabras para subtítulos más legibles.
    """
    print("Creando video con subtítulos, transiciones y zoom...")
    audio = AudioFileClip(audio_path)
    total_duration = audio.duration

    # Distribute time proportionally according to the number of sentences
    duration_per_sentence = total_duration / len(sentences)

    clips = []
    for idx, (image, sentence) in enumerate(zip(images, sentences)):
        # Split sentence into chunks of 4 words
        chunks = split_sentence_into_chunks(sentence, chunk_size=4)
        chunk_count = len(chunks)
        
        # Each chunk gets an equal slice of the sentence duration
        chunk_duration = duration_per_sentence / chunk_count

        # Create base image clip for the sentence duration
        base_clip = ImageClip(image, duration=duration_per_sentence)
        base_clip = base_clip.resize(height=720).crop(x1=10, y1=10, x2=base_clip.w-10, y2=base_clip.h-10).resize(height=720)

        # Create dynamically-timed subtitles over the base clip
        subtitles = []
        current_start = 0
        for chunk in chunks:
            subtitle = TextClip(chunk, fontsize=40, color='white', bg_color='black', 
                                size=(base_clip.w, 100), method='caption')
            subtitle = subtitle.set_duration(chunk_duration).set_position(('center', 'bottom')).set_start(current_start)
            subtitles.append(subtitle)
            current_start += chunk_duration

        # Combine base image with all subtitle chunks
        video_with_subtitles = CompositeVideoClip([base_clip] + subtitles)
        clips.append(video_with_subtitles)

    # Concatenate all sentence clips with a slight crossfade
    video = concatenate_videoclips(clips, method="compose", padding=-1, bg_color=(0, 0, 0))
    video = video.set_audio(audio)
    video.write_videofile(output_path, fps=24, threads=4)


def main():
    TEXT = """
    En el principio creó Dios los cielos y la tierra. Y la tierra estaba desordenada y vacía,
    y las tinieblas estaban sobre la faz del abismo, y el Espíritu de Dios se movía sobre la faz de las aguas.
    Y dijo Dios: Sea la luz; y fue la luz. Y vio Dios que la luz era buena; y separó Dios la luz de las tinieblas.
    Llamó Dios a la luz Día, y a las tinieblas llamó Noche. Y fue la tarde y la mañana un día.
    """

    # Dividir texto en oraciones
    sentences = split_text_into_sentences(TEXT)
    print(f"Oraciones extraídas: {sentences}")

    # Verificar si las imágenes ya existen en caché
    cached_images = [os.path.join(TEMP_DIR, f"image_{i}.jpg") for i in range(len(sentences))]
    all_images_cached = all(os.path.exists(image) for image in cached_images)

    if all_images_cached:
        print("[CACHÉ] Todas las imágenes ya existen, omitiendo generación de prompts e imágenes.")
        images = cached_images
    else:
        print("Generando prompts e imágenes...")
        prompts = generate_prompts_from_text(sentences)
        print(f"Prompts generados: {prompts}")
        images = generate_images_with_openai(prompts)

    # Generar voz desde el texto
    audio_path = os.path.join(TEMP_DIR, "voice.mp3")
    text_to_speech(TEXT, audio_path)

    # Crear video
    create_video_with_audio(images, sentences, audio_path, OUTPUT_VIDEO)
    print(f"Video creado exitosamente: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
