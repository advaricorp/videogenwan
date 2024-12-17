import os
import requests
from openai import OpenAI
from gtts import gTTS
from moviepy.editor import ImageClip, TextClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
from dotenv import load_dotenv
from textwrap import wrap


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
                {"role": "system", "content": (
                    "Eres un generador de descripciones visuales detalladas para paisajes inspirados en textos bíblicos. "
                    "Las descripciones deben enfocarse exclusivamente en paisajes naturales majestuosos y serenos. "
                    "Incorpora elementos como: montañas, cielos, ríos, árboles, bosques, barrancos, praderas y cuerpos de agua. "
                    "Describe detalles como colores específicos (dorado, verde, azul claro), texturas (rocosa, suave, reflejante), "
                    "juegos de luces (amanecer, atardecer, luz de luna) y atmósferas (calma, majestuosidad, misterio). "
                    "Evita completamente menciones de personas, animales, figuras religiosas o construcciones humanas. "
                    "Crea descripciones vivas y visuales que sirvan como referencia para ilustraciones."
                )},
                {"role": "user", "content": f"Describe un paisaje visual basado en la siguiente oración:\n{cleaned_sentence}"}
            ],
            max_tokens=150,
            temperature=0.5
        )
        if response.choices and response.choices[0].message:
            prompt = response.choices[0].message.content.strip()
        else:
            raise ValueError("La respuesta de OpenAI no contiene un mensaje válido.")
        final_prompt = (
            f"{prompt}. "
            "Asegúrate de que la imagen represente únicamente un paisaje natural con detalles visuales nítidos, "
            "como cielos despejados, montañas con sombras suaves, cuerpos de agua reflejantes o praderas verdes. "
            "No debe incluir personas, animales ni construcciones humanas."
        )
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

def create_video_with_audio(images, sentences, output_path):
    """
    Genera un video combinando imágenes y audios individuales con subtítulos sincronizados.
    Ajusta los subtítulos dividiéndolos en múltiples líneas si son demasiado largos.
    """
    print("Creando audios individuales y video...")
    audio_clips = []
    video_clips = []

    # Generar un audio por cada oración
    for idx, (image, sentence) in enumerate(zip(images, sentences)):
        audio_path = os.path.join(TEMP_DIR, f"audio_{idx}.mp3")
        
        # Crear el audio con gTTS
        tts = gTTS(sentence, lang="es")
        tts.save(audio_path)
        print(f"Audio generado: {audio_path}")

        # Cargar el audio generado
        audio_clip = AudioFileClip(audio_path)
        audio_clips.append(audio_clip)

        # Crear imagen clip con duración igual a la del audio
        image_clip = ImageClip(image, duration=audio_clip.duration)
        image_clip = image_clip.resize(height=720).crop(x1=10, y1=10, x2=image_clip.w-10, y2=image_clip.h-10)

        # Dividir el subtítulo en líneas de ~50 caracteres (ajustable)
        max_chars_per_line = 50
        wrapped_text = "\n".join(wrap(sentence, width=max_chars_per_line))

        # Ajustar el tamaño de la fuente dinámicamente si el texto es muy largo
        font_size = 40 if len(sentence) < 100 else 30

        # Agregar subtítulos sincronizados
        subtitle = TextClip(wrapped_text, fontsize=font_size, color='white', bg_color='black',
                            size=(image_clip.w, None), method='caption')
        subtitle = subtitle.set_duration(audio_clip.duration).set_position(('center', 'bottom'))

        # Combinar imagen y subtítulos
        video_with_subtitles = CompositeVideoClip([image_clip, subtitle])
        video_with_subtitles = video_with_subtitles.set_audio(audio_clip)

        video_clips.append(video_with_subtitles)

    # Unir todos los clips en un solo video
    final_video = concatenate_videoclips(video_clips, method="compose")
    final_video.write_videofile(output_path, fps=24, threads=4)


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

    # Crear video
    create_video_with_audio(images, sentences, OUTPUT_VIDEO)
    print(f"Video creado exitosamente: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
