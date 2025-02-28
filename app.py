from flask import Flask, request, jsonify, Response, send_file, render_template
import json
import time
import logging
import os
import re
import shutil
import subprocess
import uuid
import requests
import sys
from pathlib import Path
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip, TextClip, ColorClip, CompositeVideoClip

app = Flask(__name__)

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Configuraciones por defecto
DEFAULT_VOICE = "es-MX-DaliaNeural"
DEFAULT_MASTER_PROMPT = "Realistic"
DEFAULT_EFFECT = "random"
DEFAULT_TRANSITION = "fade"
DEFAULT_BACKGROUND_TYPE = "video"
DEFAULT_MODEL_SIZE = "1.3B"
DEFAULT_RESOLUTION = "832*480"

# Asegurarse de que los directorios necesarios existan
os.makedirs('output/audio', exist_ok=True)
os.makedirs('output/video', exist_ok=True)
os.makedirs('static/videos', exist_ok=True)

def clean_directories():
    """Limpia los directorios de salida."""
    for directory in ['output/audio', 'output/video', 'static/videos']:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            continue
            
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logging.error(f"Error al eliminar {file_path}: {e}")

def generate_placeholder_video(output_path, message="Video no disponible"):
    """Genera un video placeholder con un mensaje."""
    try:
        # Crear un clip de fondo negro
        bg_clip = ColorClip(size=(832, 480), color=(0, 0, 0), duration=5)
        
        # Crear el texto con el mensaje
        text_clip = TextClip(message, fontsize=30, color='white', bg_color='transparent', 
                          size=(700, 400), method='caption').set_position('center').set_duration(5)
        
        # Combinar los clips
        final_clip = CompositeVideoClip([bg_clip, text_clip])
        
        # Guardar el video
        final_clip.write_videofile(output_path, fps=24, codec='libx264', audio=False)
        
        return output_path
    except Exception as e:
        logging.exception(f"Error al crear video placeholder: {e}")
        return None

def generate_audio(text, voice="es-ES-AlvaroNeural"):
    """Genera un archivo de audio con Edge TTS."""
    try:
        output_file = f"output/audio/{uuid.uuid4().hex}.mp3"
        
        command = [
            sys.executable, "-m", "edge_tts",
            "--voice", voice,
            "--text", text,
            "--write-media", output_file
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            return output_file
        else:
            logging.error(f"Error al generar audio: {result.stderr}")
            return None
    except Exception as e:
        logging.exception(f"Excepción al generar audio: {str(e)}")
        return None

def generate_simple_video(prompt, model_size="1.3B", resolution="832*480"):
    """
    Versión simplificada de generación de video que solo usa Wan2.1.
    
    Retorna:
        str: Ruta del video generado o None si hay error.
    """
    from wan_t2v import generate_video_with_wan, ensure_wan_repo_exists
    
    try:
        # Asegurar que existe el repositorio de Wan2.1
        ensure_wan_repo_exists()
        
        # Generar un nombre de archivo único para el video
        output_file = os.path.join("static", "videos", f"output_{int(time.time())}.mp4")
        
        # Generar video
        return generate_video_with_wan(
            prompt=prompt,
            model_size=model_size,
            resolution=resolution,
            output_path=output_file
        )
    except Exception as e:
        logging.exception(f"Error al generar video simple: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Obtener parámetros del formulario
        prompt = request.form.get('prompt', '').strip()
        voice = request.form.get('voice', DEFAULT_VOICE)
        master_prompt = request.form.get('master_prompt', DEFAULT_MASTER_PROMPT)
        model_size = request.form.get('model_size', DEFAULT_MODEL_SIZE)
        resolution = request.form.get('resolution', DEFAULT_RESOLUTION)
        
        logging.info(f"Recibido prompt: '{prompt}'")
        
        # Verificar si hay un prompt
        if not prompt:
            logging.warning("Prompt vacío. Usando texto por defecto.")
            prompt = "This is a test video. The prompt was empty, so we're generating this placeholder content."
        
        # Limpiar directorios de salida
        clean_directories()
        
        # Construir prompt completo con master prompt
        full_prompt = f"{master_prompt} {prompt}"
        logging.info(f"Prompt completo: '{full_prompt}'")
        
        # Crear el directorio de salida de videos final
        os.makedirs("output", exist_ok=True)
        final_video_path = os.path.join("output", "final_video.mp4")
        
        # Generar video placeholder primero (para asegurar que siempre tenemos algo)
        default_placeholder = os.path.join("static", "videos", f"default_{int(time.time())}.mp4")
        generate_placeholder_video(default_placeholder, "Generando video, por favor espere...")
        
        # Generar video con Wan2.1
        from wan_t2v import generate_video_with_wan
        video_path = generate_video_with_wan(full_prompt, model_size, resolution)
        
        # Si no se generó el video, usar el placeholder
        if not video_path or not os.path.exists(video_path):
            logging.warning("No se pudo generar el video. Usando placeholder.")
            video_path = default_placeholder
            if not os.path.exists(video_path):
                video_path = generate_placeholder_video(
                    os.path.join("static", "videos", f"placeholder_{int(time.time())}.mp4"),
                    f"No se pudo generar el video para:\n{prompt}"
                )
        
        # Generar audio para el prompt
        audio_path = generate_audio(prompt, voice)
        
        # Crear video final con audio (si está disponible)
        try:
            if os.path.exists(video_path):
                video_clip = VideoFileClip(video_path)
                
                # Si hay audio, añadirlo al video
                if audio_path and os.path.exists(audio_path):
                    audio_clip = AudioFileClip(audio_path)
                    # Ajustar duración del video al audio
                    video_duration = video_clip.duration
                    audio_duration = audio_clip.duration
                    
                    # Si el audio es más largo que el video, repetir el video
                    if audio_duration > video_duration:
                        repeats = int(audio_duration / video_duration) + 1
                        video_clips = [video_clip] * repeats
                        extended_clip = concatenate_videoclips(video_clips)
                        extended_clip = extended_clip.subclip(0, audio_duration)
                        extended_clip = extended_clip.set_audio(audio_clip)
                        final_clip = extended_clip
                    else:
                        # Si el video es más largo, cortar el video a la duración del audio
                        video_clip = video_clip.subclip(0, audio_duration)
                        video_clip = video_clip.set_audio(audio_clip)
                        final_clip = video_clip
                else:
                    final_clip = video_clip
                
                # Guardar video final
                final_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac" if audio_path else None)
                logging.info(f"Video final guardado en {final_video_path}")
            else:
                # Si no hay video, crear uno de placeholder
                generate_placeholder_video(
                    final_video_path, 
                    f"No se pudo generar el video para:\n{prompt}"
                )
                
        except Exception as e:
            logging.exception(f"Error al combinar video y audio: {str(e)}")
            # Si falla, crear un video de placeholder
            generate_placeholder_video(
                final_video_path, 
                f"Error al procesar el video:\n{str(e)}"
            )
        
        return jsonify({
            'success': True,
            'message': 'Video generado exitosamente',
            'video_url': '/video'
        })
        
    except Exception as e:
        logging.exception(f"Error en generación: {str(e)}")
        
        # Generar un video de error para mostrar algo
        try:
            error_path = os.path.join("output", "final_video.mp4")
            os.makedirs(os.path.dirname(error_path), exist_ok=True)
            generate_placeholder_video(
                error_path, 
                f"Error al generar el video:\n{str(e)}\nPor favor, inténtalo de nuevo."
            )
        except Exception as err:
            logging.exception(f"Error al crear video de error: {str(err)}")
            
        return jsonify({"status": "error", "message": str(e)})

@app.route('/video')
def video():
    try:
        return send_file('output/final_video.mp4', mimetype='video/mp4')
    except Exception as e:
        logging.error(f"Error enviando video: {e}")
        return str(e), 404

@app.route('/download')
def download():
    try:
        return send_file(
            'output/final_video.mp4',
            as_attachment=True,
            download_name='video_generado.mp4'
        )
    except Exception as e:
        logging.error(f"Error descargando video: {e}")
        return str(e), 404

@app.route('/progress')
def progress():
    """Endpoint para mostrar el progreso de la generación."""
    try:
        # Verificar si hay un archivo de video
        video_path = os.path.join("output", "final_video.mp4")
        if os.path.exists(video_path):
            status = 100  # Completo
            message = "Video generado exitosamente"
        else:
            status = 50   # En progreso
            message = "Generando video..."
            
        return jsonify({
            "progress": status,
            "message": message
        })
    except Exception as e:
        logging.exception(f"Error en endpoint progress: {str(e)}")
        return jsonify({
            "progress": 0,
            "message": f"Error: {str(e)}"
        })

if __name__ == '__main__':
    # Importar y aplicar el parche de dashscope una vez al iniciar
    try:
        from dashscope_patch import apply_dashscope_patch
        apply_dashscope_patch()
    except Exception as e:
        logging.error(f"Error al aplicar el parche de dashscope: {str(e)}")
    
    app.run(debug=False)