from flask import Flask, request, jsonify, Response, send_file, render_template
import json
import time
import asyncio
import logging
from genesis_video import process_video

app = Flask(__name__)

# Configuraciones por defecto
DEFAULT_VOICE = "es-MX-DaliaNeural"
DEFAULT_MASTER_PROMPT = "Genera una imagen fotorrealista, cinematográfica y de alta calidad que represente:"
DEFAULT_NEGATIVE_PROMPT = "realistic human, cartoon, anime, low quality, blurry, text, watermark, bad body morphology"
DEFAULT_EFFECT = "random"
DEFAULT_TRANSITION = "fade"

# Variables globales para el progreso
current_progress = {"progress": 0, "message": ""}

def update_progress(progress, message):
    global current_progress
    current_progress = {"progress": progress, "message": message}

def get_current_progress():
    return current_progress

@app.route('/')
def index():
    return render_template('index.html', 
        voice=DEFAULT_VOICE,
        master_prompt=DEFAULT_MASTER_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        effect=DEFAULT_EFFECT,
        transition=DEFAULT_TRANSITION
    )

@app.route('/generate', methods=['POST'])
def generate():
    try:
        text = request.form.get('text', '')
        if not text:
            return jsonify({'success': False, 'error': 'No se proporcionó texto'})

        # Obtener parámetros del formulario
        effect_type = request.form.get('effect_type', DEFAULT_EFFECT)
        transition = request.form.get('transition', DEFAULT_TRANSITION)
        master_prompt = request.form.get('master_prompt', DEFAULT_MASTER_PROMPT)
        negative_prompt = request.form.get('negative_prompt', DEFAULT_NEGATIVE_PROMPT)
        voice = request.form.get('voice', DEFAULT_VOICE)
        
        logging.info(f"Iniciando generación con parámetros:")
        logging.info(f"- Efecto: {effect_type}")
        logging.info(f"- Transición: {transition}")
        logging.info(f"- Voz: {voice}")
        logging.info(f"- Master Prompt: {master_prompt}")

        # Procesar video con los parámetros especificados
        asyncio.run(process_video(
            text=text,
            progress_callback=update_progress,
            effect_type=effect_type,
            transition=transition,
            master_prompt=master_prompt,
            negative_prompt=negative_prompt,
            voice=voice
        ))

        return jsonify({
            'success': True,
            'message': 'Video generado exitosamente'
        })

    except Exception as e:
        logging.error(f"Error en generación: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/progress')
def progress():
    def generate():
        while True:
            progress = get_current_progress()
            yield f"data: {json.dumps(progress)}\n\n"
            if progress['progress'] >= 100:
                break
            time.sleep(0.5)
    
    return Response(generate(), mimetype='text/event-stream')

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

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    app.run(debug=True)