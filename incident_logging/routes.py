# routes.py
from flask import Blueprint, render_template, request, jsonify, Response # Add Response
from services.transcriber import transcribe_audio
from services.parser import parse_incident_text
from services.tts import text_to_speech_elevenlabs # Import the new service
import json

bp = Blueprint('main', __name__)

OPERATOR_NAME = "Sharun"
MACHINE_ID = "CAT-797F-451"

@bp.route('/')
def index():
    return render_template('index.html', operator_name=OPERATOR_NAME, machine_id=MACHINE_ID)

# Add this new route for Text-to-Speech
@bp.route('/synthesize_speech', methods=['POST'])
def synthesize_speech_route():
    data = request.get_json()
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    audio_content = text_to_speech_elevenlabs(text)
    
    if audio_content:
        return Response(audio_content, mimetype="audio/mpeg")
    else:
        return jsonify({"error": "Failed to generate speech"}), 500


@bp.route('/process_incident', methods=['POST'])
def process_incident_route():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file found"}), 400

    audio_file = request.files['audio_data']
    context_data_json = request.form.get('context')
    context_data = json.loads(context_data_json) if context_data_json else None
    
    transcript = transcribe_audio(audio_file)
    if transcript.startswith("Error:"):
        return jsonify({"error": transcript}), 500

    parsed_data = parse_incident_text(transcript, OPERATOR_NAME, MACHINE_ID, context=context_data)
    
    response_data = {
        "transcript": transcript,
        "parsed_incident": parsed_data
    }
    
    return jsonify(response_data)