from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from openai import OpenAI
import base64
import os
import requests
import json
import time
import tempfile
import io
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size for audio

# Enable CORS for Sahayak frontend integration
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"])

# Set up OpenRouter client (using OpenAI library but pointing to OpenRouter)
# Note: API key should be set via environment variable OPENROUTER_API_KEY
openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
if not openrouter_key:
    print("⚠️ WARNING: OPENROUTER_API_KEY not found in environment variables!")
else:
    print(f"✅ OpenRouter API key loaded: {openrouter_key[:10]}...{openrouter_key[-10:]}")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key
)

# Set up A4F client for backup AI models
# Note: API key should be set via environment variable A4F_API_KEY
a4f_api_key = os.environ.get("A4F_API_KEY", "")
a4f_base_url = "https://api.a4f.co/v1"

a4f_client = OpenAI(
    api_key=a4f_api_key,
    base_url=a4f_base_url,
)

# A4F Backup Models Configuration
A4F_BACKUP_MODELS = [
    "provider-5/gemini-2.5-flash-preview-04-17",
    "provider-5/gemini-2.0-flash-exp",
    "provider-2/gpt-3.5-turbo"
]

def get_ai_response_with_fallback(messages, task_type="general", user_id=None, session_id=None):
    """
    Get AI response with comprehensive fallback system and MCP context integration
    First tries OpenRouter, then falls back to A4F models
    """

    # Enhanced messages with MCP context
    enhanced_messages = messages.copy()

    # Fetch existing context if user_id and session_id provided
    if user_id and session_id:
        try:
            print(f"🧠 Fetching MCP context for user {user_id}, session {session_id}")

            # Fetch context from MCP server
            mcp_response = requests.get(
                'http://127.0.0.1:5001/context/fetch',
                params={'user_id': user_id, 'session_id': session_id},
                headers={'X-API-Key': 'mcp-api-key-12345'},
                timeout=5
            )

            print(f"🔍 MCP fetch response status: {mcp_response.status_code}")

            if mcp_response.status_code == 200:
                mcp_data = mcp_response.json()
                if mcp_data.get('success') and mcp_data.get('context'):
                    stored_context = mcp_data['context']
                    print(f"Retrieved {len(stored_context)} messages from MCP context")

                    # Combine stored context with new messages
                    # Keep only the last 8 messages from context to prevent token overflow
                    if len(stored_context) > 8:
                        stored_context = stored_context[-8:]

                    # Add new message to context
                    enhanced_messages = stored_context + messages
                    print(f"Enhanced messages with context: {len(enhanced_messages)} total messages")

        except Exception as e:
            print(f"Failed to fetch MCP context: {e}")

    # Primary models for different tasks
    primary_models = {
        "text": "deepseek/deepseek-r1-0528-qwen3-8b:free",
        "vision": "google/gemma-3n-e4b-it:free",
        "general": "deepseek/deepseek-r1-0528-qwen3-8b:free"
    }

    primary_model = primary_models.get(task_type, "deepseek/deepseek-r1-0528-qwen3-8b:free")

    # Try OpenRouter first
    try:
        print(f"Trying OpenRouter with {primary_model}...")
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "AI Chat Integration App",
            },
            model=primary_model,
            messages=enhanced_messages,
            timeout=120  # Add 2-minute timeout
        )

        ai_response = response.choices[0].message.content
        model_used = f"OpenRouter ({primary_model})"

        # Store updated context in MCP if user_id and session_id provided
        if user_id and session_id and ai_response:
            try:
                # Create updated context with the new interaction
                new_context = enhanced_messages + [{"role": "assistant", "content": ai_response}]

                # Keep only last 10 messages to prevent context from growing too large
                if len(new_context) > 10:
                    new_context = new_context[-10:]

                # Store context in MCP server
                update_response = requests.post(
                    'http://127.0.0.1:5001/context/update',
                    json={
                        'user_id': user_id,
                        'session_id': session_id,
                        'context': new_context
                    },
                    headers={'X-API-Key': 'mcp-api-key-12345'},
                    timeout=5
                )

                print(f"💾 Updated MCP context with new interaction (status: {update_response.status_code})")

            except Exception as e:
                print(f"Failed to update MCP context: {e}")

        return ai_response, model_used

    except Exception as e:
        print(f"OpenRouter failed: {e}")

        # Try A4F backup models
        for i, backup_model in enumerate(A4F_BACKUP_MODELS, 1):
            try:
                print(f"Trying A4F backup model {i}: {backup_model}...")
                response = a4f_client.chat.completions.create(
                    model=backup_model,
                    messages=enhanced_messages
                )

                ai_response = response.choices[0].message.content
                model_name = backup_model.split('/')[-1] if '/' in backup_model else backup_model
                model_used = f"A4F ({model_name})"

                # Store updated context in MCP if successful
                if user_id and session_id and ai_response:
                    try:
                        new_context = enhanced_messages + [{"role": "assistant", "content": ai_response}]
                        if len(new_context) > 10:
                            new_context = new_context[-10:]

                        requests.post(
                            'http://127.0.0.1:5001/context/update',
                            json={
                                'user_id': user_id,
                                'session_id': session_id,
                                'context': new_context
                            },
                            headers={'X-API-Key': 'mcp-api-key-12345'},
                            timeout=5
                        )
                        print(f"Updated MCP context with A4F response")
                    except Exception as e:
                        print(f"Failed to update MCP context: {e}")

                return ai_response, model_used

            except Exception as backup_error:
                print(f"A4F model {backup_model} failed: {backup_error}")
                continue

        # If all models fail
        return None, f"All AI models failed. OpenRouter: {str(e)}"


# -------------------------
# Health & utility endpoints
# -------------------------
@app.route('/health-check', methods=['GET'])
def health_check():
    """Simple health endpoint for frontend connectivity checks."""
    return jsonify({
        'success': True,
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat()
    })


# -------------------------
# Auth (placeholder) routes
# -------------------------
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    email = data.get('email')
    name = email.split('@')[0] if email else 'User'

    user = {
        'id': f'user_{uuid.uuid4().hex[:8]}',
        'name': name,
        'email': email or 'demo@sahayak.com',
        'grade': data.get('grade', 'Multi-grade'),
        'subjects': data.get('subjects', ['Mathematics', 'Science', 'English']),
        'region': data.get('region', 'India')
    }

    return jsonify({'success': True, 'user': user, 'token': 'demo-token'}), 200


@app.route('/api/auth/register', methods=['POST'])
def register():
    # For now just echo back a created user with a token
    return login()


# -----------------------------
# Content generation endpoint
# -----------------------------
@app.route('/api/generate/content', methods=['POST'])
def generate_content():
    try:
        data = request.get_json() or {}
        required = ['topic', 'grade', 'subject', 'contentType']
        missing = [f for f in required if not data.get(f)]
        if missing:
            return jsonify({'success': False, 'error': f"Missing fields: {', '.join(missing)}"}), 400

        topic = data['topic']
        grade = data['grade']
        subject = data['subject']
        content_type = data['contentType']
        context = data.get('context', '')
        user_id = data.get('user_id')
        session_id = data.get('session_id')

        print(f"🎯 Generating {content_type} for {subject} (Grade {grade}): {topic}")

        prompt = (
            f"Create a {content_type} for {subject} (grade: {grade}) on '{topic}'. "
            f"Keep it concise, structured, and culturally relevant for Indian students. "
            f"Local context: {context or 'N/A'}."
        )

        messages = [
            {"role": "system", "content": "You are Sahaayak, an educational content generator."},
            {"role": "user", "content": prompt}
        ]

        ai_response, model_used = get_ai_response_with_fallback(messages, task_type="text", user_id=user_id, session_id=session_id)

        if not ai_response:
            print(f"❌ AI response was None. Model used: {model_used}")
            return jsonify({'success': False, 'error': f'AI generation failed: {model_used}'}), 502

        content_item = {
            'id': f'content_{uuid.uuid4().hex[:8]}',
            'title': f"{subject}: {topic}",
            'type': content_type,
            'subject': subject,
            'grade': grade,
            'content': ai_response,
            'createdAt': int(time.time() * 1000),
            'tags': [subject, grade, content_type],
            'model_used': model_used
        }

        print(f"✅ Successfully generated content using {model_used}")
        return jsonify({'success': True, 'content': content_item}), 200

    except Exception as e:
        import traceback
        error_msg = f"Exception in /api/generate/content: {str(e)}"
        print(f"🔴 {error_msg}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': error_msg}), 500


# -----------------------------
# Coaching endpoint
# ----------------------------------------
@app.route('/api/coach/query', methods=['POST'])
def coach_query():
    try:
        data = request.get_json() or {}
        query = data.get('query')
        if not query:
            return jsonify({'success': False, 'error': 'Missing query'}), 400

        user_id = data.get('user_id')
        session_id = data.get('session_id')

        messages = [
            {"role": "system", "content": "You are a concise teaching coach for Indian teachers."},
            {"role": "user", "content": query}
        ]

        ai_response, model_used = get_ai_response_with_fallback(messages, task_type="general", user_id=user_id, session_id=session_id)
        if not ai_response:
            return jsonify({'success': False, 'error': f'AI coaching failed: {model_used}'}), 502

        coaching = {
            'id': f'coach_{uuid.uuid4().hex[:8]}',
            'query': query,
            'category': data.get('category'),
            'response': ai_response,
            'helpful': None,
            'createdAt': int(time.time() * 1000),
            'model_used': model_used
        }
        return jsonify({'success': True, 'coaching': coaching}), 200
    except Exception as e:
        import traceback
        error_msg = f"Exception in /api/coach/query: {str(e)}"
        print(f"🔴 {error_msg}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': error_msg}), 500


# -----------------------------
# Pathways endpoint
# -----------------------------
@app.route('/api/pathways/generate', methods=['POST'])
def pathways_generate():
    data = request.get_json() or {}
    subject = data.get('subject')
    grade = data.get('grade')
    if not subject or not grade:
        return jsonify({'success': False, 'error': 'Missing subject or grade'}), 400

    interests = data.get('interests', [])
    user_id = data.get('user_id')
    session_id = data.get('session_id')

    messages = [
        {"role": "system", "content": "You design short learning pathways with activities."},
        {"role": "user", "content": f"Create a 3-step pathway for {subject}, grade {grade}. Interests: {interests}."}
    ]

    ai_response, model_used = get_ai_response_with_fallback(messages, task_type="general", user_id=user_id, session_id=session_id)
    if not ai_response:
        return jsonify({'success': False, 'error': 'AI pathway generation failed'}), 502

    pathway = {
        'id': f'path_{uuid.uuid4().hex[:8]}',
        'title': f"{subject} pathway for {grade}",
        'description': ai_response,
        'activities': [],
        'culturalElements': [],
        'subject': subject,
        'grade': grade,
        'fullContent': ai_response,
        'model_used': model_used
    }

    return jsonify({'success': True, 'pathway': pathway}), 200


# -----------------------------
# Upload/analyze endpoint
# -----------------------------
@app.route('/api/upload/analyze', methods=['POST'])
def upload_analyze():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'File is required'}), 400

    file = request.files['file']
    analysis_type = request.form.get('analysis_type', 'summary')
    question = request.form.get('question')
    user_id = request.form.get('user_id')
    session_id = request.form.get('session_id')

    try:
        content_bytes = file.read()
        preview_text = content_bytes[:200].decode(errors='ignore') if content_bytes else ''
    except Exception:
        preview_text = ''

    messages = [
        {"role": "system", "content": "You analyze uploaded educational files."},
        {"role": "user", "content": f"Analysis type: {analysis_type}. Question: {question}. Preview: {preview_text}"}
    ]

    ai_response, model_used = get_ai_response_with_fallback(messages, task_type="general", user_id=user_id, session_id=session_id)
    if not ai_response:
        return jsonify({'success': False, 'error': 'AI analysis failed'}), 502

    analysis = {
        'id': f'file_{uuid.uuid4().hex[:8]}',
        'filename': file.filename,
        'type': 'pdf' if file.filename.lower().endswith('.pdf') else 'text',
        'summary': ai_response,
        'analysis': ai_response,
        'textLength': len(content_bytes) if 'content_bytes' in locals() else 0,
        'model_used': model_used
    }

    return jsonify({'success': True, 'analysis': analysis}), 200


if __name__ == '__main__':
    print("🚀 Starting Sahaayak AI Backend...")
    print(f"📍 Running on http://127.0.0.1:5000")
    print(f"🔑 OpenRouter API Key loaded: {bool(os.getenv('OPENROUTER_API_KEY'))}")
    print(f"🔑 A4F API Key loaded: {bool(os.getenv('A4F_API_KEY'))}")
    app.run(host='127.0.0.1', port=5000, debug=True)
