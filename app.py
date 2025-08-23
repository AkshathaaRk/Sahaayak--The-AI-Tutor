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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size for audio

# Enable CORS for Sahayak frontend integration
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"])

# Set up OpenRouter client (using OpenAI library but pointing to OpenRouter)
# Note: API key should be set via environment variable OPENROUTER_API_KEY
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "")
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
            messages=enhanced_messages
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
