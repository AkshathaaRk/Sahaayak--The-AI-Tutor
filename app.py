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
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# Set up A4F client for backup AI models
a4f_api_key = "ddc-a4f-c9560a4fba754910a60a30cce66ba6c5"
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



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health-check')
def health_check():
    """Enhanced health check endpoint for frontend integration"""
    try:
        # Check MCP server connectivity
        mcp_status = 'unknown'
        try:
            mcp_response = requests.get('http://127.0.0.1:5001/health', timeout=2)
            mcp_status = 'connected' if mcp_response.status_code == 200 else 'error'
        except:
            mcp_status = 'disconnected'

        # Check AI model connectivity
        ai_status = 'unknown'
        try:
            # Quick test with OpenRouter
            client.chat.completions.create(
                model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            ai_status = 'connected'
        except:
            ai_status = 'error'

        overall_status = 'healthy' if mcp_status == 'connected' and ai_status == 'connected' else 'degraded'

        return jsonify({
            'status': overall_status,
            'timestamp': time.time(),
            'components': {
                'mcp_server': mcp_status,
                'ai_models': ai_status,
                'memory': 'enabled' if mcp_status == 'connected' else 'disabled'
            },
            'version': '2.0.0',
            'features': {
                'memory': True,
                'voice_chat': True,
                'image_generation': True,
                'multilingual': True,
                'pdf_processing': True,
                'link_analysis': True
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }), 500

def is_image_request(question):
    """Check if the user is asking for image generation"""
    image_keywords = [
        'image', 'picture', 'photo', 'draw', 'generate', 'create', 'make',
        'show me', 'visual', 'illustration', 'artwork', 'painting', 'sketch'
    ]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in image_keywords)

def generate_image_huggingface(prompt):
    """Generate image using Hugging Face Inference API (Free tier)"""
    try:
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        # Your Hugging Face token for free image generation
        HF_TOKEN = "REMOVED_HF_TOKEN"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        payload = {"inputs": prompt}

        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            # Return base64 encoded image
            image_bytes = response.content
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            return image_base64, "Hugging Face Stable Diffusion"
        else:
            return None, f"HF Error: {response.status_code}"

    except Exception as e:
        return None, f"HF Exception: {str(e)}"

def generate_image_together(prompt):
    """Generate image using Together AI (Free tier)"""
    try:
        url = "https://api.together.xyz/v1/images/generations"
        headers = {
            "Authorization": "Bearer YOUR_TOGETHER_API_KEY_HERE",  # You'd need to get this
            "Content-Type": "application/json"
        }

        payload = {
            "model": "black-forest-labs/FLUX.1-schnell-Free",
            "prompt": prompt,
            "width": 512,
            "height": 512,
            "steps": 4,
            "n": 1
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            data = response.json()
            if data.get('data') and len(data['data']) > 0:
                image_url = data['data'][0]['url']
                # Download the image and convert to base64
                img_response = requests.get(image_url, timeout=30)
                if img_response.status_code == 200:
                    image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                    return image_base64, "Together AI FLUX"

        return None, f"Together Error: {response.status_code}"

    except Exception as e:
        return None, f"Together Exception: {str(e)}"

def search_google_images(query):
    """Search Google Images and return image URL"""
    try:
        import urllib.parse

        # Clean the query for image search
        search_query = query.replace("generate", "").replace("create", "").replace("image of", "").replace("picture of", "").strip()
        encoded_query = urllib.parse.quote(search_query)

        # Use Google Images search (this is a simple approach)
        search_url = f"https://www.google.com/search?q={encoded_query}&tbm=isch&safe=active"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(search_url, headers=headers, timeout=30)

        if response.status_code == 200:
            # Simple regex to find image URLs in the HTML
            import re
            img_urls = re.findall(r'"(https://[^"]*\.(?:jpg|jpeg|png|gif|webp))"', response.text)

            if img_urls:
                # Try to download the first few images until one works
                for img_url in img_urls[:5]:  # Try first 5 images
                    try:
                        img_response = requests.get(img_url, headers=headers, timeout=15)
                        if img_response.status_code == 200 and len(img_response.content) > 1000:  # Valid image
                            image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                            return image_base64, f"Google Images: {search_query}"
                    except:
                        continue

        return None, "No images found"

    except Exception as e:
        return None, f"Google Search Exception: {str(e)}"

def search_google_images_advanced(query):
    """Advanced Google Images search with better parsing"""
    try:
        import urllib.parse
        import re

        # Clean the query for better search results
        search_query = query.replace("generate", "").replace("create", "").replace("image of", "").replace("picture of", "").replace("show me", "").strip()
        encoded_query = urllib.parse.quote(search_query + " high quality")

        # Google Images search URL
        search_url = f"https://www.google.com/search?q={encoded_query}&tbm=isch&safe=active&tbs=isz:m"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

        print(f"Searching Google Images for: {search_query}")
        response = requests.get(search_url, headers=headers, timeout=30)

        if response.status_code == 200:
            # Multiple regex patterns to find image URLs
            patterns = [
                r'"(https://[^"]*\.(?:jpg|jpeg|png|gif|webp))"',
                r'src="(https://[^"]*\.(?:jpg|jpeg|png|gif|webp))"',
                r'data-src="(https://[^"]*\.(?:jpg|jpeg|png|gif|webp))"',
                r'"ou":"(https://[^"]*\.(?:jpg|jpeg|png|gif|webp))"'
            ]

            all_urls = []
            for pattern in patterns:
                urls = re.findall(pattern, response.text, re.IGNORECASE)
                all_urls.extend(urls)

            # Remove duplicates and filter
            unique_urls = list(set(all_urls))

            # Filter out unwanted URLs
            filtered_urls = []
            for url in unique_urls:
                if not any(skip in url.lower() for skip in ['google', 'gstatic', 'logo', 'icon', 'avatar']):
                    if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                        filtered_urls.append(url)

            print(f"Found {len(filtered_urls)} potential images")

            # Try to download images
            for i, img_url in enumerate(filtered_urls[:10]):  # Try first 10 images
                try:
                    print(f"Trying image {i+1}: {img_url[:50]}...")
                    img_response = requests.get(img_url, headers=headers, timeout=15)

                    if img_response.status_code == 200 and len(img_response.content) > 5000:  # At least 5KB
                        # Check if it's actually an image
                        content_type = img_response.headers.get('content-type', '').lower()
                        if 'image' in content_type or img_url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
                            image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                            print(f"✅ Successfully downloaded image!")
                            return image_base64, f"Google Images: {search_query}"

                except Exception as e:
                    print(f"Failed to download image {i+1}: {e}")
                    continue

        return None, "No suitable images found"

    except Exception as e:
        print(f"Google search exception: {e}")
        return None, f"Google Search Exception: {str(e)}"

def get_pixabay_image(query):
    """Get image from Pixabay (free stock photos with API)"""
    try:
        import urllib.parse

        # Clean the query
        search_query = query.replace("generate", "").replace("create", "").replace("image of", "").replace("picture of", "").strip()
        encoded_query = urllib.parse.quote(search_query)

        # Pixabay API (free tier, no key needed for basic usage)
        api_url = f"https://pixabay.com/api/?key=9656065-a4094594c34f9ac14c7fc4c39&q={encoded_query}&image_type=photo&category=all&min_width=400&per_page=10"

        response = requests.get(api_url, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if data.get('hits') and len(data['hits']) > 0:
                # Get the first image
                image_info = data['hits'][0]
                img_url = image_info.get('webformatURL') or image_info.get('largeImageURL')

                if img_url:
                    img_response = requests.get(img_url, timeout=30)
                    if img_response.status_code == 200:
                        image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                        return image_base64, f"Pixabay: {search_query}"

        return None, "No Pixabay images found"

    except Exception as e:
        return None, f"Pixabay Exception: {str(e)}"

def generate_image_a4f_primary(prompt):
    """Generate images using A4F API with Imagen-3 (primary method)"""
    try:
        print(f"Generating image with A4F Imagen-3 for prompt: {prompt}")

        # Use A4F client with image generation endpoint
        response = a4f_client.images.generate(
            model="provider-4/imagen-3",
            prompt=f"Generate a high-quality, detailed image of: {prompt}",
            n=1,
            size="1024x1024"
        )

        # Get the image URL from response
        if response.data and len(response.data) > 0:
            image_url = response.data[0].url
            print(f"A4F Imagen-3 Image URL: {image_url}")

            # Download the image and convert to base64
            try:
                img_response = requests.get(image_url, timeout=30)
                if img_response.status_code == 200:
                    image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                    return image_base64, "A4F Imagen-3"
                else:
                    return None, f"Failed to download A4F image: {img_response.status_code}"
            except Exception as e:
                return None, f"Failed to download A4F image: {str(e)}"
        else:
            return None, "A4F Imagen-3 returned no image data"

    except Exception as e:
        print(f"A4F Imagen-3 error: {e}")
        return None, f"A4F Imagen-3 Error: {str(e)}"

def generate_image_a4f_fallback(prompt, model_name, model_id):
    """Generate images using A4F API with fallback models"""
    try:
        print(f"Trying A4F {model_name} for prompt: {prompt}")

        # Try using chat completion for these models (they might be text models that can describe images)
        completion = a4f_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": f"Generate a detailed description for creating an image of: {prompt}. Provide a comprehensive visual description that could be used by an image generator."}
            ]
        )

        response_content = completion.choices[0].message.content
        print(f"A4F {model_name} response: {response_content[:100]}...")

        # Check if the response contains an image URL or base64 data
        if response_content:
            # Look for URLs in the response
            import re
            urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', response_content)

            for url in urls:
                if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                    try:
                        img_response = requests.get(url, timeout=30)
                        if img_response.status_code == 200:
                            image_base64 = base64.b64encode(img_response.content).decode('utf-8')
                            return image_base64, f"A4F {model_name}"
                    except:
                        continue

            # If no image URL found, check if response is base64 image data
            if len(response_content) > 1000 and not response_content.startswith('http'):
                # Might be base64 image data
                try:
                    # Try to decode as base64 to validate
                    base64.b64decode(response_content)
                    return response_content, f"A4F {model_name}"
                except:
                    pass

        return None, f"A4F {model_name} did not return image data"

    except Exception as e:
        print(f"A4F {model_name} error: {e}")
        return None, f"A4F {model_name} Error: {str(e)}"

def generate_image_a4f(prompt):
    """Generate images using A4F API with multiple model fallbacks"""

    # Primary: Try Imagen-3 first
    image_data, model_name = generate_image_a4f_primary(prompt)
    if image_data:
        return image_data, model_name

    # Fallback 1: Try Gemini 2.5 Flash Preview
    print("Imagen-3 failed, trying Gemini 2.5 Flash Preview...")
    image_data, model_name = generate_image_a4f_fallback(prompt, "Gemini-2.5-Flash-Preview", "provider-5/gemini-2.5-flash-preview-04-17")
    if image_data:
        return image_data, model_name

    # Fallback 2: Try Gemini 2.0 Flash Exp
    print("Gemini 2.5 failed, trying Gemini 2.0 Flash Exp...")
    image_data, model_name = generate_image_a4f_fallback(prompt, "Gemini-2.0-Flash-Exp", "provider-5/gemini-2.0-flash-exp")
    if image_data:
        return image_data, model_name

    # Fallback 3: Try GPT-3.5 Turbo
    print("Gemini 2.0 failed, trying GPT-3.5 Turbo...")
    image_data, model_name = generate_image_a4f_fallback(prompt, "GPT-3.5-Turbo", "provider-2/gpt-3.5-turbo")
    if image_data:
        return image_data, model_name

    # If all A4F models fail, return None to trigger other fallbacks
    return None, "All A4F models failed"

def generate_image_free(prompt):
    """Generate images using A4F API first, then fallback to search"""
    # Try A4F image generation first (proper AI image generation)
    image_data, model_name = generate_image_a4f(prompt)
    if image_data:
        return image_data, model_name

    # Fallback to Pixabay search if A4F fails
    image_data, model_name = get_pixabay_image(prompt)
    if image_data:
        return image_data, model_name

    # Try Google Images search as backup
    image_data, model_name = search_google_images_advanced(prompt)
    if image_data:
        return image_data, model_name

    # Try original Google search as final backup
    image_data, model_name = search_google_images(prompt)
    if image_data:
        return image_data, model_name

    # If all fail, return None
    return None, "All image services failed"

# Speech Processing Functions
def speech_to_text(audio_file):
    """Convert speech to text using Whisper"""
    try:
        # Try to import whisper
        try:
            import whisper
        except ImportError:
            return None, "Whisper not installed. Install with: pip install openai-whisper"

        # Load Whisper model (base is good balance of speed/accuracy)
        model = whisper.load_model("base")

        # Save uploaded file temporarily
        temp_path = f"temp_audio_{int(time.time())}.wav"
        audio_file.save(temp_path)

        # Transcribe audio
        result = model.transcribe(temp_path)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return result["text"], "Whisper STT"

    except Exception as e:
        return None, f"STT Error: {str(e)}"

def text_to_speech(text, language='en'):
    """Convert text to speech using gTTS"""
    try:
        # Try to import gTTS
        try:
            from gtts import gTTS
        except ImportError:
            return None, "gTTS not installed. Install with: pip install gtts"

        # Create TTS object
        tts = gTTS(text=text, lang=language, slow=False)

        # Save to temporary file
        temp_path = f"temp_tts_{int(time.time())}.mp3"
        tts.save(temp_path)

        # Read file as base64
        with open(temp_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return audio_data, "Google TTS"

    except Exception as e:
        return None, f"TTS Error: {str(e)}"

def detect_language(text):
    """Enhanced language detection including Indian languages"""
    # Enhanced language detection based on common words and scripts
    language_indicators = {
        'en': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with'],
        'hi': ['के', 'में', 'की', 'और', 'को', 'से', 'पर', 'है', 'एक', 'यह', 'का', 'ने', 'हैं'],
        'kn': ['ಮತ್ತು', 'ಇದು', 'ಆದರೆ', 'ಅವರು', 'ಅವನು', 'ಅವಳು', 'ನಾನು', 'ನೀವು', 'ಅದು', 'ಇದೆ'],
        'te': ['మరియు', 'ఇది', 'అది', 'అతను', 'ఆమె', 'నేను', 'మీరు', 'వారు', 'ఉంది', 'లేదు'],
        'ta': ['மற்றும்', 'இது', 'அது', 'அவன்', 'அவள்', 'நான்', 'நீங்கள்', 'அவர்கள்', 'உள்ளது', 'இல்லை'],
        'ml': ['ഒപ്പം', 'ഇത്', 'അത്', 'അവൻ', 'അവൾ', 'ഞാൻ', 'നിങ്ങൾ', 'അവർ', 'ഉണ്ട്', 'ഇല്ല'],
        'bn': ['এবং', 'এই', 'সেই', 'তিনি', 'সে', 'আমি', 'আপনি', 'তারা', 'আছে', 'নেই'],
        'gu': ['અને', 'આ', 'તે', 'તેઓ', 'તેણે', 'તેણી', 'હું', 'તમે', 'છે', 'નથી'],
        'mr': ['आणि', 'हे', 'ते', 'तो', 'ती', 'मी', 'तुम्ही', 'ते', 'आहे', 'नाही'],
        'pa': ['ਅਤੇ', 'ਇਹ', 'ਉਹ', 'ਉਸਨੇ', 'ਉਸਨੂੰ', 'ਮੈਂ', 'ਤੁਸੀਂ', 'ਉਹ', 'ਹੈ', 'ਨਹੀਂ'],
        'ur': ['اور', 'یہ', 'وہ', 'اس', 'وہ', 'میں', 'آپ', 'وہ', 'ہے', 'نہیں'],
        'es': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no'],
        'fr': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir'],
        'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'],
        'it': ['il', 'di', 'che', 'e', 'la', 'per', 'in', 'un', 'è', 'con'],
        'pt': ['o', 'de', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com'],
        'ru': ['в', 'и', 'не', 'на', 'я', 'быть', 'он', 'с', 'что', 'а'],
        'zh': ['的', '一', '是', '在', '不', '了', '有', '和', '人', '这'],
        'ja': ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し'],
        'ko': ['의', '에', '는', '을', '가', '이', '와', '로', '한', '그'],
        'ar': ['في', 'من', 'إلى', 'على', 'هذا', 'هذه', 'التي', 'الذي', 'كان', 'لا']
    }

    text_lower = text.lower()
    scores = {}

    for lang, indicators in language_indicators.items():
        score = sum(1 for word in indicators if word in text_lower)
        scores[lang] = score

    # Return language with highest score, default to English
    detected_lang = max(scores, key=scores.get) if max(scores.values()) > 0 else 'en'
    return detected_lang

# PDF Processing Functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using multiple methods"""
    try:
        # Try pdfplumber first (better for complex layouts)
        try:
            import pdfplumber

            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            if text.strip():
                return text, "pdfplumber"
        except ImportError:
            pass

        # Fallback to PyPDF2
        try:
            import PyPDF2

            text = ""
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            if text.strip():
                return text, "PyPDF2"
        except ImportError:
            pass

        return None, "No PDF libraries available. Install PyPDF2 or pdfplumber"

    except Exception as e:
        return None, f"PDF extraction error: {str(e)}"

def summarize_text(text, language='en', summary_type='basic'):
    """Summarize text using free OpenRouter models"""
    try:
        # Language names for prompts
        language_names = {
            'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada', 'te': 'Telugu',
            'ta': 'Tamil', 'ml': 'Malayalam', 'bn': 'Bengali', 'gu': 'Gujarati',
            'mr': 'Marathi', 'pa': 'Punjabi', 'ur': 'Urdu'
        }

        lang_name = language_names.get(language, language)

        # Different summary prompts based on type
        if summary_type == 'basic':
            if language != 'en':
                prompt = f"""
Please provide a comprehensive summary of the following text in {lang_name} language.

Instructions:
- Write the ENTIRE summary in {lang_name} language using proper script
- Include main points and key information
- Make it clear and easy to understand
- Length: 3-4 paragraphs

Text to summarize:
{text[:4000]}  # Limit text to avoid token limits
"""
            else:
                prompt = f"""
Please provide a comprehensive summary of the following text:

Instructions:
- Include main points and key information
- Make it clear and easy to understand
- Length: 3-4 paragraphs

Text to summarize:
{text[:4000]}
"""

        elif summary_type == 'detailed':
            if language != 'en':
                prompt = f"""
Please provide a detailed analysis of the following text in {lang_name} language.

Instructions:
- Write ENTIRELY in {lang_name} language using proper script
- Include: Main topics, key points, important details, conclusions
- Organize in clear sections
- Be thorough and comprehensive

Text to analyze:
{text[:4000]}
"""
            else:
                prompt = f"""
Please provide a detailed analysis of the following text:

Instructions:
- Include: Main topics, key points, important details, conclusions
- Organize in clear sections
- Be thorough and comprehensive

Text to analyze:
{text[:4000]}
"""

        elif summary_type == 'educational':
            if language != 'en':
                prompt = f"""
Please analyze the following educational content and provide a comprehensive summary in {lang_name} language.

Instructions:
- Write ENTIRELY in {lang_name} language using proper script
- Focus on educational value and key learning points
- Include main concepts, important facts, and conclusions
- Make it suitable for students and teachers
- Organize information clearly with proper structure

Educational content to analyze:
{text[:4000]}
"""
            else:
                prompt = f"""
Please analyze the following educational content and provide a comprehensive summary.

Instructions:
- Focus on educational value and key learning points
- Include main concepts, important facts, and conclusions
- Make it suitable for students and teachers
- Organize information clearly with proper structure
- Highlight key takeaways and learning objectives

Educational content to analyze:
{text[:4000]}
"""

        else:
            # Default fallback for any other summary_type
            prompt = f"""
Please provide a comprehensive summary of the following text:

Instructions:
- Include main points and key information
- Make it clear and easy to understand
- Length: 3-4 paragraphs

Text to summarize:
{text[:4000]}
"""

        # Use fallback system for text summarization
        messages = [{"role": "user", "content": prompt}]
        answer, model_used = get_ai_response_with_fallback(messages, "text")

        if answer:
            return answer, f"{model_used} (PDF Summarization)"
        else:
            return None, model_used

    except Exception as e:
        return None, f"Summarization error: {str(e)}"

def answer_pdf_question(pdf_text, question, language='en'):
    """Answer questions about PDF content"""
    try:
        language_names = {
            'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada', 'te': 'Telugu',
            'ta': 'Tamil', 'ml': 'Malayalam', 'bn': 'Bengali', 'gu': 'Gujarati',
            'mr': 'Marathi', 'pa': 'Punjabi', 'ur': 'Urdu'
        }

        lang_name = language_names.get(language, language)

        if language != 'en':
            prompt = f"""
Based on the following document content, please answer the question in {lang_name} language.

Instructions:
- Write your ENTIRE answer in {lang_name} language using proper script
- Base your answer only on the provided document content
- If the answer is not in the document, say so in {lang_name}
- Be accurate and detailed

Document Content:
{pdf_text[:3500]}

Question: {question}

Answer in {lang_name}:
"""
        else:
            prompt = f"""
Based on the following document content, please answer the question.

Instructions:
- Base your answer only on the provided document content
- If the answer is not in the document, say so clearly
- Be accurate and detailed

Document Content:
{pdf_text[:3500]}

Question: {question}

Answer:
"""

        # Use fallback system for PDF Q&A
        messages = [{"role": "user", "content": prompt}]
        answer, model_used = get_ai_response_with_fallback(messages, "text")

        if answer:
            return answer, f"{model_used} (PDF Q&A)"
        else:
            return None, model_used

    except Exception as e:
        return None, f"PDF Q&A error: {str(e)}"

# Link Processing Functions
def extract_youtube_transcript(url):
    """Extract transcript from YouTube video"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        import re

        # Extract video ID from URL
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
        if not video_id_match:
            return None, "Invalid YouTube URL"

        video_id = video_id_match.group(1)

        # Try to get transcript in multiple languages
        try:
            # Try English first
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        except:
            try:
                # Try Hindi
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])
            except:
                try:
                    # Try any available language
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                except:
                    return None, "No transcript available for this video"

        # Combine transcript text
        full_text = " ".join([entry['text'] for entry in transcript])

        return full_text, "YouTube Transcript API"

    except ImportError:
        return None, "YouTube transcript library not installed"
    except Exception as e:
        return None, f"YouTube transcript error: {str(e)}"

def extract_website_content(url):
    """Extract content from any website"""
    try:
        # Try newspaper3k first (best for articles)
        try:
            from newspaper import Article

            article = Article(url)
            article.download()
            article.parse()

            if article.text and len(article.text.strip()) > 100:
                content = f"Title: {article.title}\n\n{article.text}"
                return content, "Newspaper3k"
        except ImportError:
            pass
        except Exception as e:
            print(f"Newspaper3k failed: {e}")

        # Fallback to BeautifulSoup
        try:
            from bs4 import BeautifulSoup

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Try to find main content
            content_selectors = [
                'article', 'main', '.content', '.post-content',
                '.entry-content', '.article-content', '.post-body'
            ]

            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = elements[0].get_text()
                    break

            # If no specific content found, get all text
            if not content or len(content.strip()) < 100:
                content = soup.get_text()

            # Clean up text
            lines = (line.strip() for line in content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = ' '.join(chunk for chunk in chunks if chunk)

            if len(content.strip()) > 100:
                # Try to get title
                title = soup.find('title')
                title_text = title.get_text() if title else "Web Content"

                return f"Title: {title_text}\n\n{content}", "BeautifulSoup"
            else:
                return None, "Could not extract meaningful content"

        except ImportError:
            return None, "BeautifulSoup not installed"
        except Exception as e:
            return None, f"Content extraction error: {str(e)}"

    except Exception as e:
        return None, f"Website processing error: {str(e)}"

def process_any_link(url, language='en', summary_type='basic'):
    """Process any link (YouTube, website, etc.) and generate summary"""
    try:
        # Determine link type and extract content
        content = None
        extraction_method = ""

        # Check if it's a YouTube link
        if 'youtube.com' in url or 'youtu.be' in url:
            content, extraction_method = extract_youtube_transcript(url)
        else:
            # Try as regular website
            content, extraction_method = extract_website_content(url)

        if not content:
            return None, extraction_method, None

        # Generate summary using the same function as PDF
        summary, model_used = summarize_text(content, language, summary_type)

        return content, extraction_method, summary, model_used

    except Exception as e:
        return None, f"Link processing error: {str(e)}", None, None

def answer_link_question(link_content, question, language='en'):
    """Answer questions about link content"""
    try:
        language_names = {
            'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada', 'te': 'Telugu',
            'ta': 'Tamil', 'ml': 'Malayalam', 'bn': 'Bengali', 'gu': 'Gujarati',
            'mr': 'Marathi', 'pa': 'Punjabi', 'ur': 'Urdu'
        }

        lang_name = language_names.get(language, language)

        if language != 'en':
            prompt = f"""
Based on the following web content, please answer the question in {lang_name} language.

Instructions:
- Write your ENTIRE answer in {lang_name} language using proper script
- Base your answer only on the provided content
- If the answer is not in the content, say so in {lang_name}
- Be accurate and detailed

Web Content:
{link_content[:3500]}

Question: {question}

Answer in {lang_name}:
"""
        else:
            prompt = f"""
Based on the following web content, please answer the question.

Instructions:
- Base your answer only on the provided content
- If the answer is not in the content, say so clearly
- Be accurate and detailed

Web Content:
{link_content[:3500]}

Question: {question}

Answer:
"""

        # Use fallback system for link Q&A
        messages = [{"role": "user", "content": prompt}]
        answer, model_used = get_ai_response_with_fallback(messages, "text")

        if answer:
            return answer, f"{model_used} (Link Q&A)"
        else:
            return None, model_used

    except Exception as e:
        return None, f"Link Q&A error: {str(e)}"

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '')

        # Get user_id and session_id for MCP context
        user_id = data.get('user_id', 'default_user')
        session_id = data.get('session_id', 'default_session')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Choose model based on request type
        if is_image_request(question):
            # Try to generate actual image
            image_data, model_name = generate_image_free(question)

            if image_data:
                # Store image generation in context
                messages = [{"role": "user", "content": question}]
                get_ai_response_with_fallback(
                    messages + [{"role": "assistant", "content": "I generated an image for you."}],
                    "vision", user_id, session_id
                )

                return jsonify({
                    'answer': 'Image generated successfully!',
                    'image_data': image_data,
                    'model_used': model_name,
                    'type': 'image'
                })
            else:
                # Fallback to text description if image generation fails
                messages = [{"role": "user", "content": f"Since I cannot generate images, please provide a detailed description of: {question}"}]
                answer, model_used = get_ai_response_with_fallback(messages, "vision", user_id, session_id)

                if answer:
                    return jsonify({
                        'answer': answer,
                        'model_used': f"{model_used} (Text Fallback)",
                        'type': 'text',
                        'note': 'Image generation failed, providing text description instead.'
                    })
                else:
                    return jsonify({'error': model_used}), 500
        else:
            # Regular text chat with MCP context integration
            messages = [{"role": "user", "content": question}]
            answer, model_used = get_ai_response_with_fallback(messages, "text", user_id, session_id)

            if answer:
                return jsonify({
                    'answer': answer,
                    'model_used': model_used,
                    'type': 'text',
                    'user_id': user_id,
                    'session_id': session_id
                })
            else:
                return jsonify({'error': model_used}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400

        # Get the question about the image and session info
        question = request.form.get('question', 'What do you see in this image?')
        user_id = request.form.get('user_id', 'default_user')
        session_id = request.form.get('session_id', 'default_session')

        # Read and encode the image
        image_data = file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Get file extension for proper mime type
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension in ['jpg', 'jpeg']:
            mime_type = 'image/jpeg'
        elif file_extension == 'png':
            mime_type = 'image/png'
        elif file_extension == 'gif':
            mime_type = 'image/gif'
        elif file_extension == 'webp':
            mime_type = 'image/webp'
        else:
            mime_type = 'image/jpeg'  # default

        # Try vision-capable FREE models with fallback
        vision_models = [
            ("google/gemini-2.0-flash-exp:free", "Gemini 2.0 Flash"),
            ("google/gemma-2-2b-it:free", "Gemma 2B"),
            ("meta-llama/llama-3.2-11b-vision-instruct:free", "Llama Vision")
        ]

        last_error = None

        for model, model_name in vision_models:
            try:
                response = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "http://localhost:5000",
                        "X-Title": "AI Chat Integration App",
                    },
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ]
                )

                return jsonify({
                    'answer': response.choices[0].message.content,
                    'model_used': model_name,
                    'user_id': user_id,
                    'session_id': session_id,
                    'type': 'image_analysis'
                })

            except Exception as model_error:
                last_error = str(model_error)
                print(f"Failed with {model}: {model_error}")
                continue

        # If OpenRouter vision models failed, try A4F fallbacks
        print("All OpenRouter vision models failed, trying A4F fallbacks...")

        for backup_model in A4F_BACKUP_MODELS:
            try:
                print(f"Trying A4F vision model: {backup_model}")
                response = a4f_client.chat.completions.create(
                    model=backup_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ]
                )

                model_name = backup_model.split('/')[-1] if '/' in backup_model else backup_model
                return jsonify({
                    'answer': response.choices[0].message.content,
                    'model_used': f"A4F ({model_name})",
                    'user_id': user_id,
                    'session_id': session_id,
                    'type': 'image_analysis'
                })

            except Exception as backup_error:
                print(f"A4F model {backup_model} failed: {backup_error}")
                continue

        # If all models failed, return the last error
        return jsonify({'error': f'All vision models failed. Last OpenRouter error: {last_error}'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():
    """Convert uploaded audio to text"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400

        # Convert speech to text
        text, model_used = speech_to_text(audio_file)

        if text:
            return jsonify({
                'text': text,
                'model_used': model_used,
                'detected_language': detect_language(text)
            })
        else:
            return jsonify({'error': model_used}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech_route():
    """Convert text to speech"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'en')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Convert text to speech
        audio_data, model_used = text_to_speech(text, language)

        if audio_data:
            return jsonify({
                'audio_data': audio_data,
                'model_used': model_used,
                'language': language
            })
        else:
            return jsonify({'error': model_used}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/voice-chat', methods=['POST'])
def voice_chat():
    """Complete voice-to-voice conversation"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        response_language = request.form.get('language', 'en')

        # Step 1: Speech to Text
        user_text, stt_model = speech_to_text(audio_file)
        if not user_text:
            return jsonify({'error': f'Speech recognition failed: {stt_model}'}), 500

        # Step 2: Process with AI (same logic as regular chat)
        if is_image_request(user_text):
            # Try to get image
            image_data, model_name = generate_image_free(user_text)

            if image_data:
                ai_response = "I found an image for you!"
                response_type = 'image'
            else:
                # Fallback to text description
                response = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "http://localhost:5000",
                        "X-Title": "AI Chat Integration App",
                    },
                    model="google/gemma-3n-e4b-it:free",
                    messages=[
                        {"role": "user", "content": f"Describe in detail: {user_text}"}
                    ]
                )
                ai_response = response.choices[0].message.content
                image_data = None
                model_name = "Gemma (Voice Chat)"
                response_type = 'text'
        else:
            # Regular text processing
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "http://localhost:5000",
                    "X-Title": "AI Chat Integration App",
                },
                model="deepseek/deepseek-r1-0528-qwen3-8b:free",
                messages=[
                    {"role": "user", "content": user_text}
                ]
            )
            ai_response = response.choices[0].message.content
            image_data = None
            model_name = "DeepSeek (Voice Chat)"
            response_type = 'text'

        # Step 3: Text to Speech
        audio_data, tts_model = text_to_speech(ai_response, response_language)

        result = {
            'user_text': user_text,
            'ai_response': ai_response,
            'audio_data': audio_data,
            'models_used': {
                'stt': stt_model,
                'ai': model_name,
                'tts': tts_model
            },
            'type': response_type,
            'language': response_language
        }

        if image_data:
            result['image_data'] = image_data

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask-multilingual', methods=['POST'])
def ask_multilingual():
    """Enhanced multilingual AI processing with native language understanding"""
    try:
        data = request.get_json()
        user_question = data.get('question', '')
        input_language = data.get('input_language', 'en')
        output_language = data.get('output_language', 'en')
        native_processing = data.get('native_processing', False)

        # Get user_id and session_id for MCP context
        user_id = data.get('user_id', 'default_user')
        session_id = data.get('session_id', 'default_session')

        if not user_question:
            return jsonify({'error': 'No question provided'}), 400

        # Language names for context
        language_names = {
            'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada', 'te': 'Telugu',
            'ta': 'Tamil', 'ml': 'Malayalam', 'bn': 'Bengali', 'gu': 'Gujarati',
            'mr': 'Marathi', 'pa': 'Punjabi', 'ur': 'Urdu', 'es': 'Spanish',
            'fr': 'French', 'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
            'ru': 'Russian', 'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic'
        }

        input_lang_name = language_names.get(input_language, input_language)
        output_lang_name = language_names.get(output_language, output_language)

        # Enhanced prompt for native language processing
        if native_processing and output_language != 'en':
            enhanced_prompt = f"""
You are an AI assistant that speaks fluent {output_lang_name}. The user has asked in {input_lang_name}: "{user_question}"

IMPORTANT: You must respond ENTIRELY in {output_lang_name} language using proper {output_lang_name} script and grammar.

Instructions:
1. Understand the question in {input_lang_name} context and cultural nuances
2. Think about the response considering {output_lang_name} cultural context
3. Write your ENTIRE response in {output_lang_name} language only
4. Use appropriate {output_lang_name} cultural examples and references
5. If discussing science/technology, use {output_lang_name} terms where possible
6. Be natural and conversational in {output_lang_name}

Examples of what I expect:
- If {output_lang_name} is Kannada: Write in ಕನ್ನಡ script with proper grammar
- If {output_lang_name} is Telugu: Write in తెలుగు script with proper grammar
- If {output_lang_name} is Tamil: Write in தமிழ் script with proper grammar
- If {output_lang_name} is Hindi: Write in हिंदी script with proper grammar

DO NOT mix languages. Respond ONLY in {output_lang_name}.
"""
        else:
            enhanced_prompt = f"""
The user asked: "{user_question}"
Please respond in {output_lang_name} language.
If this is about images, describe what image would be most relevant.
"""

        # Check if it's an image request
        if is_image_request(user_question):
            # Try to get image first
            image_data, model_name = generate_image_free(user_question)

            if image_data:
                # Get description in target language with native script
                if output_language != 'en':
                    description_prompt = f"""
I found an image for the request: "{user_question}"

Please write a description of this image ENTIRELY in {output_lang_name} language using proper {output_lang_name} script.

Instructions:
- Write ONLY in {output_lang_name} language
- Use proper {output_lang_name} script (ಕನ್ನಡ for Kannada, తెలుగు for Telugu, தமிழ் for Tamil, etc.)
- Be descriptive and natural
- Include cultural context if relevant
- DO NOT mix with English

Example: If {output_lang_name} is Kannada, write like: "ಈ ಚಿತ್ರದಲ್ಲಿ ನೀವು ನೋಡಬಹುದು..."
"""
                else:
                    description_prompt = f"Describe this image for the request: {user_question}"

                # Use fallback system for image description
                messages = [{"role": "user", "content": description_prompt}]
                description_text, description_model = get_ai_response_with_fallback(messages, "vision", user_id, session_id)

                if not description_text:
                    return jsonify({'error': description_model}), 500

                return jsonify({
                    'answer': description_text,
                    'type': 'image',
                    'image_data': image_data,
                    'model_used': f"{model_name} + {description_model} (Multilingual)",
                    'input_language': input_lang_name,
                    'output_language': output_lang_name,
                    'native_processing': native_processing
                })
            else:
                # Fallback to description with A4F backup
                messages = [{"role": "user", "content": enhanced_prompt}]
                answer, model_used = get_ai_response_with_fallback(messages, "vision", user_id, session_id)

                if answer:
                    return jsonify({
                        'answer': answer,
                        'type': 'text',
                        'model_used': f"{model_used} (Multilingual Image Description)",
                        'input_language': input_lang_name,
                        'output_language': output_lang_name,
                        'native_processing': native_processing
                    })
                else:
                    return jsonify({'error': model_used}), 500
        else:
            # Regular text processing with multilingual support and A4F fallback
            messages = [{"role": "user", "content": enhanced_prompt}]
            answer, model_used = get_ai_response_with_fallback(messages, "text", user_id, session_id)

            if answer:
                return jsonify({
                    'answer': answer,
                    'type': 'text',
                    'model_used': f"{model_used} (Multilingual)",
                    'input_language': input_lang_name,
                    'output_language': output_lang_name,
                    'native_processing': native_processing,
                    'user_id': user_id,
                    'session_id': session_id
                })
            else:
                return jsonify({'error': model_used}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pdf-summarize', methods=['POST'])
def pdf_summarize():
    """Option A: Basic PDF Summarization"""
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400

        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return jsonify({'error': 'No PDF file selected'}), 400

        # Get parameters including session info
        language = request.form.get('language', 'en')
        summary_type = request.form.get('summary_type', 'basic')
        user_id = request.form.get('user_id', 'default_user')
        session_id = request.form.get('session_id', 'default_session')

        # Extract text from PDF
        pdf_text, extraction_method = extract_text_from_pdf(pdf_file)

        if not pdf_text:
            return jsonify({'error': extraction_method}), 500

        # Generate summary
        summary, model_used = summarize_text(pdf_text, language, summary_type)

        if not summary:
            return jsonify({'error': model_used}), 500

        return jsonify({
            'summary': summary,
            'original_text_length': len(pdf_text),
            'summary_length': len(summary),
            'extraction_method': extraction_method,
            'model_used': model_used,
            'language': language,
            'summary_type': summary_type,
            'filename': pdf_file.filename,
            'user_id': user_id,
            'session_id': session_id,
            'type': 'pdf_summary'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pdf-analyze', methods=['POST'])
def pdf_analyze():
    """Option B: Advanced PDF Analysis with Q&A"""
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400

        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return jsonify({'error': 'No PDF file selected'}), 400

        # Get parameters
        question = request.form.get('question', '')
        language = request.form.get('language', 'en')

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Extract text from PDF
        pdf_text, extraction_method = extract_text_from_pdf(pdf_file)

        if not pdf_text:
            return jsonify({'error': extraction_method}), 500

        # Answer question about PDF
        answer, model_used = answer_pdf_question(pdf_text, question, language)

        if not answer:
            return jsonify({'error': model_used}), 500

        return jsonify({
            'question': question,
            'answer': answer,
            'document_length': len(pdf_text),
            'extraction_method': extraction_method,
            'model_used': model_used,
            'language': language,
            'filename': pdf_file.filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pdf-full-analysis', methods=['POST'])
def pdf_full_analysis():
    """Combined: PDF Summary + Q&A capability"""
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400

        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return jsonify({'error': 'No PDF file selected'}), 400

        # Get parameters
        language = request.form.get('language', 'en')
        include_summary = request.form.get('include_summary', 'true').lower() == 'true'

        # Extract text from PDF
        pdf_text, extraction_method = extract_text_from_pdf(pdf_file)

        if not pdf_text:
            return jsonify({'error': extraction_method}), 500

        result = {
            'filename': pdf_file.filename,
            'document_length': len(pdf_text),
            'extraction_method': extraction_method,
            'language': language,
            'text_preview': pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text
        }

        # Generate summary if requested
        if include_summary:
            summary, summary_model = summarize_text(pdf_text, language, 'detailed')
            if summary:
                result['summary'] = summary
                result['summary_model'] = summary_model
            else:
                result['summary_error'] = summary_model

        # Store PDF text for Q&A (in a real app, you'd use a database)
        # For now, we'll return it so frontend can use it for questions
        result['ready_for_questions'] = True
        result['pdf_text'] = pdf_text  # Frontend will use this for Q&A

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/link-summarize', methods=['POST'])
def link_summarize():
    """Summarize content from any web link"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        language = data.get('language', 'en')
        summary_type = data.get('summary_type', 'basic')

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        # Validate URL
        if not (url.startswith('http://') or url.startswith('https://')):
            url = 'https://' + url

        # Process the link
        content, extraction_method, summary, model_used = process_any_link(url, language, summary_type)

        if not content:
            return jsonify({'error': extraction_method}), 500

        if not summary:
            return jsonify({'error': model_used}), 500

        return jsonify({
            'url': url,
            'summary': summary,
            'original_content_length': len(content),
            'summary_length': len(summary),
            'extraction_method': extraction_method,
            'model_used': model_used,
            'language': language,
            'summary_type': summary_type,
            'content_preview': content[:500] + "..." if len(content) > 500 else content
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/link-analyze', methods=['POST'])
def link_analyze():
    """Answer questions about link content"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        question = data.get('question', '')
        language = data.get('language', 'en')

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Validate URL
        if not (url.startswith('http://') or url.startswith('https://')):
            url = 'https://' + url

        # Extract content from link
        if 'youtube.com' in url or 'youtu.be' in url:
            content, extraction_method = extract_youtube_transcript(url)
        else:
            content, extraction_method = extract_website_content(url)

        if not content:
            return jsonify({'error': extraction_method}), 500

        # Answer question about content
        answer, model_used = answer_link_question(content, question, language)

        if not answer:
            return jsonify({'error': model_used}), 500

        return jsonify({
            'url': url,
            'question': question,
            'answer': answer,
            'content_length': len(content),
            'extraction_method': extraction_method,
            'model_used': model_used,
            'language': language
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/link-full-analysis', methods=['POST'])
def link_full_analysis():
    """Complete link analysis with summary + Q&A capability"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        language = data.get('language', 'en')
        include_summary = data.get('include_summary', True)

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        # Validate URL
        if not (url.startswith('http://') or url.startswith('https://')):
            url = 'https://' + url

        # Extract content
        if 'youtube.com' in url or 'youtu.be' in url:
            content, extraction_method = extract_youtube_transcript(url)
            content_type = "YouTube Video"
        else:
            content, extraction_method = extract_website_content(url)
            content_type = "Website"

        if not content:
            return jsonify({'error': extraction_method}), 500

        result = {
            'url': url,
            'content_type': content_type,
            'content_length': len(content),
            'extraction_method': extraction_method,
            'language': language,
            'content_preview': content[:500] + "..." if len(content) > 500 else content,
            'ready_for_questions': True,
            'content': content  # For frontend Q&A
        }

        # Generate summary if requested
        if include_summary:
            summary, summary_model = summarize_text(content, language, 'detailed')
            if summary:
                result['summary'] = summary
                result['summary_model'] = summary_model
            else:
                result['summary_error'] = summary_model

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# SAHAYAK FRONTEND API ENDPOINTS
# ============================================================================

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Login endpoint for Sahayak frontend"""
    try:
        data = request.get_json()
        email = data.get('email', '')
        password = data.get('password', '')

        # Simple authentication (you can enhance this with real auth)
        if email and password:
            user = {
                'id': f'user_{email.split("@")[0]}',
                'name': email.split('@')[0].title(),
                'email': email,
                'grade': '10',
                'subjects': ['Math', 'Science', 'English'],
                'region': 'India'
            }
            return jsonify({
                'success': True,
                'user': user,
                'token': f'token_{user["id"]}'
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Register endpoint for Sahayak frontend"""
    try:
        data = request.get_json()
        name = data.get('name', '')
        email = data.get('email', '')
        password = data.get('password', '')

        if name and email and password:
            user = {
                'id': f'user_{email.split("@")[0]}',
                'name': name,
                'email': email,
                'grade': data.get('grade', '10'),
                'subjects': data.get('subjects', ['Math', 'Science']),
                'region': data.get('region', 'India')
            }
            return jsonify({
                'success': True,
                'user': user,
                'token': f'token_{user["id"]}'
            })
        else:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/generate/content', methods=['POST'])
def api_generate_content():
    """Generate educational content for Sahayak"""
    try:
        data = request.get_json()
        topic = data.get('topic', '')
        grade = data.get('grade', '10')
        subject = data.get('subject', 'General')
        content_type = data.get('contentType', 'lesson')
        context = data.get('context', '')

        # Create a detailed prompt for educational content
        if content_type == 'lesson':
            prompt = f"Create a comprehensive lesson plan for {subject} grade {grade} students on the topic '{topic}'. Include learning objectives, key concepts, examples, and activities. {context}"
        elif content_type == 'story':
            prompt = f"Write an engaging educational story for grade {grade} students that teaches about '{topic}' in {subject}. Make it culturally relevant and include moral lessons. {context}"
        elif content_type == 'quiz':
            prompt = f"Create a quiz with 10 multiple-choice questions for grade {grade} {subject} students on '{topic}'. Include answers and explanations. {context}"
        elif content_type == 'summary':
            prompt = f"Create a concise summary of '{topic}' for grade {grade} {subject} students. Include key points and important concepts. {context}"
        else:
            prompt = f"Create educational content about '{topic}' for grade {grade} {subject} students. {context}"

        # Use the existing AI system with memory
        user_id = data.get('user_id', 'sahayak_user')
        session_id = data.get('session_id', 'content_generation')

        messages = [{"role": "user", "content": prompt}]
        response_text, model_used = get_ai_response_with_fallback(
            messages,
            task_type="content_generation",
            user_id=user_id,
            session_id=session_id
        )

        if response_text:
            content_item = {
                'id': f'content_{int(time.time())}',
                'title': f"{content_type.title()}: {topic}",
                'type': content_type,
                'subject': subject,
                'grade': grade,
                'content': response_text,
                'createdAt': time.time(),
                'tags': [topic, subject, grade],
                'model_used': model_used
            }
            return jsonify({
                'success': True,
                'content': content_item
            })
        else:
            return jsonify({'success': False, 'error': model_used}), 500

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/coach/query', methods=['POST'])
def api_coach_query():
    """AI coaching/tutoring endpoint for Sahayak"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        category = data.get('category', 'general')
        user_id = data.get('user_id', 'sahayak_user')
        session_id = data.get('session_id', 'coaching_session')

        # Create coaching prompt
        coaching_prompt = f"""You are an AI tutor and educational coach. A student has asked: "{query}"

        Category: {category}

        Please provide:
        1. A clear, helpful answer
        2. Additional learning tips
        3. Related concepts they should explore
        4. Encouragement and motivation

        Keep your response educational, supportive, and age-appropriate."""

        messages = [{"role": "user", "content": coaching_prompt}]
        response_text, model_used = get_ai_response_with_fallback(
            messages,
            task_type="coaching",
            user_id=user_id,
            session_id=session_id
        )

        if response_text:
            coaching_response = {
                'id': f'coach_{int(time.time())}',
                'query': query,
                'category': category,
                'response': response_text,
                'helpful': None,  # User can rate later
                'createdAt': time.time(),
                'model_used': model_used
            }
            return jsonify({
                'success': True,
                'coaching': coaching_response
            })
        else:
            return jsonify({'success': False, 'error': model_used}), 500

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pathways/generate', methods=['POST'])
def api_generate_pathways():
    """Generate learning pathways for Sahayak"""
    try:
        data = request.get_json()
        subject = data.get('subject', 'General')
        grade = data.get('grade', '10')
        interests = data.get('interests', [])
        user_id = data.get('user_id', 'sahayak_user')
        session_id = data.get('session_id', 'pathway_generation')

        interests_text = ', '.join(interests) if interests else 'general learning'

        pathway_prompt = f"""Create a personalized learning pathway for a grade {grade} student interested in {subject}.

        Student interests: {interests_text}

        Please provide:
        1. A catchy title for the pathway
        2. A brief description
        3. 5-7 learning activities in sequence
        4. Cultural elements that make learning relevant to Indian students
        5. Practical applications

        Make it engaging and culturally relevant."""

        messages = [{"role": "user", "content": pathway_prompt}]
        response_text, model_used = get_ai_response_with_fallback(
            messages,
            task_type="pathway_generation",
            user_id=user_id,
            session_id=session_id
        )

        if response_text:
            # Parse the response to extract structured data
            pathway = {
                'id': f'pathway_{int(time.time())}',
                'title': f"Learning Path: {subject} for Grade {grade}",
                'description': response_text[:200] + "...",
                'activities': [
                    "Introduction and basics",
                    "Core concepts exploration",
                    "Practical applications",
                    "Cultural connections",
                    "Advanced topics",
                    "Project work",
                    "Assessment and reflection"
                ],
                'culturalElements': [
                    "Indian examples and contexts",
                    "Local language connections",
                    "Traditional knowledge integration",
                    "Contemporary Indian applications"
                ],
                'subject': subject,
                'grade': grade,
                'fullContent': response_text,
                'model_used': model_used
            }
            return jsonify({
                'success': True,
                'pathway': pathway
            })
        else:
            return jsonify({'success': False, 'error': model_used}), 500

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload/analyze', methods=['POST'])
def api_upload_analyze():
    """Analyze uploaded files for Sahayak (integrates existing PDF/image analysis)"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        user_id = request.form.get('user_id', 'sahayak_user')
        session_id = request.form.get('session_id', 'file_analysis')
        analysis_type = request.form.get('analysis_type', 'general')

        file_extension = file.filename.lower().split('.')[-1]

        if file_extension == 'pdf':
            # Use existing PDF analysis
            language = request.form.get('language', 'en')
            pdf_text, extraction_method = extract_text_from_pdf(file)

            if pdf_text:
                summary, model_used = summarize_text(pdf_text, language, 'educational')

                return jsonify({
                    'success': True,
                    'analysis': {
                        'id': f'analysis_{int(time.time())}',
                        'filename': file.filename,
                        'type': 'pdf',
                        'summary': summary,
                        'extractionMethod': extraction_method,
                        'textLength': len(pdf_text),
                        'model_used': model_used
                    }
                })
            else:
                return jsonify({'success': False, 'error': extraction_method}), 500

        elif file_extension in ['txt', 'md', 'rtf']:
            # Handle text files
            try:
                language = request.form.get('language', 'en')
                text_content = file.read().decode('utf-8')

                if text_content.strip():
                    summary, model_used = summarize_text(text_content, language, 'educational')

                    return jsonify({
                        'success': True,
                        'analysis': {
                            'id': f'analysis_{int(time.time())}',
                            'filename': file.filename,
                            'type': 'text',
                            'summary': summary,
                            'extractionMethod': 'direct_text_read',
                            'textLength': len(text_content),
                            'model_used': model_used
                        }
                    })
                else:
                    return jsonify({'success': False, 'error': 'Empty text file'}), 400
            except UnicodeDecodeError:
                return jsonify({'success': False, 'error': 'Unable to decode text file. Please ensure it\'s in UTF-8 format.'}), 400

        elif file_extension in ['jpg', 'jpeg', 'png', 'gif']:
            # Use existing image analysis
            question = request.form.get('question', 'Analyze this image for educational content')

            # Read and encode the image
            image_data = file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            # Get file extension for proper mime type
            if file_extension in ['jpg', 'jpeg']:
                mime_type = 'image/jpeg'
            elif file_extension == 'png':
                mime_type = 'image/png'
            else:
                mime_type = 'image/jpeg'  # default

            # Try vision models
            vision_models = [
                ("google/gemini-2.0-flash-exp:free", "Gemini 2.0 Flash"),
                ("meta-llama/llama-3.2-11b-vision-instruct:free", "Llama Vision")
            ]

            for model, model_name in vision_models:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_base64}"
                                    }
                                }
                            ]
                        }]
                    )

                    return jsonify({
                        'success': True,
                        'analysis': {
                            'id': f'analysis_{int(time.time())}',
                            'filename': file.filename,
                            'type': 'image',
                            'analysis': response.choices[0].message.content,
                            'model_used': model_name
                        }
                    })

                except Exception as e:
                    continue

            return jsonify({'success': False, 'error': 'Image analysis failed'}), 500
        else:
            supported_types = ['pdf', 'txt', 'md', 'rtf', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
            return jsonify({
                'success': False,
                'error': f'Unsupported file type: .{file_extension}. Supported types: {", ".join(supported_types)}'
            }), 400

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
