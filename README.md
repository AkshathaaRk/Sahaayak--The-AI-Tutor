# Sahaayak-The AI Tutor
Sahaayak is a comprehensive AI-powered educational platform designed specifically for Indian students, offering multilingual support, personalized learning, and advanced AI capabilities. Built with both Python backend and React frontend, it provides an intelligent tutoring system that understands and responds in multiple Indian languages.

## 🚀 *Key Features*

### 🤖 *AI-Powered Learning*
- *Multilingual AI Chat* - Conversations in Kannada, Telugu, Tamil, Hindi, and 15+ languages
- *Voice-to-Voice Chat* - Live conversations like Google Gemini with native language support
- *Image Analysis* - Upload images for AI-powered analysis and explanations
- *PDF Processing* - Upload PDFs for summaries, Q&A, and educational content extraction
- *Link Analysis* - Summarize YouTube videos, articles, and websites

### 📚 *Educational Tools*
- *Personalized Learning Paths* - Custom pathways based on grade and interests
- *AI Coaching/Tutoring* - Intelligent tutoring system for all subjects
- *Content Generation* - Generate lessons, stories, quizzes, and summaries
- *Grade-Specific Content* - Tailored for grades 1-12 across all subjects

### 🌍 *Multilingual Support*
- *Native Language Processing* - Full support for Indian languages
- *Cultural Context* - Content relevant to Indian education system
- *Regional Adaptation* - Examples and references from Indian culture

### 🔧 *Advanced Features*
- *Memory System* - MCP (Model Context Protocol) for persistent conversations
- *Fallback AI Models* - Multiple AI providers for reliability
- *Real-time Processing* - Instant responses and analysis
- *Cross-platform* - Works on web, mobile, and desktop

## 🏗 *Architecture*

### *Backend (Python Flask)*
- *Primary Server* (app.py) - Main AI processing and API endpoints
- *MCP Server* (mcp_server.py) - Context management and memory system
- *Security Layer* (mcp_security.py) - Authentication and encryption
- *Monitoring* (mcp_monitoring.py) - Performance tracking and analytics

### *Frontend (React + TypeScript)*
- *Modern UI* - Built with React 18, TypeScript, and Tailwind CSS
- *Responsive Design* - Works on all devices
- *Real-time Updates* - Live chat and voice features
- *Progressive Web App* - Installable on mobile devices

## 🛠 *Installation & Setup*

### *Prerequisites*
- Python 3.8+
- Node.js 16+
- Redis (for MCP server)
- Git

### *Backend Setup*

1. *Clone the repository*
bash
git clone https://github.com/yourusername/sahaayak.git
cd sahaayak


2. *Create virtual environment*
bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate


3. *Install Python dependencies*
bash
pip install -r requirements.txt


4. *Set up environment variables*
bash
# Create .env file
cp .env.example .env

# Edit .env with your API keys
# Required: OpenRouter API key
# Optional: A4F API key for fallback models


5. *Start Redis server*
bash
# On Windows (using WSL or Redis for Windows)
redis-server

# On macOS
brew install redis
redis-server

# On Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis


6. *Start the servers*
bash
# Terminal 1: Start MCP server
python mcp_server.py

# Terminal 2: Start main Flask app
python app.py


### *Frontend Setup*

1. *Navigate to frontend directory*
bash
cd Sahayak


2. *Install Node.js dependencies*
bash
npm install


3. *Start development server*
bash
npm run dev


4. *Build for production*
bash
npm run build


## 🔑 *API Keys Setup*

### *Required API Keys*
1. *OpenRouter API* (Free tier available)
   - Get your key from: https://openrouter.ai/keys
   - Add to .env: OPENROUTER_API_KEY=your_key_here

2. *A4F API* (Optional - for fallback models)
   - Get your key from: https://a4f.co
   - Add to .env: A4F_API_KEY=your_key_here

### *Environment Variables (.env)*
bash
# AI API Keys
OPENROUTER_API_KEY=your_openrouter_key_here
A4F_API_KEY=your_a4f_key_here

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Security
MCP_API_KEY=your_mcp_api_key_here
MCP_SECRET_KEY=your_secret_key_here

# Application Settings
FLASK_ENV=development
FLASK_DEBUG=True

## 📡 *API Endpoints*

### *Core AI Endpoints*
- POST /ask - General AI chat
- POST /ask-multilingual - Multilingual AI chat
- POST /analyze-image - Image analysis
- POST /voice-chat - Voice-to-voice conversation
- POST /pdf-summarize - PDF summarization
- POST /link-summarize - Link summarization

### *Educational Endpoints*
- POST /api/generate/content - Generate educational content
- POST /api/coach/query - AI tutoring
- POST /api/pathways/generate - Create learning paths
- POST /api/upload/analyze - Analyze uploaded files

### *MCP Endpoints*
- GET /health - Health check
- POST /context/update - Update conversation context
- GET /context/fetch - Fetch conversation context
- DELETE /context/clear - Clear conversation context

## 🧪 *Testing*

### *Backend Testing*
bash
# Test Flask endpoints
python -m pytest tests/

# Manual testing with curl
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is photosynthesis?"}'


### *Frontend Testing*
bash
# Run React tests
cd Sahayak
npm test

## 📊 *Monitoring & Analytics*

### *Performance Monitoring*
- *MCP Dashboard*: http://localhost:5001/admin/dashboard
- *Health Check*: http://localhost:5001/health
- *Metrics*: http://localhost:5001/metrics

### *Usage Analytics*
- Real-time request tracking
- Model performance metrics
- User engagement analytics
- Error rate monitoring

## 🔒 *Security Features*

- *API Key Authentication* - Secure API access
- *Rate Limiting* - Prevent abuse
- *Input Validation* - Sanitize user inputs
- *Encryption* - Secure sensitive data
- *CORS Protection* - Cross-origin security

## 🌍 *Supported Languages*

### *Indian Languages*
- *Kannada* (ಕನ್ನಡ)
- *Telugu* (తెలుగు)
- *Tamil* (தமిఴ்)
- *Hindi* (हिंदी)
- *Malayalam* (മലയാളം)
- *Bengali* (বাংলা)
- *Marathi* (मराठी)
- *Gujarati* (ગુજરાતી)
- *Punjabi* (ਪੰਜਾਬੀ)
- *Urdu* (اردو)

### *International Languages*
- English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic

### *Development Setup*
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 *License*

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 *Acknowledgments*

- *OpenRouter* for providing free AI models
- *A4F* for backup AI services
- *Indian Education System* for inspiration
- *Open Source Community* for amazing tools and libraries

## *Author*

- Akshatha RK(https://github.com/AkshathaaRk/)
- Abhishek KS(https://github.com/Abhishekmystic-KS/)

---

*Made with ❤ for Indian students and educators*
