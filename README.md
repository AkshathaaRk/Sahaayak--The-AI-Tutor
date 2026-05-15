# Sahaayak — AI-Powered Educational Assistant

Sahaayak (सहायक) is an AI-powered learning platform built for Indian students. It supports multiple Indian languages, voice chat, document analysis, and personalized tutoring — all in one place.

**Stack:** Python (Flask) + React (TypeScript)

---

## Features

- **Multilingual AI Chat** — Talk in Kannada, Telugu, Tamil, Hindi, Malayalam, and 10+ more languages
- **Voice Chat** — Real-time voice-to-voice conversations
- **Image & PDF Analysis** — Upload files and get AI-powered explanations
- **Link Summarizer** — Summarize YouTube videos, articles, and websites
- **AI Tutoring** — Personalized coaching and learning paths for grades 1–12
- **Memory System** — Persistent context across conversations via MCP

---

## Prerequisites

- Python 3.8+
- Node.js 16+
- Redis

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/sahaayak.git
cd sahaayak
```

### 2. Backend

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # Fill in your API keys
```

Start Redis, then:

```bash
python mcp_server.py            # Terminal 1
python app.py                   # Terminal 2
```

### 3. Frontend

```bash
cd Sahayak
npm install
npm run dev
```

---

## Environment Variables (`.env`)

```env
OPENROUTER_API_KEY=your_key     # Required — get at openrouter.ai/keys
A4F_API_KEY=your_key            # Optional fallback models

REDIS_HOST=localhost
REDIS_PORT=6379

MCP_API_KEY=your_key
MCP_SECRET_KEY=your_secret

FLASK_ENV=development
FLASK_DEBUG=True
```

---

## API Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ask` | General AI chat |
| POST | `/ask-multilingual` | Chat in Indian languages |
| POST | `/analyze-image` | Image analysis |
| POST | `/voice-chat` | Voice conversation |
| POST | `/pdf-summarize` | PDF summarization |
| POST | `/link-summarize` | Summarize links |
| POST | `/api/generate/content` | Generate lessons/quizzes |
| POST | `/api/coach/query` | AI tutoring |
| POST | `/api/pathways/generate` | Learning paths |

---

## License

MIT
