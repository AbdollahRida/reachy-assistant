# Reachy Assistant

A fully local, voice-powered AI desktop assistant built on top of the [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) robot by Pollen Robotics. Say a wake word, speak naturally, and Reachy answers your questions, searches the web, remembers things about you, and animates expressively — all powered by Claude.

![Python](https://img.shields.io/badge/python-3.12-blue) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Demo

> Wake word → voice input → Claude thinks → Reachy speaks and animates

- Ask about the weather, news, or anything current → Reachy searches DuckDuckGo and answers
- Tell Reachy something personal → it remembers across sessions
- Ask Reachy to move its head or antennas → it physically responds
- Say "goodbye" → clean exit

---

## Stack

| Layer | Tool | Notes |
|---|---|---|
| Wake word | [Porcupine](https://picovoice.ai) | Always-on, runs locally |
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Local, no API needed |
| Voice activity detection | [silero-vad](https://github.com/snakers4/silero-vad) | Dynamic recording cutoff |
| Brain | [Claude API](https://console.anthropic.com) (Haiku) | Tool use for search, memory, movement |
| Web search | [ddgs](https://github.com/deedy5/ddgs) | DuckDuckGo, no API key needed |
| Text-to-speech | [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | Local, natural-sounding voice |
| Robot control | [reachy-mini SDK](https://github.com/pollen-robotics/reachy_mini) | Antennas, head movement, animations |

Everything runs locally except the Claude API call.

---

## Features

- **Wake word detection** — say "Porcupine" to activate (powered by Picovoice)
- **Dynamic voice recording** — stops automatically when you go silent
- **Continuous conversation** — no need to repeat the wake word mid-conversation
- **Inactivity timeout** — goes back to sleep after 10 seconds of silence
- **Web search** — Claude autonomously searches when it needs current information
- **Persistent memory** — remembers facts about you across sessions, purges conversations older than 7 days
- **Expressive animations** — listening, thinking, and speaking animations with head and antenna movement
- **Autonomous movement** — Claude controls Reachy's physical body using the full SDK
- **Sleep word** — say "goodbye" or "bye" to exit cleanly

---

## Requirements

- Python 3.12+
- [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) robot (Wireless or Lite) with daemon running
- [Reachy Mini Desktop App](https://github.com/pollen-robotics/reachy-mini-desktop-app) (recommended for daemon management)
- Anthropic API key — [console.anthropic.com](https://console.anthropic.com)
- Picovoice API key — [console.picovoice.ai](https://console.picovoice.ai) (free tier)

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/reachy-assistant
cd reachy-assistant
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install system dependencies (macOS)

```bash
brew install portaudio
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Download Kokoro voice models

```bash
curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### 6. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
ANTHROPIC_API_KEY=your-anthropic-key
PICOVOICE_API_KEY=your-picovoice-key
```

### 7. Start the Reachy Mini daemon

Open the Reachy Mini Desktop App and confirm the robot shows **Ready**.

### 8. Run

```bash
python assistant.py
```

---

## Usage

| Action | How |
|---|---|
| Wake Reachy | Say **"Porcupine"** |
| Ask anything | Speak naturally after Reachy says "Yes?" |
| Pause mid-sentence | Reachy waits up to ~2 seconds of silence |
| End conversation | Say **"goodbye"** or **"bye"** |
| Inactivity sleep | Reachy sleeps automatically after 10 seconds |

---

## Project Structure

```
reachy-assistant/
├── assistant.py          # Main application
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── .gitignore
├── memory/               # Created at runtime (gitignored)
│   ├── facts.json        # Persistent facts about the user
│   └── conversations.json # Recent conversation history
├── kokoro-v1.0.onnx      # Kokoro voice model (gitignored, download separately)
└── voices-v1.0.bin       # Kokoro voices (gitignored, download separately)
```

---

## Roadmap (v2)

- [ ] Camera vision — pipe Reachy's camera feed into Claude for visual awareness
- [ ] Persistent head poses — hold a position between moves
- [X] Custom wake word — replace "Porcupine" with "Hey Reachy"
- [ ] Local fallback — use Ollama when Claude API is unreachable
- [ ] Emotion detection — respond to tone of voice with matching animations
- [ ] Multi-step movement sequences — choreographed gestures

---

## License

MIT
