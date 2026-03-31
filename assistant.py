import wave
import struct
import threading
import time
import json
import os
import math
import logging
import numpy as np
import pyaudio
import sounddevice as sd
import torch
import emoji
from datetime import datetime, timedelta
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from silero_vad import VADIterator, load_silero_vad
from ddgs import DDGS
import anthropic
import pvporcupine
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# --- Suppress noisy logs ---
logging.getLogger('reachy_mini').setLevel(logging.ERROR)

# --- Config ---
load_dotenv()
MIC_DEVICE_INDEX = 1
AUDIO_FILE = 'recording.wav'
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SILENCE_THRESHOLD = 2
INACTIVITY_TIMEOUT = 10
MEMORY_DIR = 'memory'
FACTS_FILE = f'{MEMORY_DIR}/facts.json'
CONVERSATIONS_FILE = f'{MEMORY_DIR}/conversations.json'
CONVERSATION_TTL_DAYS = 7
WAKE_WORD_PATH = os.getenv('WAKE_WORD_PATH')       # Path to custom .ppn file (optional)
WAKE_WORD_NAME = os.getenv('WAKE_WORD_NAME', 'Porcupine')  # Display name for the wake word

# --- Memory setup ---
os.makedirs(MEMORY_DIR, exist_ok=True)

def load_facts():
    if os.path.exists(FACTS_FILE):
        with open(FACTS_FILE) as f:
            return json.load(f)
    return []

def save_facts(facts):
    with open(FACTS_FILE, 'w') as f:
        json.dump(facts, f, indent=2)

def load_conversations():
    if os.path.exists(CONVERSATIONS_FILE):
        with open(CONVERSATIONS_FILE) as f:
            return json.load(f)
    return []

def save_conversations(convs):
    with open(CONVERSATIONS_FILE, 'w') as f:
        json.dump(convs, f, indent=2)

def purge_old_conversations():
    convs = load_conversations()
    now = datetime.now()
    kept = [c for c in convs if now - datetime.fromisoformat(c['timestamp']) <= timedelta(days=CONVERSATION_TTL_DAYS)]
    save_conversations(kept)
    return kept

def serialize_messages(messages):
    serialized = []
    for msg in messages:
        if isinstance(msg['content'], str):
            serialized.append(msg)
        elif isinstance(msg['content'], list):
            safe_content = []
            for block in msg['content']:
                if hasattr(block, 'type'):
                    if block.type == 'text':
                        safe_content.append({'type': 'text', 'text': block.text})
                    elif block.type == 'tool_use':
                        safe_content.append({'type': 'tool_use', 'id': block.id, 'name': block.name, 'input': block.input})
                    else:
                        safe_content.append(block)
                else:
                    safe_content.append(block)
            serialized.append({'role': msg['role'], 'content': safe_content})
    return serialized

def save_conversation_summary(messages):
    if not messages:
        return
    convs = load_conversations()
    convs.append({
        'timestamp': datetime.now().isoformat(),
        'messages': serialize_messages(messages[-20:])
    })
    save_conversations(convs)
    purge_old_conversations()

def build_memory_context():
    parts = []
    facts = load_facts()
    if facts:
        parts.append("Facts you know about the user:\n" + "\n".join(f"- {f}" for f in facts))
    convs = purge_old_conversations()
    if convs:
        recent = convs[-3:]
        summary_lines = []
        for c in recent:
            ts = datetime.fromisoformat(c['timestamp']).strftime('%b %d')
            user_msgs = [m['content'] for m in c['messages'] if isinstance(m['content'], str) and m['role'] == 'user'][:2]
            if user_msgs:
                summary_lines.append(f"[{ts}] User asked about: {'; '.join(user_msgs)}")
        if summary_lines:
            parts.append("Recent conversation history:\n" + "\n".join(summary_lines))
    return "\n\n".join(parts)

# --- Load models ---
print("Loading Whisper...")
whisper = WhisperModel('tiny', device='cpu', compute_type='int8')

print("Loading Kokoro...")
kokoro = Kokoro('kokoro-v1.0.onnx', 'voices-v1.0.bin')

print("Loading VAD...")
vad_model = load_silero_vad()
vad_iterator = VADIterator(vad_model, sampling_rate=SAMPLE_RATE)

print("Connecting to Claude...")
claude = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

print("Loading wake word detector...")
if WAKE_WORD_PATH:
    porcupine = pvporcupine.create(
        access_key=os.getenv('PICOVOICE_API_KEY'),
        keyword_paths=[WAKE_WORD_PATH]
    )
else:
    porcupine = pvporcupine.create(
        access_key=os.getenv('PICOVOICE_API_KEY'),
        keywords=['porcupine']
    )

print("Connecting to Reachy Mini...")
reachy = ReachyMini()

conversation = []
_stop_animation = threading.Event()

# --- Safe movement ---
def safe_goto(antennas=None, head=None, duration=0.5):
    try:
        kwargs = {'duration': duration}
        if antennas is not None:
            kwargs['antennas'] = antennas
        if head is not None:
            kwargs['head'] = head
        reachy.goto_target(**kwargs)
    except (Exception, KeyboardInterrupt):
        pass

# --- Animations ---
def animate_listening():
    _stop_animation.clear()
    def loop():
        safe_goto(head=create_head_pose(pitch=5, degrees=True), duration=0.8)
        while not _stop_animation.is_set():
            safe_goto(antennas=[0.2, 0.2], duration=0.8)
            time.sleep(0.9)
            if _stop_animation.is_set():
                break
            safe_goto(antennas=[0.0, 0.0], duration=0.8)
            time.sleep(0.9)
    threading.Thread(target=loop, daemon=True).start()

def animate_thinking():
    _stop_animation.clear()
    def loop():
        while not _stop_animation.is_set():
            safe_goto(antennas=[0.2, -0.2], head=create_head_pose(roll=8, degrees=True), duration=0.7)
            time.sleep(0.8)
            if _stop_animation.is_set():
                break
            safe_goto(antennas=[-0.2, 0.2], head=create_head_pose(roll=-8, degrees=True), duration=0.7)
            time.sleep(0.8)
    threading.Thread(target=loop, daemon=True).start()

def animate_speaking():
    _stop_animation.clear()
    def loop():
        while not _stop_animation.is_set():
            safe_goto(antennas=[0.2, 0.2], head=create_head_pose(z=3, mm=True), duration=0.5)
            time.sleep(0.6)
            if _stop_animation.is_set():
                break
            safe_goto(antennas=[0.0, 0.0], head=create_head_pose(), duration=0.5)
            time.sleep(0.6)
    threading.Thread(target=loop, daemon=True).start()

def stop_animation():
    _stop_animation.set()
    time.sleep(0.3)
    safe_goto(antennas=[0, 0], head=create_head_pose(), duration=0.3)

# --- SDK reference ---
REACHY_SDK_REFERENCE = """
Reachy Mini SDK - goto_target reference:

create_head_pose(yaw=0, pitch=0, roll=0, z=0, degrees=True, mm=True)
- yaw:   left/right turn. Positive=right, negative=left.
- pitch: up/down tilt. Positive=up, negative=down.
- roll:  side tilt. Positive=tilt right.
- z:     height offset in mm.

antennas=[left, right] in radians.

SAFE LIMITS (stay within these for natural movement):
- Head yaw:   ±30°
- Head pitch: ±20°
- Head roll:  ±20°
- Antennas:   ±0.25 radians
- Duration:   0.5–2.0 seconds recommended
"""

# --- Tools ---
tools = [
    {
        "name": "web_search",
        "description": "Search the web for current information like weather, news, or any real-time data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "remember_fact",
        "description": "Save an important fact about the user for future conversations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "fact": {"type": "string"}
            },
            "required": ["fact"]
        }
    },
    {
        "name": "move_reachy",
        "description": f"Move Reachy Mini's physical body. Use when asked to move, turn, nod, look, or express something physically. Also use proactively to express emotions (nod when agreeing, tilt when curious).\n\n{REACHY_SDK_REFERENCE}",
        "input_schema": {
            "type": "object",
            "properties": {
                "head_yaw":      {"type": "number", "description": "Head yaw in degrees. Positive=right, negative=left."},
                "head_pitch":    {"type": "number", "description": "Head pitch in degrees. Positive=up, negative=down."},
                "head_roll":     {"type": "number", "description": "Head roll in degrees."},
                "head_z_mm":     {"type": "number", "description": "Head height offset in mm."},
                "antenna_left":  {"type": "number", "description": "Left antenna in radians. Safe: ±0.25."},
                "antenna_right": {"type": "number", "description": "Right antenna in radians. Safe: ±0.25."},
                "duration":      {"type": "number", "description": "Duration in seconds. Default 0.8."},
                "method":        {"type": "string", "description": "minjerk | linear | ease | cartoon"}
            }
        }
    }
]

def web_search(query):
    print(f"  [Searching: {query}]")
    results = DDGS().text(query, max_results=3)
    if not results:
        return "No results found."
    return "\n\n".join([f"{r['title']}\n{r['body']}" for r in results])

def remember_fact(fact):
    facts = load_facts()
    if fact not in facts:
        facts.append(fact)
        save_facts(facts)
        print(f"  [Remembered: {fact}]")
    return f"Remembered: {fact}"

def move_reachy(params):
    head = None
    antennas = None
    duration = params.get('duration', 0.8)

    has_head = any(k in params for k in ['head_yaw', 'head_pitch', 'head_roll', 'head_z_mm'])
    if has_head:
        head = create_head_pose(
            yaw=params.get('head_yaw', 0),
            pitch=params.get('head_pitch', 0),
            roll=params.get('head_roll', 0),
            z=params.get('head_z_mm', 0),
            degrees=True,
            mm=True
        )

    if 'antenna_left' in params or 'antenna_right' in params:
        antennas = [params.get('antenna_left', 0), params.get('antenna_right', 0)]

    print(f"  [Moving Reachy: {params}]")
    safe_goto(head=head, antennas=antennas, duration=duration)
    return f"Moved: {params}"

def wait_for_wake_word():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1,
                    rate=porcupine.sample_rate,
                    input=True, input_device_index=MIC_DEVICE_INDEX,
                    frames_per_buffer=porcupine.frame_length)
    print(f"\nWaiting for wake word... (say '{WAKE_WORD_NAME}')")
    while True:
        pcm = stream.read(porcupine.frame_length)
        pcm = struct.unpack_from('h' * porcupine.frame_length, pcm)
        if porcupine.process(pcm) >= 0:
            print("Wake word detected!")
            stream.close()
            p.terminate()
            return

def transcribe():
    segments, _ = whisper.transcribe(AUDIO_FILE)
    return ' '.join([s.text for s in segments]).strip()

def ask_claude(user_text):
    conversation.append({'role': 'user', 'content': user_text})
    memory_context = build_memory_context()

    system = f"""You are Reechy, a friendly desktop robot assistant with a physical body — a small expressive robot with a head and antennas. Keep responses short and conversational — 1 to 3 sentences max.
- Use web_search whenever you need current information.
- Use remember_fact whenever the user tells you something personal or important worth remembering.
- Use move_reachy to physically move when asked, or proactively to express emotions (nod when agreeing, tilt head when curious, raise antennas when excited).
- Always extract specific facts and numbers from search results rather than telling the user to check elsewhere.

{REACHY_SDK_REFERENCE}"""

    if memory_context:
        system += f"\n\n{memory_context}"

    while True:
        response = claude.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=300,
            system=system,
            tools=tools,
            messages=conversation
        )

        if response.stop_reason == 'tool_use':
            tool_use = next(b for b in response.content if b.type == 'tool_use')
            if tool_use.name == 'web_search':
                tool_result = web_search(tool_use.input['query'])
            elif tool_use.name == 'remember_fact':
                tool_result = remember_fact(tool_use.input['fact'])
            elif tool_use.name == 'move_reachy':
                tool_result = move_reachy(tool_use.input)
            else:
                tool_result = "Unknown tool."

            conversation.append({'role': 'assistant', 'content': response.content})
            conversation.append({
                'role': 'user',
                'content': [{'type': 'tool_result', 'tool_use_id': tool_use.id, 'content': tool_result}]
            })
            continue

        reply = next(b.text for b in response.content if hasattr(b, 'text'))
        conversation.append({'role': 'assistant', 'content': reply})
        return reply

def speak(text):
    clean_text = emoji.replace_emoji(text, replace='')
    samples, sample_rate = kokoro.create(clean_text, voice='af_heart', speed=1.0, lang='en-us')
    animate_speaking()
    sd.play(samples, sample_rate)
    sd.wait()
    stop_animation()

# --- Main loop ---
print(f"\nReechy is ready! Say '{WAKE_WORD_NAME}' to wake her up. Ctrl+C to quit.\n")
try:
    while True:
        wait_for_wake_word()
        stop_animation()
        speak("Yes?")

        last_interaction = time.time()
        while True:
            animate_listening()

            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                            input=True, input_device_index=MIC_DEVICE_INDEX,
                            frames_per_buffer=CHUNK_SIZE)
            frames = []
            silence_count = 0
            speaking = False
            timed_out = False

            while True:
                if not speaking and (time.time() - last_interaction) > INACTIVITY_TIMEOUT:
                    timed_out = True
                    break

                chunk = stream.read(CHUNK_SIZE)
                frames.append(chunk)
                audio_chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                tensor = torch.from_numpy(audio_chunk)
                speech_dict = vad_iterator(tensor, return_seconds=False)

                if speech_dict:
                    if 'start' in speech_dict:
                        speaking = True
                        last_interaction = time.time()
                        silence_count = 0
                        print("Speech detected...")
                    if 'end' in speech_dict:
                        silence_count += 1

                if speaking and silence_count >= SILENCE_THRESHOLD:
                    print("Speech ended, processing...")
                    break

                if len(frames) > (SAMPLE_RATE / CHUNK_SIZE) * 15:
                    break

            stream.stop_stream()
            stream.close()
            p.terminate()
            vad_iterator.reset_states()
            stop_animation()

            if timed_out or not speaking:
                print("No activity detected, going back to sleep.")
                save_conversation_summary(conversation)
                conversation.clear()
                break

            with wave.open(AUDIO_FILE, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(frames))

            print("Transcribing...")
            user_text = transcribe()
            if not user_text:
                print("Didn't catch that.")
                continue

            print(f"You: {user_text}")

            # Sleep words — exit assistant
            sleep_phrases = ['goodbye', 'good bye', 'bye', 'go to sleep', 'sleep', 'stop listening']
            if any(phrase in user_text.lower() for phrase in sleep_phrases):
                speak("Goodbye! Talk to you later.")
                save_conversation_summary(conversation)
                conversation.clear()
                raise KeyboardInterrupt

            animate_thinking()
            reply = ask_claude(user_text)
            stop_animation()
            print(f"Reechy: {reply}")
            speak(reply)
            last_interaction = time.time()

except KeyboardInterrupt:
    print("\nGoodbye!")
    save_conversation_summary(conversation)
    stop_animation()
    porcupine.delete()