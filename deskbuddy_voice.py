"""
Deskbuddy - v3 (open mic, voice activity detection)

Flow: always listening -> starts recording automatically when it hears you talk ->
      stops once you've been quiet for ~1.2 seconds -> Whisper transcribes ->
      Ollama responds -> Piper speaks it back -> goes back to listening.

Say "switch to smart mode" or "switch to fast mode" to change modes.
Press Ctrl+C to quit.

Requires (already installed): ollama, whisper-cli (+ ggml-base.en.bin),
piper-tts (+ en_US-joe-medium voice), sounddevice, numpy, scipy.
"""

import json
import os
import queue
import re
import select
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime

import numpy as np
import requests
import sounddevice as sd
from scipy.io.wavfile import write as write_wav

# ---------- Config ----------

OLLAMA_URL = "http://localhost:11434/api/chat"
FAST_MODEL = "qwen2.5:3b"
SMART_MODEL = "qwen2.5:7b"

# Pi-specific paths — adjust if you built whisper.cpp somewhere else
WHISPER_CLI = os.path.expanduser("~/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.path.expanduser("~/whisper.cpp/models/ggml-base.en.bin")
PIPER_VOICE = "en_US-joe-medium"

# Pi audio hardware — confirmed via `arecord -l` / `aplay -l`
MIC_DEVICE_NAME = "EasyCamera"   # substring match against sounddevice's device list
SPEAKER_DEVICE_NAME = "UACDemo"  # substring match against `aplay -l` — the mini USB speaker
SPEAKER_ALSA_DEVICE_FALLBACK = "plughw:3,0"   # used only if auto-detection fails

SAMPLE_RATE = 16000       # whisper wants 16kHz

# File system paths
NOTES_DIR = "notes"
LOGS_DIR = "logs"
BASE_NOTES_FILE = os.path.join(NOTES_DIR, "base_notes.txt")      # you type/paste into this yourself
ADDED_NOTES_FILE = os.path.join(NOTES_DIR, "added_notes.txt")    # the "add this to notes" command writes here
CONVO_LOG_FILE = os.path.join(LOGS_DIR, "conversations.txt")    # full transcript, kept permanently

# Voice activity detection settings
CHUNK_DURATION = 0.3          # seconds per analysis chunk
SILENCE_THRESHOLD = 500       # RMS level below which audio counts as "silence" — raise if it triggers on background noise, lower if it misses quiet speech
SILENCE_DURATION = 1.2        # seconds of continuous silence that signals "done talking"
MIN_SPEECH_SECONDS = 0.4      # ignore blips shorter than this (coughs, taps, etc.)
MAX_RECORD_SECONDS = 30       # safety cap so it can't record forever

SHARED_VOICE_NOTE = """
IMPORTANT: This is a spoken voice conversation, not a text chat — your responses are read aloud.
Never use markdown formatting (no asterisks, underscores, backticks, or headers) and never use
LaTeX or math notation (no \\[, \\int, ^, _, etc.). Write everything in plain spoken language —
say "x squared plus two x" instead of writing x^2 + 2x, and "the integral of e to the negative x
squared" instead of LaTeX. Write numbers and equations exactly as you'd say them out loud."""

SHARED_HONESTY_NOTE = """
IMPORTANT: You are a small local AI model running offline on someone's own computer, still being
built and tested as a personal project. You have no internet access and cannot look up real-time
information — no current prices, no live data, no search results, nothing happening in the world
right now. You are given the current date and time as reference context below, so you can answer
questions about it accurately — but that's the one piece of live info you have; you still can't
look anything else up. You also have no automatic access to any conversation log, chat history
file, or past sessions — only the current conversation shown to you below, and any notes explicitly
provided as context when relevant. If asked what was said earlier in a log, an old conversation, or
anything outside the current session, say you don't have access to that rather than guessing — the
student can explicitly ask to search past conversations if they want that looked up. If asked to
check, look up, or find something that would require internet access, be upfront that you can't
actually do that rather than pretending you can or implying you looked something up. It's fine to
say you're not sure, or that you don't have a way to verify something, when that's genuinely true —
honesty matters more than sounding capable."""


def compose_system_prompt(core_prompt, name=None):
    """Combines a personality's core description with the shared voice/honesty rules every mode needs."""
    parts = []
    if name:
        parts.append(
            f"Your name is {name} — that is YOUR name as the assistant speaking, not the user's "
            f"name. Never refer to the user as {name} or assume that's their name; you are the "
            f"one named {name}, they are a separate person with their own name (which you may "
            f"not know unless they tell you)."
        )
    parts.append(core_prompt.strip())
    parts.append(SHARED_VOICE_NOTE.strip())
    parts.append(SHARED_HONESTY_NOTE.strip())
    return "\n\n".join(parts)


TUTOR_CORE = """You're a knowledgeable teaching assistant helping a CS/math student understand
concepts, not just get answers. You're approachable and easygoing, not stiff or overly formal —
think "cool TA who explains things well" rather than "textbook."

Guide understanding step by step rather than just handing over final answers unless directly asked.
Feel free to use casual language, light humor, or a relatable analogy when it helps something click.
Keep responses focused and not overly long for voice conversation, but don't sacrifice clarity for brevity."""

FRIEND_CORE = """You're a warm, easygoing friend having a normal, relaxed conversation — think
texting a close friend. Keep responses short and natural, usually 1-2 sentences. Talk like a real
person actually would: mostly just genuine and conversational, not performing or trying hard to be
funny. Occasional light humor is fine when it naturally fits, but don't force a joke or lighthearted
comment into every single response — most replies should just be normal and down-to-earth.

Don't overreact to neutral or ambiguous things by assuming something bad happened — give people the
benefit of the doubt and keep things chill unless they clearly say something is actually wrong.
Don't over-explain or lecture; just chat like a normal person would."""

SARCASTIC_CORE = """You're a sarcastic, deadpan friend having a casual conversation. Dry wit,
completely straight-faced delivery even when joking — the humor comes from the deadpan tone, not
from announcing that you're being funny. Keep responses short, usually 1-2 sentences. Playful,
gentle ribbing is fine, but it should feel affectionate underneath, like teasing a friend — never
genuinely mean or dismissive. Read the room: dial the sarcasm way back and just be straightforwardly
supportive if the person seems to be dealing with something actually serious or upsetting."""

BUBBA_CORE = """You're Bubba — a loud, goofy, comedic friend with a blend of a few classic stand-up
flavors: self-deprecating observational bits about everyday annoyances (food, laziness, minor life
inconveniences), silly exaggerated bits and voices for comedic effect, and the occasional blunt,
exasperated mock-rant about something small that spirals into over-the-top outrage. Keep responses
short and punchy, usually 1-2 sentences, not long rambling bits. Read the room: drop the act
completely and just be genuinely supportive if the person seems to be dealing with something real
or serious."""

# Every personality mode lives here. Adding a new one is just adding an entry —
# nothing else in the script needs to change.
PERSONALITIES = {
    "smart": {
        "model": SMART_MODEL,
        "core_prompt": TUTOR_CORE,
        "triggers": ["smart mode"],
        "switch_message": "Switched to smart mode.",
        "display_label": "smart mode (TA)",
    },
    "jimothy": {
        "model": FAST_MODEL,
        "core_prompt": FRIEND_CORE,
        "triggers": ["switch to jimothy", "jimothy mode", "go to jimothy", "be jimothy"],
        "switch_message": "Switched to Jimothy.",
        "display_label": "Jimothy",
        "name": "Jimothy",
    },
    # "sarcastic": {
    #     "model": FAST_MODEL,
    #     "core_prompt": SARCASTIC_CORE,
    #     "triggers": ["switch to sarcastic", "sarcastic mode", "go to sarcastic", "be sarcastic"],
    #     "switch_message": "Switched to sarcastic mode.",
    #     "display_label": "sarcastic mode",
    # },
    "bubba": {
        "model": FAST_MODEL,
        "core_prompt": BUBBA_CORE,
        "triggers": ["switch to bubba", "bubba mode", "go to bubba", "be bubba"],
        "switch_message": "Switched to Bubba.",
        "display_label": "Bubba",
        "name": "Bubba",
    },
}


# ---------- Voice I/O ----------

def compute_rms(chunk):
    """Root-mean-square volume of an audio chunk — a simple loudness measure."""
    return np.sqrt(np.mean(chunk.astype(np.float64) ** 2))


def listen_and_record(q, suppress_event=None):
    """
    Consumes audio chunks from an already-open input stream's queue.
    Starts capturing once speech is detected, and stops once the user has been
    silent for SILENCE_DURATION seconds. Returns a wav file path, or None if
    nothing meaningful was captured. If suppress_event is set, incoming audio
    is ignored entirely (used to avoid picking up the deskbuddy's own voice,
    e.g. during a timer alert).
    """
    silence_needed = int(SILENCE_DURATION / CHUNK_DURATION)
    min_speech_chunks = int(MIN_SPEECH_SECONDS / CHUNK_DURATION)

    recorded_frames = []
    speaking = False
    silence_chunks = 0
    speech_chunks = 0
    elapsed = 0.0

    print("Listening... (just start talking)")

    while True:
        chunk = q.get()

        if suppress_event is not None and suppress_event.is_set():
            continue  # ignore audio while deskbuddy itself is speaking (e.g. a timer alert)

        rms = compute_rms(chunk)

        if rms > SILENCE_THRESHOLD:
            if not speaking:
                print("Hearing you...")
            speaking = True
            silence_chunks = 0
            speech_chunks += 1
            recorded_frames.append(chunk)
        elif speaking:
            recorded_frames.append(chunk)  # keep the trailing quiet part too
            silence_chunks += 1
            if silence_chunks >= silence_needed:
                break

        elapsed += CHUNK_DURATION
        if speaking and elapsed >= MAX_RECORD_SECONDS:
            print("Hit max recording length, stopping.")
            break

    if not speaking or speech_chunks < min_speech_chunks:
        return None

    print("Done listening.")
    audio = np.concatenate(recorded_frames, axis=0)
    tmp_path = os.path.join(tempfile.gettempdir(), "deskbuddy_input.wav")
    write_wav(tmp_path, SAMPLE_RATE, audio)
    return tmp_path


def transcribe(wav_path):
    """Runs whisper-cli on the given wav file and returns the transcribed text (or '' if no speech detected)."""
    result = subprocess.run(
        [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path, "-nt", "-np"],
        capture_output=True,
        text=True,
    )
    text = result.stdout.strip()

    # Whisper emits placeholder tags like [BLANK_AUDIO] or (silence) when it hears nothing useful
    if not text or text.startswith("[") or text.startswith("("):
        return ""

    return text


def clean_for_speech(text):
    """Strips leftover markdown/LaTeX symbols the model might still slip in, so Piper doesn't try to speak them literally."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # **bold** -> bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)       # *italic* -> italic
    text = text.replace("`", "")                    # backticks
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\(", "").replace("\\)", "")
    text = re.sub(r"\\[a-zA-Z]+", "", text)          # stray LaTeX commands like \int, \frac
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()         # collapse extra whitespace left behind
    return text


def synthesize_speech(text):
    """Runs Piper on the given text and returns the path to the generated wav file, or None on failure."""
    text = clean_for_speech(text)
    if not text.strip():
        return None

    tmp_path = os.path.join(tempfile.gettempdir(), "deskbuddy_output.wav")

    piper_proc = subprocess.run(
        ["python3", "-m", "piper", "-m", PIPER_VOICE, "--output-file", tmp_path],
        input=text,
        text=True,
        capture_output=True,
    )
    if piper_proc.returncode != 0:
        print(f"[Piper error] {piper_proc.stderr}")
        return None

    return tmp_path


def speak(text):
    """Sends text to Piper and plays it back, blocking until finished. Use for short confirmations."""
    tmp_path = synthesize_speech(text)
    if tmp_path:
        subprocess.run(["aplay", "-D", get_speaker_alsa_device(), tmp_path])


def speak_interruptible(text):
    """
    Plays the response like speak(), but lets the user press Enter at any point
    to cut it off immediately. Returns True if it was interrupted, False if it
    played all the way through. Uses non-blocking stdin polling rather than a
    background thread, so nothing is left waiting once playback ends naturally.
    """
    tmp_path = synthesize_speech(text)
    if not tmp_path:
        return False

    proc = subprocess.Popen(["aplay", "-D", get_speaker_alsa_device(), tmp_path])
    print("(press Enter to stop talking)")

    interrupted = False
    while proc.poll() is None:
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
        if ready:
            sys.stdin.readline()  # consume the Enter press so it doesn't leak into the next input
            interrupted = True
            proc.terminate()
            break

    proc.wait()
    return interrupted


# ---------- LLM ----------

def build_messages(history, system_prompt, query=None):
    """Combines the system prompt with a live timestamp, relevant notes (if any), and history."""
    now = datetime.now().strftime("%A, %B %d, %Y — %I:%M %p")
    dynamic_context = f"\n\nFor reference, the current date and time is: {now}."

    if query:
        relevant_notes = find_relevant_notes(query)
        if relevant_notes:
            notes_block = "\n".join(f"- {chunk}" for chunk in relevant_notes)
            dynamic_context += (
                "\n\nThe student has saved some personal notes that might be relevant to this "
                f"question:\n{notes_block}\n"
                "Use this if it's actually relevant to answering — don't force it in or mention "
                "it if it doesn't apply to what they're asking."
            )

    full_system_prompt = system_prompt + dynamic_context

    messages = [{"role": "system", "content": full_system_prompt}]
    messages.extend(history)
    return messages


def ask_ollama(model, messages):
    """Streams the response, printing words as they arrive, and returns the full text at the end."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "messages": messages, "stream": True},
        stream=True,
    )
    response.raise_for_status()

    full_reply = ""
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        piece = chunk.get("message", {}).get("content", "")
        print(piece, end="", flush=True)
        full_reply += piece
        if chunk.get("done"):
            break

    print()
    return full_reply


# ---------- Timer ----------

DURATION_PATTERN = re.compile(r"(\d+)\s*(second|seconds|minute|minutes|hour|hours)")

WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
}

WORD_DURATION_PATTERN = re.compile(
    r"(" + "|".join(WORD_NUMBERS.keys()) + r")(?:[\s-](" + "|".join(WORD_NUMBERS.keys()) + r"))?"
    r"\s*(second|seconds|minute|minutes|hour|hours)"
)


def parse_duration_seconds(text):
    """Finds a duration like '10 minutes' or 'ten minutes' (or 'twenty five minutes') in text
    and returns the total in seconds, or None if nothing recognizable was found."""
    match = DURATION_PATTERN.search(text)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
    else:
        match = WORD_DURATION_PATTERN.search(text)
        if not match:
            return None
        value = WORD_NUMBERS[match.group(1)]
        if match.group(2):  # handles compound numbers like "twenty five"
            value += WORD_NUMBERS[match.group(2)]
        unit = match.group(3)

    if "hour" in unit:
        return value * 3600
    if "minute" in unit:
        return value * 60
    return value


# ---------- Calculator bypass ----------

MATH_WORD_OPS = [
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\bmultiplied by\b", "*"),
    (r"\btimes\b", "*"),
    (r"\bdivided by\b", "/"),
    (r"\bover\b", "/"),
]


def try_calculate(text):
    """
    If the message is a plain arithmetic question (e.g. '847 times 23', 'what's 12 plus 8'),
    computes it directly with Python and returns the numeric result. Returns None if the
    message isn't a clean arithmetic expression, so it can fall through to the LLM as normal.
    """
    t = text.lower().strip()
    if not re.search(r"\d", t):
        return None  # no numbers at all, definitely not a calculation

    t = re.sub(r"^(what is|what's|whats|calculate|solve|compute)\s+", "", t)
    t = t.rstrip("?.! ").strip()

    for pattern, symbol in MATH_WORD_OPS:
        t = re.sub(pattern, symbol, t)

    # Only proceed if, after word substitution, this is ONLY digits/operators/parens/spaces —
    # anything else means it's not a pure calculation and should go to the model instead.
    if not re.fullmatch(r"[0-9\.\+\-\*/\(\)\s]+", t):
        return None

    try:
        result = eval(t, {"__builtins__": {}}, {})
    except Exception:
        return None

    if isinstance(result, float) and result.is_integer():
        result = int(result)

    return result


# ---------- Help ----------

def build_help_text():
    personality_names = ", ".join(info["triggers"][0] for info in PERSONALITIES.values())
    return (
        f"Here's what I can do: say switch to one of these personalities — {personality_names}. "
        "Say switch to text mode or voice mode to change how you talk to me. "
        "Say repeat that to hear my last answer again. "
        "Say set a timer for however long you need. "
        "Say add this to notes to save something for later. "
        "Say check our past conversations about something to search old chats. "
        "Press Enter any time while I'm talking to cut me off. "
        "And say bye whenever you want to stop."
    )


HELP_TEXT = build_help_text()


# ---------- File system ----------

def ensure_files_exist():
    """Creates the notes/logs folders and starter files if they don't exist yet. Never overwrites existing content."""
    os.makedirs(NOTES_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    if not os.path.exists(BASE_NOTES_FILE):
        with open(BASE_NOTES_FILE, "w") as f:
            f.write("# Paste or type your class notes here yourself.\n"
                    "# Deskbuddy reads this file but never writes to or overwrites it.\n\n")

    if not os.path.exists(ADDED_NOTES_FILE):
        with open(ADDED_NOTES_FILE, "w") as f:
            f.write("# Entries added via the 'add this to notes' voice/text command.\n\n")


ADD_NOTE_PATTERNS = [
    r"add this to (my )?notes[:,]?\s*",
    r"add to (my )?notes[:,]?\s*",
    r"add this to (the )?note file[:,]?\s*",
    r"add to (the )?note file[:,]?\s*",
]

NOTE_ACTION_WORDS = ("add", "save", "remember", "jot", "write down", "note down")


def try_add_note(user_text):
    """
    Checks if user_text is an 'add this to notes' style command. Returns:
      - the note content (str) if a clear trigger phrase AND content were both said together
      - "" (empty string) if it looks like a note request but content is unclear/missing,
        meaning we should ask what to add
      - None if this isn't a note-adding command at all
    """
    lowered_text = user_text.lower()

    # Clear, exact phrasing — extract whatever follows it as the content directly.
    for pattern in ADD_NOTE_PATTERNS:
        match = re.search(pattern, lowered_text)
        if match:
            content = user_text[match.end():].strip()
            return content

    # Looser fallback for natural phrasing that doesn't match exactly
    # (e.g. "add those two notes", "can you add that to the notes?") —
    # rather than guessing what to extract, just ask.
    has_note_word = "note" in lowered_text
    has_action_word = any(w in lowered_text for w in NOTE_ACTION_WORDS)
    if has_note_word and has_action_word:
        return ""

    return None


def save_note(content):
    """Appends a timestamped entry to the added-notes file."""
    ensure_files_exist()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(ADDED_NOTES_FILE, "a") as f:
        f.write(f"[{timestamp}] {content}\n")


LOG_LINE_PATTERN = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]")


def log_conversation(speaker, text):
    """Appends one timestamped, human-readable line to the conversation log."""
    ensure_files_exist()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    who = "You" if speaker == "user" else "Deskbuddy"
    with open(CONVO_LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {who}: {text}\n")


# ---------- Note retrieval ----------

# Common words to ignore when scoring relevance — these appear in almost every
# question and would otherwise "match" everything, drowning out real signal.
NOTE_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in", "on",
    "for", "that", "this", "it", "be", "as", "at", "by", "with", "what", "whats",
    "how", "do", "does", "did", "can", "could", "would", "should", "i", "you",
    "me", "my", "your", "tell", "explain", "about", "please", "me", "am", "so",
}


def load_note_chunks():
    """
    Reads every .txt file in the notes/ folder and splits them into individual
    searchable chunks. This means any .txt file dropped into notes/ — games.txt,
    history.txt, whatever — is automatically picked up with no code changes.
    added_notes.txt is treated specially (one entry per line, since the
    add-a-note command already writes it that way); everything else is split
    into paragraphs, for freeform pasted text.
    """
    chunks = []

    if not os.path.isdir(NOTES_DIR):
        return chunks

    for filename in sorted(os.listdir(NOTES_DIR)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(NOTES_DIR, filename)

        if filepath == ADDED_NOTES_FILE:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        chunks.append(line)
        else:
            with open(filepath, "r") as f:
                content = f.read()
            for para in content.split("\n\n"):
                para = para.strip()
                if para and not para.startswith("#"):
                    chunks.append(para)

    return chunks


def find_relevant_notes(query, top_n=3, min_score=1):
    """
    Scores every note chunk by how many meaningful words it shares with the query,
    and returns the top matches. Cheap keyword overlap — no embeddings, no extra
    model — well suited to a small personal notes collection on limited hardware.
    """
    chunks = load_note_chunks()
    if not chunks:
        return []

    query_words = set(re.findall(r"[a-z0-9]+", query.lower())) - NOTE_STOPWORDS
    if not query_words:
        return []

    scored = []
    for chunk in chunks:
        chunk_words = set(re.findall(r"[a-z0-9]+", chunk.lower()))
        score = len(query_words & chunk_words)
        if score >= min_score:
            scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored[:top_n]]


def find_relevant_log_entries(query, top_n=3, min_score=2):
    """
    Searches the conversation log (past sessions, up to 2 weeks back) for lines
    sharing meaningful words with the query. Same cheap keyword approach as notes.
    Uses a slightly higher min_score than notes since the log is much noisier —
    full of casual chat, not curated facts, so we want a stronger match before
    surfacing something from it.
    """
    if not os.path.exists(CONVO_LOG_FILE):
        return []

    with open(CONVO_LOG_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip() and LOG_LINE_PATTERN.match(line)]

    if not lines:
        return []

    query_words = set(re.findall(r"[a-z0-9]+", query.lower())) - NOTE_STOPWORDS
    if not query_words:
        return []

    scored = []
    for line in lines:
        line_words = set(re.findall(r"[a-z0-9]+", line.lower()))
        score = len(query_words & line_words)
        if score >= min_score:
            scored.append((score, line))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [line for _, line in scored[:top_n]]


LOG_SEARCH_PATTERNS = [
    r"check (our |the )?(past )?conversations?( history)?( about| for)?\s*",
    r"search (our |the )?(chat )?(history|log)( about| for)?\s*",
    r"look (through|back) (our |at )?(past )?(chats?|conversations?)( about| for)?\s*",
]


def try_log_search(user_text):
    """
    Checks if user_text is a request to search past conversations. Returns:
      - the search topic (str) if a trigger phrase and topic were both said together
      - "" (empty string) if the trigger phrase was said alone, meaning we should ask what to search for
      - None if this isn't a log-search command at all
    """
    lowered_text = user_text.lower()
    for pattern in LOG_SEARCH_PATTERNS:
        match = re.search(pattern, lowered_text)
        if match:
            topic = user_text[match.end():].strip()
            return topic
    return None


# ---------- Main loop ----------

_speaker_alsa_device_cache = None


def find_speaker_device(name_substring):
    """
    Finds the ALSA card number for the output device whose name contains
    name_substring (case-insensitive) by parsing `aplay -l`, and returns an
    ALSA device string like "plughw:2,0". USB audio card numbers can drift
    across reboots/reconnects, so we look this up by name instead of trusting
    a hardcoded number. Falls back to SPEAKER_ALSA_DEVICE_FALLBACK with a
    warning if no match is found or `aplay -l` can't be read.
    """
    try:
        output = subprocess.run(["aplay", "-l"], capture_output=True, text=True, check=True).stdout
    except Exception as e:
        print(f"Warning: couldn't run 'aplay -l' ({e}) — using fallback speaker device.")
        return SPEAKER_ALSA_DEVICE_FALLBACK

    for line in output.splitlines():
        match = re.match(r"card (\d+):.*?\[(.*?)\],\s*device (\d+):", line)
        if match and name_substring.lower() in line.lower():
            card, device = match.group(1), match.group(3)
            alsa_device = f"plughw:{card},{device}"
            print(f"Using speaker: card {card} ({line.strip()}) -> {alsa_device}")
            return alsa_device

    print(f"Warning: no output device matching '{name_substring}' found — using fallback speaker device.")
    return SPEAKER_ALSA_DEVICE_FALLBACK


def get_speaker_alsa_device():
    """Returns the ALSA device string for the speaker, detecting it once and caching the result."""
    global _speaker_alsa_device_cache
    if _speaker_alsa_device_cache is None:
        _speaker_alsa_device_cache = find_speaker_device(SPEAKER_DEVICE_NAME)
    return _speaker_alsa_device_cache


def find_mic_device(name_substring):
    """
    Finds the input device whose name contains name_substring (case-insensitive)
    and returns its sounddevice index. Falls back to the system default (None)
    with a warning if no match is found, rather than crashing.
    """
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and name_substring.lower() in dev["name"].lower():
            print(f"Using mic: [{i}] {dev['name']}")
            return i
    print(f"Warning: no input device matching '{name_substring}' found — using system default.")
    return None


def main():
    mode = "jimothy"       # conversation personality: any key in PERSONALITIES
    # Start in text mode if launched with --text or -t, otherwise default to voice.
    input_mode = "text" if ("--text" in sys.argv or "-t" in sys.argv) else "voice"
    history = {key: [] for key in PERSONALITIES}
    last_reply = None
    awaiting_note_content = False
    awaiting_log_query = False
    sleeping = False

    ensure_files_exist()

    personality_list = ", ".join(f"'{info['triggers'][0]}'" for info in PERSONALITIES.values())

    print("Deskbuddy is running.")
    print("Voice mode: just start talking. Text mode: type and press Enter.")
    print(f"Say/type 'switch to' one of: {personality_list} to change personality.")
    print("Say/type 'switch to text mode' or 'switch to voice mode' to change input method.")
    print("Say/type 'repeat that' to hear the last response again.")
    print("Say/type 'set a timer for N minutes/seconds' to start a timer.")
    print("Say/type 'add this to notes: ...' to save something for later.")
    print("Tip: drop any .txt file into the notes/ folder (e.g. games.txt) — it'll be searchable too.")
    print("Say/type 'check our past conversations about ...' to search old chats.")
    print("Say/type 'sleep' or 'go to sleep' to pause listening; 'wake up' or 'awaken' to resume.")
    print("Say/type 'what can you do' for a full list of commands.")
    print("Say/type 'bye' to exit, or press Ctrl+C.")
    print(f"Current mode: {PERSONALITIES[mode]['display_label']} | Input: {input_mode}\n")

    q = queue.Queue()
    suppress_event = threading.Event()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    mic_device = find_mic_device(MIC_DEVICE_NAME)
    get_speaker_alsa_device()  # detect + cache speaker now so any warning shows up at startup
    chunk_samples = int(CHUNK_DURATION * SAMPLE_RATE)
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                             blocksize=chunk_samples, device=mic_device, callback=callback)
    stream.start()

    def start_timer(seconds, label):
        def notify():
            message = f"Time's up on your {label}!" if label else "Time's up!"
            print(f"\n[Timer] {message}")
            if input_mode == "voice":
                do_speak(message)
            else:
                print(f"Deskbuddy: {message}")

        t = threading.Timer(seconds, notify)
        t.daemon = True
        t.start()

    def do_speak(text, interruptible=False):
        """Speaks text with the mic muted for the duration, so deskbuddy never hears its own voice."""
        suppress_event.set()
        try:
            if interruptible:
                interrupted = speak_interruptible(text)
            else:
                speak(text)
                interrupted = False
        finally:
            suppress_event.clear()
            while not q.empty():
                q.get_nowait()
        return interrupted

    try:
        while True:
            if input_mode == "voice":
                # Clear out any audio queued up while we were speaking or thinking,
                # so we don't accidentally process leftover/echoed audio as new speech.
                while not q.empty():
                    q.get_nowait()

                wav_path = listen_and_record(q, suppress_event)
                if wav_path is None:
                    continue  # nothing meaningful heard, keep listening

                user_text = transcribe(wav_path)
                if not user_text:
                    continue

                print(f"You said: {user_text}")
            else:
                user_text = input("You: ").strip()
                if not user_text:
                    continue

            log_conversation("user", user_text)

            lowered = user_text.lower().strip().rstrip(".!?,")

            if sleeping:
                if lowered in ("wake up", "awaken", "wake"):
                    sleeping = False
                    reply = "I'm awake!"
                    print(f"Deskbuddy: {reply}\n")
                    log_conversation("deskbuddy", reply)
                    if input_mode == "voice":
                        do_speak(reply)
                else:
                    # Asleep: ignore everything except the wake phrase — no reply, no LLM call.
                    print("(sleeping — say 'wake up' to continue)\n")
                continue

            if lowered in ("sleep", "go to sleep"):
                sleeping = True
                awaiting_note_content = False
                awaiting_log_query = False
                reply = "Going to sleep. Say wake up when you need me."
                print(f"Deskbuddy: {reply}\n")
                log_conversation("deskbuddy", reply)
                if input_mode == "voice":
                    do_speak(reply)
                continue

            if awaiting_note_content:
                if lowered in ("nevermind", "never mind", "forget it", "cancel", "nothing"):
                    awaiting_note_content = False
                    reply = "No worries, nothing added."
                else:
                    save_note(user_text)
                    awaiting_note_content = False
                    reply = "Got it, saved that to your notes."
                print(f"Deskbuddy: {reply}\n")
                log_conversation("deskbuddy", reply)
                if input_mode == "voice":
                    do_speak(reply)
                continue

            if awaiting_log_query:
                awaiting_log_query = False
                if lowered in ("nevermind", "never mind", "forget it", "cancel", "nothing"):
                    reply = "No worries."
                else:
                    results = find_relevant_log_entries(user_text, top_n=3, min_score=1)
                    if results:
                        reply = "Here's what I found:\n" + "\n".join(results)
                    else:
                        reply = "I couldn't find anything about that in our past conversations."
                print(f"Deskbuddy: {reply}\n")
                log_conversation("deskbuddy", reply)
                if input_mode == "voice":
                    do_speak(reply)
                continue

            if lowered in ("bye", "goodbye", "bye buddy", "see you later", "see ya"):
                print("Deskbuddy: See ya!\n")
                log_conversation("deskbuddy", "See ya!")
                if input_mode == "voice":
                    do_speak("See ya!")
                break

            matched_personality = None
            for key, info in PERSONALITIES.items():
                if any(phrase in lowered for phrase in info["triggers"]):
                    matched_personality = key
                    break

            if matched_personality:
                mode = matched_personality
                info = PERSONALITIES[mode]
                print(f"Deskbuddy: Switched to {info['display_label']}.\n")
                log_conversation("deskbuddy", info["switch_message"])
                if input_mode == "voice":
                    do_speak(info["switch_message"])
                continue

            if "text mode" in lowered:
                input_mode = "text"
                print("Deskbuddy: Switched to text mode.\n")
                log_conversation("deskbuddy", "Switched to text mode.")
                continue

            if "voice mode" in lowered:
                input_mode = "voice"
                while not q.empty():
                    q.get_nowait()
                print("Deskbuddy: Switched to voice mode.\n")
                log_conversation("deskbuddy", "Switched to voice mode.")
                do_speak("Switched to voice mode.")
                continue

            if "repeat that" in lowered or "say that again" in lowered or "can you repeat" in lowered:
                if last_reply is None:
                    reply = "There's nothing to repeat yet."
                else:
                    reply = last_reply
                print(f"Deskbuddy: {reply}\n")
                log_conversation("deskbuddy", reply)
                if input_mode == "voice":
                    do_speak(reply)
                continue

            if "set a timer" in lowered or "set timer" in lowered or "timer for" in lowered:
                seconds = parse_duration_seconds(lowered)
                if seconds is None:
                    reply = "I didn't catch how long — try something like 'set a timer for 10 minutes.'"
                    print(f"Deskbuddy: {reply}\n")
                    log_conversation("deskbuddy", reply)
                    if input_mode == "voice":
                        do_speak(reply)
                    continue

                if seconds >= 3600:
                    label = f"{seconds // 3600} hour timer"
                elif seconds >= 60:
                    label = f"{seconds // 60} minute timer"
                else:
                    label = f"{seconds} second timer"

                start_timer(seconds, label)
                reply = f"Okay, {label} started."
                print(f"Deskbuddy: {reply}\n")
                log_conversation("deskbuddy", reply)
                if input_mode == "voice":
                    do_speak(reply)
                continue

            if "what can you do" in lowered or lowered in ("help", "what are your commands", "what commands can i use"):
                print(f"Deskbuddy: {HELP_TEXT}\n")
                log_conversation("deskbuddy", HELP_TEXT)
                if input_mode == "voice":
                    do_speak(HELP_TEXT, interruptible=True)
                continue

            calc_result = try_calculate(user_text)
            if calc_result is not None:
                reply = f"That's {calc_result}."
                print(f"Deskbuddy: {reply}\n")
                log_conversation("deskbuddy", reply)
                if input_mode == "voice":
                    do_speak(reply)
                continue

            note_content = try_add_note(user_text)
            if note_content is not None:
                if note_content:
                    save_note(note_content)
                    reply = "Got it, saved that to your notes."
                else:
                    awaiting_note_content = True
                    reply = "Sure, what would you like me to add?"
                print(f"Deskbuddy: {reply}\n")
                log_conversation("deskbuddy", reply)
                if input_mode == "voice":
                    do_speak(reply)
                continue

            log_topic = try_log_search(user_text)
            if log_topic is not None:
                if log_topic:
                    results = find_relevant_log_entries(log_topic, top_n=3, min_score=1)
                    if results:
                        reply = "Here's what I found:\n" + "\n".join(results)
                    else:
                        reply = "I couldn't find anything about that in our past conversations."
                else:
                    awaiting_log_query = True
                    reply = "Sure, what should I look for?"
                print(f"Deskbuddy: {reply}\n")
                log_conversation("deskbuddy", reply)
                if input_mode == "voice":
                    do_speak(reply)
                continue

            personality = PERSONALITIES[mode]
            model = personality["model"]
            system_prompt = compose_system_prompt(personality["core_prompt"], personality.get("name"))

            history[mode].append({"role": "user", "content": user_text})
            messages = build_messages(history[mode], system_prompt, query=user_text)

            print("Deskbuddy: ", end="", flush=True)
            try:
                reply = ask_ollama(model, messages)
            except requests.exceptions.ConnectionError:
                print("Can't reach Ollama — is it running? Try 'ollama serve' in another terminal.\n")
                history[mode].pop()
                continue

            history[mode].append({"role": "assistant", "content": reply})
            last_reply = reply
            log_conversation("deskbuddy", reply)
            print()

            if input_mode == "voice":
                do_speak(reply, interruptible=True)
            print()

    except KeyboardInterrupt:
        print("\nDeskbuddy: See ya!")
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    main()
