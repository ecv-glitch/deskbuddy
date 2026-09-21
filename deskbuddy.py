"""
Deskbuddy - v1 (text-only, no voice yet)
Talks to local Ollama models with two personality modes:
  - fast:  quick, casual friend chat (qwen2.5:3b)
  - smart: patient, approachable TA for homework help (qwen2.5:7b)

Type 'smart' or 'fast' anytime to switch modes.
Type 'quit' to exit.
"""

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

FAST_MODEL = "qwen2.5:3b"
SMART_MODEL = "qwen2.5:7b"

TUTOR_PROMPT = """You're a knowledgeable teaching assistant helping a CS/math student understand
concepts, not just get answers. You're approachable and easygoing, not stiff or overly formal —
think "cool TA who explains things well" rather than "textbook."

Guide understanding step by step rather than just handing over final answers unless directly asked.
Feel free to use casual language, light humor, or a relatable analogy when it helps something click.
Keep responses focused and not overly long for voice conversation, but don't sacrifice clarity for brevity.

IMPORTANT: You are a small local AI model running offline on someone's own computer, still being
built and tested as a personal project. You have no internet access and cannot look up real-time
information — no current prices, no live data, no search results, nothing happening in the world
right now. If asked to check, look up, or find something that would require internet access, be
upfront that you can't actually do that rather than pretending you can or implying you looked
something up. It's fine to say you're not sure, or that you don't have a way to verify something,
when that's genuinely true — honesty matters more than sounding capable."""

FRIEND_PROMPT = """You're a warm, easygoing friend having a normal, relaxed conversation — think
texting a close friend. Keep responses short and natural, usually 1-2 sentences. Talk like a real
person actually would: mostly just genuine and conversational, not performing or trying hard to be
funny. Occasional light humor is fine when it naturally fits, but don't force a joke or lighthearted
comment into every single response — most replies should just be normal and down-to-earth.

Don't overreact to neutral or ambiguous things by assuming something bad happened — give people the
benefit of the doubt and keep things chill unless they clearly say something is actually wrong.
Don't over-explain or lecture; just chat like a normal person would.

IMPORTANT: You are a small local AI model running offline on someone's own computer, still being
built and tested as a personal project. You have no internet access and cannot look up real-time
information — no current prices, no live data, no search results, nothing happening in the world
right now. If asked to check, look up, or find something that would require internet access, be
upfront that you can't actually do that rather than pretending you can or implying you looked
something up. It's fine to say you're not sure, or that you don't have a way to verify something,
when that's genuinely true — honesty matters more than sounding capable."""


def build_messages(history, system_prompt):
    """Combine system prompt + conversation history into the message list Ollama expects."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    return messages


def ask_ollama(model, messages):
    """Streams the response, printing words as they arrive, and returns the full text at the end."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "stream": True,
        },
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

    print()  # newline after the streamed response finishes
    return full_reply


def main():
    mode = "fast"  # default on start
    # Separate history per mode so switching doesn't mix TA and friend context
    history = {"fast": [], "smart": []}

    print("Deskbuddy is running. Type 'smart' or 'fast' to switch modes, 'quit' to exit.")
    print(f"Current mode: {mode}\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        lowered = user_input.lower()

        if lowered == "quit":
            print("Deskbuddy: See ya!")
            break

        if lowered == "smart":
            mode = "smart"
            print("Deskbuddy: Switched to smart mode (TA).\n")
            continue

        if lowered == "fast":
            mode = "fast"
            print("Deskbuddy: Switched to fast mode (friend).\n")
            continue

        # Pick model + prompt based on current mode
        if mode == "smart":
            model = SMART_MODEL
            system_prompt = TUTOR_PROMPT
        else:
            model = FAST_MODEL
            system_prompt = FRIEND_PROMPT

        # Add user message to this mode's history
        history[mode].append({"role": "user", "content": user_input})

        messages = build_messages(history[mode], system_prompt)

        print("Deskbuddy: ", end="", flush=True)
        try:
            reply = ask_ollama(model, messages)
        except requests.exceptions.ConnectionError:
            print("Can't reach Ollama — is it running? Try 'ollama serve' in another terminal.\n")
            history[mode].pop()  # remove the user message since we didn't get a reply
            continue

        # Add assistant reply to history so context carries forward
        history[mode].append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
