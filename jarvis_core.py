import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import tkinter as tk
import threading
import time
import math
import subprocess
import webbrowser
import keyboard
import pyttsx3
import ollama
from ddgs import DDGS
from faster_whisper import WhisperModel
import speech_recognition as sr
import numpy as np
from colorama import init, Fore, Style

init(autoreset=True)

print(Fore.GREEN + "Starting Jarvis... loading models, this may take a few minutes on first run.", flush=True)

popup = None
canvas = None
orb_state = "idle"
orb_t = 0

def create_popup():
    global popup, canvas
    popup = tk.Tk()
    popup.overrideredirect(True)
    popup.attributes("-topmost", True)
    size = 220
    popup.geometry(f"{size}x{size}+780+60")
    popup.configure(bg="black")
    popup.attributes("-transparentcolor", "black")
    canvas = tk.Canvas(popup, width=size, height=size, bg="black", highlightthickness=0)
    canvas.pack()

def draw_orb():
    global orb_t
    if canvas is None:
        return
    canvas.delete("all")
    cx, cy = 110, 110
    t = orb_t

    if orb_state == "speaking":
        core_color = "#00FFC6"
        ring_color = "#00FFC6"
    elif orb_state == "listening":
        core_color = "#00A3FF"
        ring_color = "#00A3FF"
    else:
        core_color = "#0066AA"
        ring_color = "#004C80"

    for i in range(8):
        angle = math.radians(t * 60 + i * 45)
        x1 = cx + 95 * math.cos(angle)
        y1 = cy + 95 * math.sin(angle)
        x2 = cx + 80 * math.cos(angle)
        y2 = cy + 80 * math.sin(angle)
        canvas.create_line(x1, y1, x2, y2, fill=ring_color, width=3)

    for i in range(12):
        angle = math.radians(-t * 90 + i * 30)
        x = cx + 65 * math.cos(angle)
        y = cy + 65 * math.sin(angle)
        canvas.create_oval(x-2, y-2, x+2, y+2, fill=ring_color, outline="")

    if orb_state == "speaking":
        pulse = 28 + 10 * abs(math.sin(t * 8))
    elif orb_state == "listening":
        pulse = 28 + 6 * abs(math.sin(t * 3))
    else:
        pulse = 28 + 3 * abs(math.sin(t * 1.2))

    canvas.create_oval(cx-pulse, cy-pulse, cx+pulse, cy+pulse, fill=core_color, outline="")
    canvas.create_oval(cx-pulse*0.5, cy-pulse*0.5, cx+pulse*0.5, cy+pulse*0.5, fill="#FFFFFF", outline="")

    for i in range(3):
        angle = math.radians(t * 120 + i * 120)
        px = cx + 45 * math.cos(angle)
        py = cy + 45 * math.sin(angle)
        canvas.create_oval(px-4, py-4, px+4, py+4, fill=core_color, outline="")

def pump_orb():
    global orb_t
    if popup is None:
        return
    orb_t += 0.05
    draw_orb()
    try:
        popup.update()
    except tk.TclError:
        pass

def show_popup():
    global popup
    if popup is None:
        create_popup()
    pump_orb()

def hide_popup():
    global popup, canvas
    if popup is not None:
        popup.destroy()
        popup = None
        canvas = None

def set_orb_state(state):
    global orb_state
    orb_state = state

print(Fore.GREEN + "Loading wake-word model (tiny)...", flush=True)
wake_model = WhisperModel("tiny", device="cuda", compute_type="float16")
print(Fore.GREEN + "Loading conversation model (medium)...", flush=True)
whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")
print(Fore.GREEN + "Models loaded successfully.", flush=True)

recognizer = sr.Recognizer()
interrupt_flag = threading.Event()
voice_stop_triggered = threading.Event()

def search_web(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        summary = "\n".join([f"{r['title']}: {r['body']}" for r in results])
        return summary
    except Exception as e:
        return f"Search failed: {e}"

def voice_interrupt_listener():
    while not interrupt_flag.is_set():
        try:
            with sr.Microphone(sample_rate=16000) as source:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=2)
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = wake_model.transcribe(audio_data, language="en", beam_size=1)
            text = " ".join([s.text for s in segments]).strip().lower()
            if "stop" in text or "wait" in text:
                voice_stop_triggered.set()
                interrupt_flag.set()
        except sr.WaitTimeoutError:
            continue
        except Exception:
            continue

def speak(text, allow_interrupt=True):
    print(Fore.CYAN + Style.BRIGHT + "Jarvis: " + Style.RESET_ALL + Fore.WHITE + text, flush=True)
    set_orb_state("speaking")
    interrupt_flag.clear()
    voice_stop_triggered.clear()
    engine = pyttsx3.init()
    engine.say(text)

    if allow_interrupt:
        threading.Thread(target=voice_interrupt_listener, daemon=True).start()

    engine.startLoop(False)
    while engine.isBusy():
        engine.iterate()
        pump_orb()
        time.sleep(0.01)
        if keyboard.is_pressed('space') or interrupt_flag.is_set():
            engine.stop()
            break
    engine.endLoop()
    interrupt_flag.set()
    set_orb_state("idle")
    pump_orb()

    if voice_stop_triggered.is_set():
        speak("Yes? What do you need?", allow_interrupt=False)

def listen():
    set_orb_state("listening")
    pump_orb()
    with sr.Microphone(sample_rate=16000) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.pause_threshold = 3
        audio = recognizer.listen(source)

    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = whisper_model.transcribe(
        audio_data,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5,
        best_of=5,
        condition_on_previous_text=False,
        temperature=0.0
    )
    text = " ".join([segment.text for segment in segments]).strip()
    set_orb_state("idle")
    pump_orb()

    if text:
        print(Fore.YELLOW + "You said: " + Style.RESET_ALL + text, flush=True)
        return text
    else:
        return None

APP_PATHS = {
    "google chrome": "chrome",
    "google": "chrome",
    "chrome": "chrome",
    "microsoft edge": "msedge",
    "edge": "msedge",
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "notepad": "notepad",
    "calculator": "calc",
    "paint": "mspaint",
    "file explorer": "explorer",
    "files": "explorer",
    "settings": "start ms-settings:",
    "task manager": "taskmgr",
    "control panel": "control",
    "command prompt": "cmd",
    "cmd": "cmd",
    "terminal": "wt",
    "spotify": "spotify",
    "vs code": "code",
    "visual studio code": "code",
    "vlc": "vlc",
    "whatsapp": "whatsapp:",
    "camera": "microsoft.windows.camera:",
    "photos": "ms-photos:",
    "mail": "outlookmail:",
    "store": "ms-windows-store:",
}

PROCESS_NAMES = {
    "chrome": "chrome.exe",
    "google": "chrome.exe",
    "edge": "msedge.exe",
    "word": "winword.exe",
    "excel": "excel.exe",
    "powerpoint": "powerpnt.exe",
    "notepad": "notepad.exe",
    "spotify": "spotify.exe",
    "vs code": "Code.exe",
    "vlc": "vlc.exe",
}

def find_app_key(text):
    """Return the longest matching app key found anywhere in text."""
    matches = [key for key in APP_PATHS if key in text]
    if not matches:
        return None
    return max(matches, key=len)

def handle_command(prompt):
    p = prompt.lower()

    # ---- Open app ----
    if "open " in p:
        key = find_app_key(p)
        if key:
            command = APP_PATHS[key]
            try:
                subprocess.Popen(f"start {command}", shell=True)
                return f"Opening {key.title()}."
            except Exception as e:
                return f"I couldn't open {key}. Error: {e}"
        after = p.split("open ", 1)[1].strip()
        after = after.split(" ")[0] if after else ""
        if after:
            try:
                subprocess.Popen(f"start {after}", shell=True)
                return f"Trying to open {after}."
            except Exception:
                pass
        return "I couldn't find that app. Tell me its exact name and I'll remember it."

    # ---- Close app ----
    if "close " in p:
        for key, proc in PROCESS_NAMES.items():
            if key in p:
                subprocess.Popen(f"taskkill /IM {proc} /F", shell=True)
                return f"Closing {key.title()}."
        return "I don't know how to close that yet."

    # ---- Lock ----
    if "lock" in p and ("laptop" in p or "screen" in p or "pc" in p or "computer" in p):
        subprocess.Popen("rundll32.exe user32.dll,LockWorkStation", shell=True)
        return "Locking your laptop now."

    # ---- Web search ----
    if "search for" in p:
        query = p.split("search for", 1)[1].strip()
        webbrowser.open(f"https://www.google.com/search?q={query}")
        return f"Searching for {query}."

    return None

def ask_jarvis(prompt):
    command_result = handle_command(prompt)
    if command_result is not None:
        return command_result

    keywords = ["today", "current", "latest", "news", "who is", "what happened", "weather", "price", "score", "date"]
    needs_search = any(word in prompt.lower() for word in keywords)

    if needs_search:
        print(Fore.MAGENTA + "[Searching the web...]", flush=True)
        search_results = search_web(prompt)
        full_prompt = f"Using this real-time information:\n{search_results}\n\nAnswer this question: {prompt}"
    else:
        full_prompt = prompt

    response = ollama.chat(model="jarvis", messages=[
        {"role": "user", "content": full_prompt}])
    return response['message']['content']

def listen_for_wake_word():
    with sr.Microphone(sample_rate=16000) as source:
        audio = recognizer.listen(source, phrase_time_limit=3)

    audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = wake_model.transcribe(audio_data, language="en", beam_size=1)
    text = " ".join([segment.text for segment in segments]).strip().lower()
    return text

print(Fore.GREEN + "Calibrating microphone...", flush=True)
with sr.Microphone(sample_rate=16000) as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)

print(Fore.GREEN + "Jarvis is sleeping. Say 'hello' to begin.", flush=True)

while True:
    heard = listen_for_wake_word()
    if "hello" in heard:
        show_popup()
        speak("I'm listening. How can I help you?")

        while True:
            user_input = listen()
            if user_input is None:
                continue

            lowered = user_input.lower()

            if "hide" in lowered and ("interface" in lowered or "face" in lowered):
                hide_popup()
                continue

            if "hello" in lowered:
                show_popup()
                speak("I'm here. How can I help you?")
                continue

            if any(word in lowered for word in ["stop", "break"]):
                speak("Shutting down.")
                hide_popup()
                os._exit(0)

            reply = ask_jarvis(user_input)
            show_popup()
            speak(reply)