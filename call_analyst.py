#!/usr/bin/env python3
"""Real-time call analyst. Captures system audio, transcribes live, provides AI analysis.
Runs as a WebSocket server for the Electron GUI, or with --terminal for Rich TUI."""

import argparse
import asyncio
import json
import os
import queue
import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import websockets
from faster_whisper import WhisperModel

# ─── Audio Config ───
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1
ENERGY_THRESHOLD = 0.005
SILENCE_TIMEOUT = 1.5
MIN_SPEECH_DURATION = 0.5
MAX_CHUNK_DURATION = 30.0

# ─── Load .env ───
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

# ─── Groq Config ───
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
ANALYSIS_INTERVAL = 30
MIN_NEW_CHUNKS = 3

# ─── Audio Sources ───
AUDIO_SOURCES = {
    "laptop": "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor",
    "dock": "alsa_output.usb-Generic_USB_Audio_200901010001-00.HiFi__hw_Dock__sink.monitor",
    "mic": "alsa_input.pci-0000_00_1f.3.analog-stereo",
}

# ─── Groq Analysis Prompt ───
ANALYSIS_SYSTEM_PROMPT = """You are a real-time call analyst helping during discovery and sales calls.

Analyze the conversation transcript and return a JSON object with exactly these fields:

{
  "sentiment": "positive|neutral|negative|concerned|excited",
  "key_points": ["Up to 5 most important points discussed so far"],
  "objections": ["Any objections, concerns, or hesitations expressed"],
  "questions_asked": ["Questions asked that may need follow-up"],
  "action_items": ["Any commitments, next steps, or deliverables mentioned"],
  "suggested_responses": ["2-3 tactical suggestions for what to say or ask next"],
  "summary": "2-3 sentence summary of the conversation so far"
}

Rules:
- Focus on what is ACTIONABLE right now
- For suggested_responses, be specific and tactical, not generic
- If you detect an unaddressed concern, flag it in objections
- Keep key_points to genuinely important items
- For discovery calls: suggest probing questions to uncover pain points
- Return ONLY valid JSON, no markdown, no code fences"""


# ─── Data Models ───

@dataclass
class TranscriptEntry:
    timestamp: float
    text: str
    duration: float


@dataclass
class AnalysisResult:
    timestamp: float
    sentiment: str = "neutral"
    key_points: list = field(default_factory=list)
    objections: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    action_items: list = field(default_factory=list)
    suggested_responses: list = field(default_factory=list)
    summary: str = ""


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    transcript: deque = field(default_factory=lambda: deque(maxlen=200))
    latest_analysis: Optional[AnalysisResult] = None
    is_speaking: bool = False
    audio_level: float = 0.0
    chunks_transcribed: int = 0
    status_message: str = "Initializing..."
    paused: bool = False
    start_time: float = 0.0
    save_dir: str = "~/call-transcripts"
    source_name: str = ""


# ─── Hallucination Detection ───

HALLUCINATION_PHRASES = {
    "thank you for watching", "thanks for watching", "please subscribe",
    "like and subscribe", "thank you.", "you", "thanks.", "bye.",
    "thank you for listening", "see you next time",
}


def is_hallucination(text: str) -> bool:
    lower = text.lower().strip().rstrip(".")
    if lower in HALLUCINATION_PHRASES or len(lower) < 3:
        return True
    words = lower.split()
    if len(words) >= 6:
        for phrase_len in range(1, 4):
            phrase = " ".join(words[:phrase_len])
            if lower.count(phrase) >= 3:
                return True
    return False


# ─── Thread 1: Audio Capture ───

def audio_capture_thread(state: SharedState, audio_queue: queue.Queue,
                         stop_event: threading.Event):
    buffer = []
    vad_state = "IDLE"
    silence_start = 0.0
    speech_start = 0.0

    def flush_buffer():
        nonlocal buffer, vad_state
        if buffer:
            audio_array = np.concatenate(buffer)
            duration = len(audio_array) / SAMPLE_RATE
            if duration >= MIN_SPEECH_DURATION:
                try:
                    audio_queue.put((audio_array, duration), timeout=0.5)
                except queue.Full:
                    with state.lock:
                        state.status_message = "Warning: dropping audio chunk"
        buffer = []
        vad_state = "IDLE"
        with state.lock:
            state.is_speaking = False

    def callback(indata, frames, time_info, status):
        nonlocal buffer, vad_state, silence_start, speech_start

        if state.paused:
            return

        chunk = indata[:, 0].copy()
        energy = float(np.sqrt(np.mean(chunk ** 2)))

        with state.lock:
            state.audio_level = energy

        if vad_state == "IDLE":
            if energy > ENERGY_THRESHOLD:
                vad_state = "SPEAKING"
                speech_start = time.time()
                buffer = [chunk]
                with state.lock:
                    state.is_speaking = True

        elif vad_state == "SPEAKING":
            buffer.append(chunk)
            if energy < ENERGY_THRESHOLD:
                vad_state = "TRAILING_SILENCE"
                silence_start = time.time()
            elif time.time() - speech_start > MAX_CHUNK_DURATION:
                flush_buffer()

        elif vad_state == "TRAILING_SILENCE":
            buffer.append(chunk)
            if energy > ENERGY_THRESHOLD:
                vad_state = "SPEAKING"
            elif time.time() - silence_start > SILENCE_TIMEOUT:
                flush_buffer()

    blocksize = int(SAMPLE_RATE * CHUNK_DURATION)
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
        callback=callback, blocksize=blocksize, device=None,
    )

    with stream:
        with state.lock:
            state.status_message = "Listening..."
        while not stop_event.is_set():
            stop_event.wait(0.1)


# ─── Thread 2: Transcription ───

def transcription_thread(state: SharedState, audio_queue: queue.Queue,
                          analysis_queue: queue.Queue, model: WhisperModel,
                          stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            audio_array, duration = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            segments, info = model.transcribe(
                audio_array, language="en", beam_size=3,
                vad_filter=True, condition_on_previous_text=False,
                no_speech_threshold=0.6,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()

            if text and not is_hallucination(text):
                entry = TranscriptEntry(timestamp=time.time(), text=text, duration=duration)
                with state.lock:
                    state.transcript.append(entry)
                    state.chunks_transcribed += 1
                    state.status_message = "Listening..."
                analysis_queue.put("new_text")

        except Exception as e:
            with state.lock:
                state.status_message = f"Transcription error: {e}"


# ─── Thread 3: Analysis ───

def analysis_thread(state: SharedState, analysis_queue: queue.Queue,
                     stop_event: threading.Event):
    last_analysis_time = 0.0
    last_analyzed_count = 0

    while not stop_event.is_set():
        try:
            analysis_queue.get(timeout=5.0)
        except queue.Empty:
            pass

        now = time.time()
        with state.lock:
            current_count = state.chunks_transcribed
            transcript_copy = list(state.transcript)

        if current_count - last_analyzed_count < MIN_NEW_CHUNKS and now - last_analysis_time < ANALYSIS_INTERVAL:
            continue
        if not transcript_copy:
            continue

        lines = []
        for e in transcript_copy:
            t = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
            lines.append(f"[{t}] {e.text}")
        full_text = "\n".join(lines)

        words = full_text.split()
        if len(words) > 2000:
            full_text = " ".join(words[-2000:])

        with state.lock:
            state.status_message = "Analyzing..."

        try:
            result = call_groq_analysis(full_text)
            if result:
                with state.lock:
                    state.latest_analysis = result
                    state.status_message = "Listening..."
                last_analysis_time = now
                last_analyzed_count = current_count
        except Exception as e:
            with state.lock:
                state.status_message = f"Analysis error: {e}"


def call_groq_analysis(transcript_text: str) -> Optional[AnalysisResult]:
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this live call transcript:\n\n{transcript_text}"},
            ],
            "temperature": 0.1,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
        },
        timeout=10,
    )

    if response.status_code != 200:
        return None

    content = response.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None

    return AnalysisResult(
        timestamp=time.time(),
        sentiment=data.get("sentiment", "neutral"),
        key_points=data.get("key_points", []),
        objections=data.get("objections", []),
        questions=data.get("questions_asked", []),
        action_items=data.get("action_items", []),
        suggested_responses=data.get("suggested_responses", []),
        summary=data.get("summary", ""),
    )


# ─── Transcript Save ───

def save_transcript(state: SharedState) -> Optional[str]:
    save_path = Path(state.save_dir).expanduser()
    save_path.mkdir(parents=True, exist_ok=True)

    with state.lock:
        entries = list(state.transcript)
        analysis = state.latest_analysis

    if not entries:
        return None

    timestamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(state.start_time))
    filepath = save_path / f"call_{timestamp}.md"
    duration = time.time() - state.start_time
    duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))

    lines = [
        f"# Call Transcript - {time.strftime('%Y-%m-%d %H:%M', time.localtime(state.start_time))}",
        "", f"Duration: {duration_str}", "", "## Transcript", "",
    ]
    for entry in entries:
        t = time.strftime("%H:%M:%S", time.localtime(entry.timestamp))
        lines.append(f"**[{t}]** {entry.text}")
        lines.append("")

    if analysis:
        lines.append("## Analysis")
        lines.append("")
        lines.append(f"**Sentiment:** {analysis.sentiment}")
        lines.append("")
        for label, items in [
            ("Key Points", analysis.key_points),
            ("Objections / Concerns", analysis.objections),
            ("Unanswered Questions", analysis.questions),
            ("Suggested Responses", analysis.suggested_responses),
        ]:
            if items:
                lines.append(f"### {label}")
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")
        if analysis.action_items:
            lines.append("### Action Items")
            for a in analysis.action_items:
                lines.append(f"- [ ] {a}")
            lines.append("")
        lines.append("### Summary")
        lines.append(analysis.summary)
        lines.append("")

    filepath.write_text("\n".join(lines))
    return str(filepath)


# ─── WebSocket Server ───

def get_state_json(state: SharedState) -> str:
    with state.lock:
        entries = [
            {"timestamp": e.timestamp, "time": time.strftime("%H:%M:%S", time.localtime(e.timestamp)), "text": e.text}
            for e in state.transcript
        ]
        analysis = None
        if state.latest_analysis:
            a = state.latest_analysis
            analysis = {
                "sentiment": a.sentiment,
                "key_points": a.key_points,
                "objections": a.objections,
                "questions": a.questions,
                "action_items": a.action_items,
                "suggested_responses": a.suggested_responses,
                "summary": a.summary,
            }

        elapsed = time.time() - state.start_time
        data = {
            "type": "state",
            "transcript": entries,
            "analysis": analysis,
            "is_speaking": state.is_speaking,
            "audio_level": state.audio_level,
            "chunks": state.chunks_transcribed,
            "elapsed": elapsed,
            "elapsed_str": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
            "status": state.status_message,
            "paused": state.paused,
            "source": state.source_name,
        }

    return json.dumps(data)


async def ws_handler(websocket, state: SharedState, stop_event: threading.Event):
    try:
        # Send initial state
        await websocket.send(get_state_json(state))

        async def send_updates():
            while not stop_event.is_set():
                await websocket.send(get_state_json(state))
                await asyncio.sleep(0.25)

        async def receive_commands():
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                    if cmd.get("cmd") == "pause":
                        with state.lock:
                            state.paused = not state.paused
                    elif cmd.get("cmd") == "save":
                        filepath = save_transcript(state)
                        await websocket.send(json.dumps({"type": "saved", "path": filepath or ""}))
                    elif cmd.get("cmd") == "source":
                        src = cmd.get("value", "mic")
                        if src in AUDIO_SOURCES:
                            os.environ["PULSE_SOURCE"] = AUDIO_SOURCES[src]
                            with state.lock:
                                state.source_name = src
                except json.JSONDecodeError:
                    pass

        await asyncio.gather(send_updates(), receive_commands())
    except websockets.exceptions.ConnectionClosed:
        pass


async def run_ws_server(state: SharedState, stop_event: threading.Event, port: int = 8765):
    async with websockets.serve(
        lambda ws: ws_handler(ws, state, stop_event),
        "localhost", port
    ):
        print(f"WebSocket server running on ws://localhost:{port}", flush=True)
        while not stop_event.is_set():
            await asyncio.sleep(0.5)


def ws_server_thread(state: SharedState, stop_event: threading.Event, port: int = 8765):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_ws_server(state, stop_event, port))


# ─── Terminal UI (--terminal mode) ───

def run_terminal_ui(state, start_time):
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel

    SENTIMENT_COLORS = {
        "positive": "green", "excited": "bold green",
        "neutral": "yellow", "negative": "red", "concerned": "magenta",
    }

    def build_layout():
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="transcript", ratio=1),
            Layout(name="analysis", ratio=1),
        )

        elapsed = time.time() - start_time
        with state.lock:
            is_speaking = state.is_speaking
            chunks = state.chunks_transcribed
            level = state.audio_level
            status = state.status_message
            paused = state.paused

        indicator = "[bold yellow]PAUSED[/]" if paused else "[bold red]REC[/]" if is_speaking else "[dim]IDLE[/]"
        vu_bars = int(min(level * 500, 20))
        vu = "|" * vu_bars + "." * (20 - vu_bars)
        layout["header"].update(Panel(
            f"  {indicator}  |  {time.strftime('%H:%M:%S', time.gmtime(elapsed))}  |  Chunks: {chunks}  |  [{vu}]  |  {status}",
            title="[bold]CALL ANALYST[/]", border_style="bright_blue"))

        with state.lock:
            entries = list(state.transcript)
        lines = [f"[dim]{time.strftime('%H:%M:%S', time.localtime(e.timestamp))}[/] {e.text}" for e in entries[-25:]]
        layout["transcript"].update(Panel("\n".join(lines) or "[dim]Waiting...[/]", title="Transcript", border_style="blue"))

        with state.lock:
            analysis = state.latest_analysis
        if analysis:
            parts = []
            color = SENTIMENT_COLORS.get(analysis.sentiment, "white")
            parts.append(f"[{color}]{analysis.sentiment.upper()}[/{color}]\n")
            for label, items, style in [
                ("KEY POINTS", analysis.key_points, "bold"),
                ("OBJECTIONS", analysis.objections, "bold red"),
                ("SUGGESTIONS", analysis.suggested_responses, "bold green"),
                ("QUESTIONS", analysis.questions, "bold yellow"),
                ("ACTION ITEMS", analysis.action_items, "bold cyan"),
            ]:
                if items:
                    parts.append(f"[{style}]{label}[/{style}]")
                    for i in items:
                        parts.append(f"  * {i}")
                    parts.append("")
            parts.append(f"[dim]---[/]\n[italic]{analysis.summary}[/italic]")
            analysis_text = "\n".join(parts)
        else:
            analysis_text = "[dim]Waiting for analysis...[/]"
        layout["analysis"].update(Panel(analysis_text, title="Analysis", border_style="green"))
        layout["footer"].update(Panel("[dim]Ctrl+C[/] quit & save", style="dim"))
        return layout

    console = Console()
    with Live(build_layout(), refresh_per_second=4, console=console, screen=True) as live:
        while True:
            live.update(build_layout())
            time.sleep(0.25)


# ─── Main ───

def main():
    global ENERGY_THRESHOLD, SILENCE_TIMEOUT

    parser = argparse.ArgumentParser(description="Real-time call analyst")
    parser.add_argument("--source", choices=list(AUDIO_SOURCES.keys()),
                        default="mic", help="Audio source (default: mic)")
    parser.add_argument("--source-raw", type=str, default=None,
                        help="Raw PulseAudio source name")
    parser.add_argument("--energy", type=float, default=ENERGY_THRESHOLD,
                        help=f"VAD energy threshold (default: {ENERGY_THRESHOLD})")
    parser.add_argument("--silence", type=float, default=SILENCE_TIMEOUT,
                        help=f"Silence timeout seconds (default: {SILENCE_TIMEOUT})")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Disable Groq analysis")
    parser.add_argument("--save-dir", type=str, default="~/call-transcripts",
                        help="Transcript save directory")
    parser.add_argument("--whisper-model", type=str, default="small",
                        help="Whisper model size (default: small)")
    parser.add_argument("--terminal", action="store_true",
                        help="Use terminal UI instead of WebSocket server")
    parser.add_argument("--port", type=int, default=8765,
                        help="WebSocket server port (default: 8765)")
    args = parser.parse_args()

    ENERGY_THRESHOLD = args.energy
    SILENCE_TIMEOUT = args.silence

    if not args.no_analysis and not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not set. Use --no-analysis for transcript-only mode.")
        sys.exit(1)

    source = args.source_raw or AUDIO_SOURCES.get(args.source, AUDIO_SOURCES["mic"])
    os.environ["PULSE_SOURCE"] = source

    print(f"Loading Whisper model ({args.whisper_model})...", flush=True)
    model = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")
    print(f"Model loaded. Source: {source}", flush=True)

    state = SharedState()
    state.start_time = time.time()
    state.save_dir = args.save_dir
    state.source_name = args.source

    audio_queue = queue.Queue(maxsize=50)
    analysis_queue = queue.Queue()
    stop_event = threading.Event()

    threads = [
        threading.Thread(target=audio_capture_thread,
                         args=(state, audio_queue, stop_event), daemon=True),
        threading.Thread(target=transcription_thread,
                         args=(state, audio_queue, analysis_queue, model, stop_event), daemon=True),
    ]
    if not args.no_analysis:
        threads.append(threading.Thread(target=analysis_thread,
                                         args=(state, analysis_queue, stop_event), daemon=True))

    for t in threads:
        t.start()

    try:
        if args.terminal:
            run_terminal_ui(state, state.start_time)
        else:
            ws_server_thread(state, stop_event, args.port)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        filepath = save_transcript(state)
        if filepath:
            print(f"Transcript saved: {filepath}", flush=True)
        print("Goodbye.", flush=True)


if __name__ == "__main__":
    main()
