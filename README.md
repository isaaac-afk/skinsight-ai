# VoiceDrop

A free, offline voice dictation tool for Windows. Press a hotkey, speak, and your words appear in any text field — browser, Word, email, or any application. No subscriptions, no cloud processing, and no data leaves your machine.

Built as a free alternative to Wispr Flow using OpenAI's Whisper model running fully locally.

## Features

- **Fully offline** — audio is processed locally and never leaves your computer
- **System-wide compatibility** — works in browsers, documents, messaging apps, or any text field
- **Auto-stop on silence** — pauses for 1.5 seconds and transcribes automatically
- **Hallucination filtering** — filters out common Whisper artifacts produced on silence or background noise
- **Clipboard-safe** — preserves the original clipboard contents after pasting
- **Zero cost** — no API keys, no subscriptions, no usage limits

## Demo

```
========================================================
  VoiceDrop - Offline Voice Dictation
========================================================
  Model loaded!
  Press SHIFT+SPACE to START. Press again to STOP.
  Auto-stop after 1.5s of silence.
========================================================
  >>> Recording...
  Auto-stopped (silence detected)
  Transcribing...
  Done: 'Hey, this is a test of my voice dictation tool.'
```

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install ffmpeg

```bash
winget install ffmpeg
```

### 3. Run

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
python dictate.py
```

**Note:** Run Command Prompt as Administrator so the global hotkey can be registered system-wide.

The first run will download the Whisper model (~140MB for `base`). This is a one-time download.

## Usage

1. Click into any text field
2. Press **Shift + Space**
3. Speak naturally
4. Pause for 1.5 seconds (or press the hotkey again) and the text appears at your cursor

Close the application window or press **Esc** in the terminal to quit.

## Configuration

Edit the top of `dictate.py` to customize behavior:

| Setting | Description | Default |
|---|---|---|
| `WHISPER_MODEL` | `tiny`, `base`, `small`, `medium`, `large` | `base` |
| `SILENCE_LIMIT` | Seconds of silence before auto-stop | `1.5` |
| `MAX_DURATION` | Hard cap on recording length (seconds) | `60.0` |
| `HOTKEY` | Global keyboard shortcut | `shift+space` |

### Model selection

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| tiny | 75MB | Fastest | Adequate |
| base | 140MB | Fast | Recommended |
| small | 460MB | Medium | Better |
| medium | 1.5GB | Slow | Great |
| large | 3GB | Slowest | Best |

## How It Works

1. PyAudio captures microphone input in real time
2. RMS energy detection identifies silence and triggers auto-stop
3. The audio buffer is written to a temporary WAV file
4. OpenAI Whisper transcribes the file locally
5. A hallucination filter removes common artifacts produced on silence
6. PyAutoGUI pastes the result into the currently focused window

## Tech Stack

- **OpenAI Whisper** — speech-to-text inference
- **PyAudio** — microphone input capture
- **NumPy** — real-time audio signal analysis
- **keyboard** — global hotkey registration
- **Tkinter** — minimal status indicator overlay
- **pyperclip + pyautogui** — clipboard-based text injection

## License

MIT

## Author

Built by Isaac as an open-source alternative to paid dictation tools.
