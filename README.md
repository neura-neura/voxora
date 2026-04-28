<p align="center">
  <img src="assets/img/icon.png" alt="Voxora logo" width="220">
</p>

# Voxora

Voxora is a local desktop transcription app for Windows. It lets you transcribe
audio files, video files, and supported URLs using `whisper.cpp` through the
`pywhispercpp` Python bindings. The interface is built with PyQt6 and keeps the
transcription engine local by default.

## Features

- Drag and drop audio or video files into the app.
- Pick files through a standard file dialog.
- Transcribe YouTube and Bilibili URLs when `yt-dlp` and `ffmpeg` are installed.
- Convert video or non-compatible audio to 16 kHz mono WAV through `ffmpeg`.
- Use Whisper models managed by `pywhispercpp`.
- Choose model size from the app.
- Use automatic language detection or manually select a language.
- Translate supported audio to English.
- Edit, copy, save, and manage transcription history.
- Use a dark PyQt6 desktop interface.
- Build a standalone Windows executable with PyInstaller.

## Privacy

Voxora is designed to run locally. Audio/video processing and transcription are
performed on your machine with `whisper.cpp`.

Important notes:

- The app does not require an API key.
- The source code does not include private tokens or credentials.
- Transcription history is local user data and is intentionally ignored by Git.
- Whisper model files are downloaded and stored locally; they are not committed.
- URL transcription depends on `yt-dlp`, which accesses the requested media URL.

## Requirements

For running from source:

- Windows 10 or later.
- Python 3.10 or later. Python 3.12 was used for the packaged build.
- `ffmpeg` available on `PATH`.
- Optional: `yt-dlp` available on `PATH` for URL transcription.
- Enough free disk space for Whisper model files. Larger models can require
  multiple gigabytes.

For using the packaged release:

- Windows 10 or later.
- `ffmpeg` on `PATH` for video files, audio conversion, and URL transcription.
- Optional: `yt-dlp` on `PATH` for URL transcription.

## Install ffmpeg

Voxora calls the `ffmpeg` command directly.

1. Download a Windows build from [ffmpeg.org](https://ffmpeg.org/download.html)
   or install it with a package manager.
2. Add the folder that contains `ffmpeg.exe` to your `PATH`.
3. Open a new terminal and verify:

```powershell
ffmpeg -version
```

With Winget, you can usually install it with:

```powershell
winget install Gyan.FFmpeg
```

## Install yt-dlp

`yt-dlp` is only required for URL transcription.

```powershell
python -m pip install --upgrade yt-dlp
yt-dlp --version
```

If YouTube extraction requires a JavaScript runtime, install Node.js and make
sure `node` is available on `PATH`.

## Run From Source

Clone the repository:

```powershell
git clone https://github.com/neura-neura/voxora.git
cd voxora
```

Create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\activate
```

Upgrade pip and install Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install `ffmpeg` and, if you want URL transcription, install `yt-dlp` as
described above.

Run the app:

```powershell
python script.py
```

## First Run

On the first transcription, Voxora may download the selected Whisper model.
Download time depends on the selected model size and your connection. Later
transcriptions reuse the model stored on disk.

Model storage is managed locally by the app and `pywhispercpp`. Model files are
not part of the repository.

## Using Voxora

1. Start the app.
2. Choose the input mode:
   - Files: drag and drop audio/video files or select them from disk.
   - Links: paste a supported YouTube or Bilibili URL.
3. Select a transcription language or use automatic detection.
4. Pick a Whisper model. Smaller models are faster; larger models are usually
   more accurate.
5. Start transcription.
6. Edit the result if needed.
7. Copy, save, or manage the transcript from the app.

## Supported Inputs

Voxora can process common audio and video formats as long as `ffmpeg` can read
them. Examples include:

- Audio: WAV, MP3, M4A, OGG, FLAC.
- Video: MP4, MKV, MOV, AVI, WEBM.
- URLs: YouTube and Bilibili through `yt-dlp`.

## Build a Windows Executable

Install the build dependencies:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install pyinstaller
```

Build the one-file executable:

```powershell
pyinstaller Voxora.spec
```

The executable will be created at:

```text
dist/Voxora.exe
```

Build folders such as `build/`, `dist/`, and virtual environments are ignored by
Git.

## Release Package

The GitHub release provides a Windows executable asset. Download the release
asset, place it wherever you prefer, and run `Voxora.exe`.

`ffmpeg` and `yt-dlp` are not bundled in the release asset. Install them
separately when you need video conversion or URL transcription.

## Project Structure

```text
.
├── assets/
│   └── img/
│       ├── icon.ico
│       └── icon.png
├── script.py
├── Voxora.spec
├── requirements.txt
├── README.md
└── LICENSE
```

## Files Intentionally Excluded

The repository intentionally excludes:

- Virtual environments: `venv/`, `venv_build/`, `.venv/`.
- Build output: `build/`, `dist/`.
- Local transcription history: `transcripts_history.json`.
- Whisper model files: `ggml-*.bin`, `*.gguf`.
- Logs, caches, temporary files, and local environment files.

## Troubleshooting

### `ffmpeg` was not found

Install `ffmpeg`, add it to `PATH`, close and reopen your terminal, then run:

```powershell
ffmpeg -version
```

### URL transcription is disabled

Install both `yt-dlp` and `ffmpeg`, then restart Voxora:

```powershell
pip install --upgrade yt-dlp
yt-dlp --version
ffmpeg -version
```

### The first transcription is slow

The first transcription can download and load a Whisper model. Choose a smaller
model for faster startup and lower memory use.

### A larger model runs out of memory

Use a smaller Whisper model. Large models require substantially more RAM and
disk space.

### PyQt6 or pywhispercpp fails to install

Upgrade pip and try again inside a fresh virtual environment:

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Development Notes

- The app entry point is `script.py`.
- The app icon is loaded from `assets/img/icon.ico`.
- `Voxora.spec` configures the PyInstaller build.
- The app uses `QSettings("neura", "Voxora")` for user settings.
- Transcription history is local runtime data and should not be committed.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
