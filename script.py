"""
PyQt6-based audio and video transcription tool.

This application allows you to drag & drop audio/video files (e.g. OGG, MP3, M4A,
MP4, MKV, etc.) onto a drop zone or use a file dialog to pick them.  Internally it
extracts audio from video using ffmpeg and then runs a speech‑to‑text model
powered by `whisper.cpp` via the ``pywhispercpp`` library.  The underlying
Whisper models provide multilingual transcription with punctuation and
capitalisation.  The resulting transcript can be edited,
copied to the clipboard and saved automatically to a JSON history file for
later reuse.  The program also supports optional manual language selection
if automatic detection does not yield the desired results.

This code is intended to be run from within a virtual environment on Windows
(see README section at the bottom of this file for setup instructions).  The
application is self‑contained and requires only free and locally available
dependencies (PyQt6, pywhispercpp and ffmpeg).  Since Whisper.cpp is a
CPU‑only implementation by default, no GPU is required, although advanced
users can build Whisper.cpp with CUDA support if desired.

Highlights:

* Drag-and-drop zone and clickable button to select files.
* Automatic detection of audio files; video files are converted to WAV via ffmpeg.
* Uses Whisper models (through ``pywhispercpp``) for multilingual
  transcription with punctuation and capitalization.
* Supports Spanish, English and Chinese out of the box with automatic language
  detection; manual override via a dropdown selector.
* Maintains a simple history of transcriptions in a JSON file stored in the
  application directory.  History entries can be reviewed, copied or deleted.
* Runs transcription in a separate worker thread to keep the GUI responsive.
* Adopts a dark Fusion theme compatible with Windows dark mode.

The code below is extensively commented to facilitate understanding and
modification.  You can extend it with additional features such as exporting to
plain text/SRT, selecting different Whisper model sizes, or integrating
translation support.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import re
import threading
import time
import shutil
import urllib.request
import urllib.error
import importlib
import wave
from urllib.parse import urlparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Callable

"""
IMPORTANT: Since Windows builds of PyTorch can fail to initialise when used
in concert with PyQt, this application no longer
relies on the `openai-whisper`/PyTorch stack for transcription.  Instead it
uses the `pywhispercpp` library, which provides Python bindings to the
`whisper.cpp` C/C++ implementation.  These bindings download pre‑built
Whisper models and perform the transcription in native code, avoiding the
dynamic‑library initialisation issues seen with Torch.  If you wish to use
GPU acceleration or larger models, consult the `pywhispercpp` documentation.

To install the required dependencies on Windows, create a virtual
environment and run:

```
pip install pyqt6
pip install pywhispercpp  # CPU version of Whisper.cpp bindings
# For audio/video file handling, ensure ffmpeg is on your PATH.
```

Once installed, the application will automatically download the default
`small` Whisper model the first time you transcribe audio.  Subsequent
transcriptions reuse the cached model to reduce startup time.
"""

# Import the Whisper.cpp bindings.  These bindings do not depend on PyTorch
# and therefore avoid the DLL initialisation issues that plague the
# `openai-whisper` library on Windows.  If the package is not installed, an
# informative error will be raised when a transcription is attempted.
try:
    from pywhispercpp.model import Model as WhisperCppModel  # type: ignore
except Exception:
    WhisperCppModel = None  # type: ignore

from PyQt6.QtCore import (Qt, QObject, QThread, pyqtSignal, QSize, QTimer, QSettings, QStandardPaths, QUrl)
from PyQt6.QtGui import (QDragEnterEvent, QDropEvent, QPalette, QColor,
                         QFont, QIcon, QGuiApplication, QAction, QActionGroup,
                         QKeySequence, QShortcut, QTextCursor, QTextDocument,
                         QDesktopServices)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGroupBox, QLabel, QVBoxLayout, QPushButton,
    QFileDialog, QTextEdit, QComboBox, QHBoxLayout, QListWidget, QListWidgetItem,
    QMessageBox, QProgressBar, QSplitter, QTabWidget, QToolBar, QStatusBar, QPlainTextEdit,
    QLineEdit, QCheckBox, QToolButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFormLayout
)

# External library for transcription.
#
# This application originally used the ``faster-whisper`` library, however on
# some Windows systems the CTranslate2 backend could crash the Python process
# during model teardown.  We initially switched to the
# ``openai-whisper`` implementation, but later discovered that importing
# PyTorch after PyQt on Windows can trigger a DLL initialisation error
#.  To avoid these issues entirely, we now
# use the ``pywhispercpp`` bindings for ``whisper.cpp``, which depend only on
# a few lightweight libraries and do not require PyTorch.  See the
# documentation at https://github.com/absadiki/pywhispercpp for details.

# -----------------------------------------------------------------------------
# Optional configuration for model download warnings
#
# The original ``faster-whisper`` implementation would download models from the
# HuggingFace Hub, which on Windows could emit warnings when symlinks were
# unsupported.  We preserve the environment variable to silence such warnings
# should any underlying library still rely on HuggingFace caching.  If you have
# your own HuggingFace token, you can set ``HF_TOKEN`` in your environment to
# benefit from higher rate limits.
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

# -----------------------------------------------------------------------------
# PyInstaller --windowed / pythonw safety:
# In no-console mode, sys.stdout and sys.stderr can be None. Some libraries
# (and our own debug prints) expect them to exist and have .flush().
# We replace missing streams with a devnull file so code can safely flush.
# See PyInstaller docs: common issues and pitfalls.
def _ensure_std_streams() -> None:
    try:
        if sys.stdout is None:
            sys.stdout = open(os.devnull, 'w', encoding='utf-8', errors='ignore')
    except Exception:
        pass
    try:
        if sys.stderr is None:
            sys.stderr = open(os.devnull, 'w', encoding='utf-8', errors='ignore')
    except Exception:
        pass




# PyInstaller-friendly resource resolver (icons, etc.)
def resource_path(relative_path: str) -> str:
    """Return absolute path to a resource (works in dev and PyInstaller)."""
    try:
        base_path = sys._MEIPASS  # type: ignore[attr-defined]
    except Exception:
        base_path = os.path.abspath('.')
    return os.path.join(base_path, relative_path)


# Use a module-level cache to store the Whisper.cpp model.  Keeping a
# persistent reference avoids reloading the model for each transcription.
_WHISPER_MODELS: Dict[str, object] = {}  # cache_key -> pywhispercpp.Model


def get_whisper_model(model_size: str = "small", models_dir: Optional[str] = None) -> "WhisperCppModel":
    """Return a cached Whisper.cpp model instance using the pywhispercpp library.

    Args:
        model_size: Model name (e.g. 'small', 'base.en', 'large-v3').
        models_dir: Directory where ggml models are stored/downloaded.

    Returns:
        A loaded ``pywhispercpp.model.Model`` instance.
    """
    global _WHISPER_MODELS, WhisperCppModel
    cache_key = f"{models_dir or ''}::{model_size}"
    if cache_key in _WHISPER_MODELS:
        return _WHISPER_MODELS[cache_key]  # type: ignore[return-value]
    if WhisperCppModel is None:
        raise RuntimeError(
            "The 'pywhispercpp' package is not installed. Please run 'pip install pywhispercpp'."
        )
    _ensure_std_streams()
    try:
        print(f"[DEBUG] Loading Whisper.cpp model size={model_size} models_dir={models_dir}", flush=True)
    except Exception:
        pass
    try:
        model = WhisperCppModel(model=model_size, models_dir=models_dir)
        _WHISPER_MODELS[cache_key] = model
    except Exception as exc:
        raise RuntimeError(f"Failed to load Whisper.cpp model '{model_size}': {exc}")
    return model  # type: ignore[return-value]


###############################################################################
# Utility functions and data structures
###############################################################################

# Directory where transcripts history JSON will be stored.  The file lives in
# the same directory as this script (i.e., the working directory when running
# the program).  If you package the application differently you may wish to
# change this path to a more appropriate location (e.g. AppData).
HISTORY_FILENAME = "transcripts_history.json"


# -----------------------------------------------------------------------------
# App UI language (interface text)
# -----------------------------------------------------------------------------

UI_TEXT: Dict[str, Dict[str, str]] = {
    "en": {
        "tab_files": "Files",
        "tab_links": "Links",
        "tab_history": "History",
        "tab_req": "Requirements",
        "drop_default": "Drop your audio or video files here\n(or click to select)",
        "drop_selected": "Selected: {name}\n(drop another file or click to change)",
        "label_audio_lang": "Audio language:",
        "audio_auto": "Auto detect",
        "lang_en": "English",
        "lang_es": "Spanish",
        "lang_zh": "Chinese",
        "label_model": "Model:",
        "btn_copy": "Copy to clipboard",
        "btn_translate": "Translate",
        "label_translate_to": "Translate to:",
        "menu_models": "Models",
        "menu_settings": "Settings",
        "menu_app_lang": "App language",
        "menu_help": "Help",
        "act_manage_models": "Manage models...",
        "act_open_models_dir": "Open models folder",
        "act_install_engine": "Install/update engine (pywhispercpp)",
        "act_about": "About Voxora",
        "status_ready": "Ready",
        "status_transcribing": "Transcribing",
        "status_downloading_audio": "Downloading audio",
        "status_extracting_audio": "Extracting audio",
        "status_downloading_model": "Downloading model",
        "status_retrying": "Retrying",
        "status_installing_engine": "Installing engine",
        "status_cancelling": "Cancelling",
        "status_cancelling_download": "Cancelling download",
        "msg_busy_title": "Transcription in progress",
        "msg_busy_body": "A transcription is already running. Please wait or cancel.",
        "err_transcription": "Transcription error",
        "toolbar_cancel": "Cancel",
        "toolbar_terminal": "Terminal",
        "label_url": "URL:",
        "btn_transcribe_url": "Transcribe URL",
        "group_url": "Transcribe from URL (YouTube / Bilibili)",
        "label_youtube_cookies": "YouTube cookies:",
        "label_bilibili_cookies": "Bilibili cookies:",
        "btn_load": "Load",
        "btn_remove": "Remove",
        "placeholder_url": "Paste a YouTube or Bilibili link",
        "placeholder_transcript": "The transcription will appear here...",
        "placeholder_terminal": "Logs (optional)...",
        "find_label": "Find:",
        "find_placeholder": "Search in the transcription...",
        "find_prev": "Previous",
        "find_next": "Next",
        "find_close": "Close",
        "find_case": "Aa",
        "btn_copy_selected": "Copy selected",
        "btn_delete_selected": "Delete selected",
        "status_translating": "Translating",
        "err_no_audio_to_translate": "No previous audio available to translate. Transcribe something first.",
        "msg_translate_title": "Translate",
        "msg_translate_body": "This will reprocess the same audio and produce an English translation.",
    },
    "es": {
        "tab_files": "Archivo",
        "tab_links": "Enlaces",
        "tab_history": "Historial",
        "tab_req": "Requisitos",
        "drop_default": "Arrastra aquí tus archivos de audio o video\n(o haz clic para seleccionarlos)",
        "drop_selected": "Seleccionado: {name}\n(arrastra otro archivo o haz clic para cambiar)",
        "label_audio_lang": "Idioma del audio:",
        "audio_auto": "Auto detectar",
        "lang_en": "Inglés",
        "lang_es": "Español",
        "lang_zh": "Chino",
        "label_model": "Modelo:",
        "btn_copy": "Copiar al portapapeles",
        "btn_translate": "Traducir",
        "label_translate_to": "Traducir a:",
        "menu_models": "Modelos",
        "menu_settings": "Ajustes",
        "menu_app_lang": "Idioma de la app",
        "menu_help": "Ayuda",
        "act_manage_models": "Administrar modelos...",
        "act_open_models_dir": "Abrir carpeta de modelos",
        "act_install_engine": "Instalar/actualizar motor (pywhispercpp)",
        "act_about": "Acerca de Voxora",
        "status_ready": "Listo",
        "status_transcribing": "Transcribiendo",
        "status_downloading_audio": "Descargando audio",
        "status_extracting_audio": "Extrayendo audio",
        "status_downloading_model": "Descargando modelo",
        "status_retrying": "Reintentando",
        "status_installing_engine": "Instalando motor",
        "status_cancelling": "Cancelando",
        "status_cancelling_download": "Cancelando descarga",
        "msg_busy_title": "Transcripción en curso",
        "msg_busy_body": "Ya hay una transcripción en progreso. Por favor espera o cancela.",
        "err_transcription": "Error de transcripción",
        "toolbar_cancel": "Cancelar",
        "toolbar_terminal": "Terminal",
        "label_url": "URL:",
        "btn_transcribe_url": "Transcribir URL",
        "group_url": "Transcribir desde URL (YouTube / Bilibili)",
        "label_youtube_cookies": "Cookies de YouTube:",
        "label_bilibili_cookies": "Cookies de Bilibili:",
        "btn_load": "Cargar",
        "btn_remove": "Quitar",
        "placeholder_url": "Pega un enlace de YouTube o Bilibili",
        "placeholder_transcript": "La transcripción aparecerá aquí...",
        "placeholder_terminal": "Logs (opcional)...",
        "find_label": "Buscar:",
        "find_placeholder": "Buscar en la transcripción...",
        "find_prev": "Anterior",
        "find_next": "Siguiente",
        "find_close": "Cerrar",
        "find_case": "Aa",
        "btn_copy_selected": "Copiar seleccionado",
        "btn_delete_selected": "Eliminar seleccionado",
        "status_translating": "Traduciendo",
        "err_no_audio_to_translate": "No hay un audio previo disponible para traducir. Transcribe algo primero.",
        "msg_translate_title": "Traducir",
        "msg_translate_body": "Esto volverá a procesar el mismo audio y generará una traducción al inglés.",
    },
    "zh": {
        "tab_files": "文件",
        "tab_links": "链接",
        "tab_history": "历史",
        "tab_req": "配置建议",
        "drop_default": "将音频或视频文件拖到这里\n(或点击选择)",
        "drop_selected": "已选择: {name}\n(拖入其他文件或点击更换)",
        "label_audio_lang": "音频语言:",
        "audio_auto": "自动识别",
        "lang_en": "英语",
        "lang_es": "西班牙语",
        "lang_zh": "中文",
        "label_model": "模型:",
        "btn_copy": "复制到剪贴板",
        "btn_translate": "翻译",
        "label_translate_to": "翻译为:",
        "menu_models": "模型",
        "menu_settings": "设置",
        "menu_app_lang": "界面语言",
        "menu_help": "帮助",
        "act_manage_models": "管理模型...",
        "act_open_models_dir": "打开模型文件夹",
        "act_install_engine": "安装/更新引擎 (pywhispercpp)",
        "act_about": "关于 Voxora",
        "status_ready": "就绪",
        "status_transcribing": "正在转写",
        "status_downloading_audio": "正在下载音频",
        "status_extracting_audio": "正在提取音频",
        "status_downloading_model": "正在下载模型",
        "status_retrying": "正在重试",
        "status_installing_engine": "正在安装引擎",
        "status_cancelling": "正在取消",
        "status_cancelling_download": "正在取消下载",
        "msg_busy_title": "正在转写",
        "msg_busy_body": "已有转写任务在运行，请等待或取消。",
        "err_transcription": "转写错误",
        "toolbar_cancel": "取消",
        "toolbar_terminal": "终端",
        "label_url": "URL:",
        "btn_transcribe_url": "转写链接",
        "group_url": "从链接转写 (YouTube / Bilibili)",
        "label_youtube_cookies": "YouTube cookies:",
        "label_bilibili_cookies": "Bilibili cookies:",
        "btn_load": "加载",
        "btn_remove": "移除",
        "placeholder_url": "粘贴 YouTube 或 Bilibili 链接",
        "placeholder_transcript": "转写结果会显示在这里...",
        "placeholder_terminal": "日志 (可选)...",
        "find_label": "查找:",
        "find_placeholder": "在转写中搜索...",
        "find_prev": "上一个",
        "find_next": "下一个",
        "find_close": "关闭",
        "find_case": "Aa",
        "btn_copy_selected": "复制所选",
        "btn_delete_selected": "删除所选",
        "status_translating": "正在翻译",
        "err_no_audio_to_translate": "没有可用于翻译的上一次音频，请先转写一次。",
        "msg_translate_title": "翻译",
        "msg_translate_body": "将重新处理同一段音频并输出英文翻译。",
    },
}


def ui_tr(lang: str, key: str, **fmt) -> str:
    base = UI_TEXT.get(lang) or UI_TEXT["en"]
    s = base.get(key) or UI_TEXT["en"].get(key) or key
    try:
        return s.format(**fmt)
    except Exception:
        return s

# Whisper.cpp / pywhispercpp model selector
#
# Model names follow the conventions used by whisper.cpp model files (ggml-*.bin).
# Common multilingual models: tiny, base, small, medium, large-v1/v2/v3, large-v3-turbo.
# English-optimized variants: *.en (e.g. tiny.en, base.en, small.en, medium.en).
# Some quantized variants (q5_1 / q8_0 / q5_0) are also available.
DEFAULT_MODEL_SIZE = "small"

# UI list: (label, model_key_for_pywhispercpp)
MODEL_CHOICES = [
    ("tiny (rápido)", "tiny"),
    ("tiny.en (inglés, rápido)", "tiny.en"),
    ("base (rápido)", "base"),
    ("base.en (inglés)", "base.en"),
    ("small (equilibrado, recomendado)", "small"),
    ("small.en (inglés)", "small.en"),
    ("medium (alta calidad)", "medium"),
    ("medium.en (inglés)", "medium.en"),
    ("large-v1 (máxima calidad)", "large-v1"),
    ("large-v2 (máxima calidad)", "large-v2"),
    ("large-v3 (máxima calidad)", "large-v3"),
    ("large-v3-turbo (rápido + preciso)", "large-v3-turbo"),
    ("large-v3-turbo-q5_0 (turbo, menos RAM)", "large-v3-turbo-q5_0"),
    ("large-v3-turbo-q8_0 (turbo, más preciso)", "large-v3-turbo-q8_0"),
]


# Try to align the UI list with the models supported by the installed pywhispercpp build.
def _get_available_models_from_pywhispercpp() -> Optional[List[str]]:
    try:
        from pywhispercpp.constants import AVAILABLE_MODELS  # type: ignore
        return list(AVAILABLE_MODELS)
    except Exception:
        return None

_AVAILABLE_MODELS = _get_available_models_from_pywhispercpp()
if _AVAILABLE_MODELS:
    _avail = set(_AVAILABLE_MODELS)
    MODEL_CHOICES = [(lbl, key) for (lbl, key) in MODEL_CHOICES if key in _avail]


# -----------------------------------------------------------------------------
# Hardware guidance per model (mínimo / recomendado)
#
# Nota: Estos valores son orientativos. VRAM viene del repositorio oficial
# openai/whisper, y el uso de memoria en CPU se inspira en la tabla de
# whisper.cpp (model disk/mem). La recomendación de RAM del sistema agrega
# margen para Python, buffers y el sistema operativo.
# -----------------------------------------------------------------------------

# Disk sizes (aprox) en GiB para estimar memoria de modelos que no están en la
# tabla resumida de whisper.cpp (p. ej. large-v3-turbo y sus cuantizados).
# Fuentes: lista de archivos en Hugging Face ggerganov/whisper.cpp.
_MODEL_DISK_GIB_APPROX: dict[str, float] = {
    "large-v3": 2.9,
    "large-v3-turbo": 1.5,
    "large-v3-turbo-q5_0": 0.547,
    "large-v3-turbo-q8_0": 0.874,
}

# Factor aproximado para pasar de disco->memoria en whisper.cpp.
# Derivado de la fila: large 2.9 GiB disk -> ~3.9 GB mem.
_DISK_TO_MEM_FACTOR = 3.9 / 2.9


def _estimate_model_mem_gb(model_key: str) -> float:
    """Estimación de memoria del modelo en CPU (GB) para whisper.cpp."""
    base = {
        "tiny": 0.273,
        "tiny.en": 0.273,
        "base": 0.388,
        "base.en": 0.388,
        "small": 0.852,
        "small.en": 0.852,
        "medium": 2.1,
        "medium.en": 2.1,
        "large": 3.9,
        "large-v1": 3.9,
        "large-v2": 3.9,
        "large-v3": 3.9,
    }
    if model_key in base:
        return base[model_key]
    if model_key in _MODEL_DISK_GIB_APPROX:
        return _MODEL_DISK_GIB_APPROX[model_key] * _DISK_TO_MEM_FACTOR
    return 1.0


def _round_gb(x: float) -> str:
    v = round(x * 2) / 2
    if abs(v - int(v)) < 1e-9:
        return f"{int(v)} GB"
    return f"{v:.1f} GB"


def _suggest_system_ram(model_mem_gb: float) -> tuple[str, str]:
    """Regresa (mínimo, recomendado) de RAM del sistema."""
    ram_min = max(2.0, model_mem_gb + 1.5)
    ram_rec = max(ram_min + 2.0, model_mem_gb * 2 + 2.0)

    def snap(v: float) -> float:
        steps = [2, 4, 6, 8, 12, 16, 24, 32]
        for s in steps:
            if v <= s:
                return float(s)
        return float(steps[-1])

    return _round_gb(snap(ram_min)), _round_gb(snap(ram_rec))


def _gpu_vram_requirements_gb(model_key: str) -> tuple[str, str]:
    """VRAM (mínimo, recomendado) si usas backend GPU tipo PyTorch/Transformers."""
    if model_key in {"tiny", "tiny.en", "base", "base.en"}:
        vmin = 1.0
    elif model_key in {"small", "small.en"}:
        vmin = 2.0
    elif model_key in {"medium", "medium.en"}:
        vmin = 5.0
    elif model_key in {"large", "large-v1", "large-v2", "large-v3"}:
        vmin = 10.0
    elif model_key in {"large-v3-turbo", "large-v3-turbo-q5_0", "large-v3-turbo-q8_0", "turbo"}:
        vmin = 6.0
    else:
        vmin = 2.0

    vrec = vmin + (2.0 if vmin <= 6.0 else 4.0)
    return _round_gb(vmin), _round_gb(vrec)


def _cpu_guidance(model_key: str) -> tuple[str, str]:
    """CPU (mínimo, recomendado) orientativo."""
    if model_key in {"tiny", "tiny.en", "base", "base.en"}:
        return "2 núcleos", "4 núcleos"
    if model_key in {"small", "small.en"}:
        return "4 núcleos", "6 a 8 núcleos"
    if model_key in {"medium", "medium.en"}:
        return "6 núcleos", "8 a 12 núcleos"
    if model_key in {"large", "large-v1", "large-v2", "large-v3"}:
        return "8 núcleos", "12+ núcleos"
    if model_key in {"large-v3-turbo", "large-v3-turbo-q5_0", "large-v3-turbo-q8_0", "turbo"}:
        return "6 núcleos", "8 a 12 núcleos"
    return "4 núcleos", "6 a 8 núcleos"


def build_model_hw_specs() -> dict[str, dict[str, str]]:
    """Construye el mapa de specs para los modelos del selector."""
    specs: dict[str, dict[str, str]] = {}
    for _label, key in MODEL_CHOICES:
        mem = _estimate_model_mem_gb(key)
        ram_min, ram_rec = _suggest_system_ram(mem)
        vmin, vrec = _gpu_vram_requirements_gb(key)
        cmin, crec = _cpu_guidance(key)
        notes = []
        if key.endswith("-q5_0") or key.endswith("-q5_1") or key.endswith("-q8_0"):
            notes.append("cuantizado: menos RAM y disco")
        if key.endswith(".en"):
            notes.append("optimizado para inglés")
        if key.startswith("large-v3-turbo") or key == "turbo":
            notes.append("turbo: optimización de large-v3")
        specs[key] = {
            "cpu": f"{cmin} / {crec}",
            "ram": f"{ram_min} / {ram_rec}",
            "vram": f"{vmin} / {vrec}",
            "notes": "; ".join(notes) if notes else "",
        }
    return specs


MODEL_HW_SPECS = build_model_hw_specs()



def is_audio_file(path: Path) -> bool:
    """Return True if the given path has an audio file extension.

    Accepts common audio formats used for voice messages.  Video formats are
    handled separately.
    """
    audio_exts = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.opus', '.aac', '.caf', '.webm'}
    return path.suffix.lower() in audio_exts


def is_video_file(path: Path) -> bool:
    """Return True if the given path has a typical video file extension.
    """
    video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    return path.suffix.lower() in video_exts


@dataclass
class TranscriptEntry:
    """Represents a single transcript entry in the history."""
    filename: str
    language: str
    timestamp: float
    transcript: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def load_history(history_path: Path) -> List[TranscriptEntry]:
    """Load transcript history from JSON file.

    If the file does not exist or is malformed, an empty list is returned.
    """
    entries: List[TranscriptEntry] = []
    if history_path.exists():
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                entries.append(TranscriptEntry(**item))
        except Exception as exc:
            print(f"Warning: failed to load history: {exc}")
    return entries


def save_history(history_path: Path, entries: List[TranscriptEntry]) -> None:
    """Write transcript history to disk.

    The file will be overwritten with the contents of ``entries``.
    """
    try:
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump([entry.to_dict() for entry in entries], f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"Error saving history: {exc}")


def smart_join_segments(segments: List[object]) -> str:
    """Join transcription segments while preserving sensible spacing.

    Whisper/whisper.cpp often returns segment texts with inconsistent leading
    whitespace. Joining them with an empty separator can produce words stuck
    together across segment boundaries (e.g. "graciasera").

    This function inserts a space between segments when needed, without adding
    spaces before punctuation.
    """
    no_space_before = set(",.;:!?)]}%”’")
    out = ""
    for seg in segments:
        t = getattr(seg, "text", "")
        if not t:
            continue
        t = t.lstrip()
        if not out:
            out = t
            continue
        if out and not out[-1].isspace() and t and not t[0].isspace():
            if t[0] not in no_space_before:
                out += " "
        out += t
    out = re.sub(r"[ \t]+", " ", out)
    return out.strip()

def _normalize_text_for_dedupe(s: str) -> str:
    """Normalize text to compare for repetition / loop detection."""
    s = (s or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    # Drop most punctuation so minor differences don't defeat dedupe.
    s = re.sub(r"[\"'“”‘’.,;:!?()\[\]{}<>¡¿]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def dedupe_repeated_segments(segments: List[object]) -> List[object]:
    """Drop consecutive duplicate segments produced by rare Whisper repetition loops."""
    out: List[object] = []
    prev_norm: Optional[str] = None
    for seg in segments:
        t = getattr(seg, "text", "") or ""
        norm = _normalize_text_for_dedupe(t)
        if prev_norm is not None and norm and norm == prev_norm:
            # Skip consecutive duplicates.
            continue
        out.append(seg)
        prev_norm = norm
    return out


def collapse_repetitions_in_text(
    text: str,
    *,
    max_phrase_len: int = 12,
    min_repeats: int = 4,
    min_single_token_repeats: int = 8,
) -> str:
    """Collapse pathological repeated phrases like 'texto de los mensajes y' x 100.

    This is a pragmatic post-processing step for the known Whisper failure mode
    where the decoder gets stuck repeating the same short phrase many times.
    We only collapse *consecutive* repetitions and require multiple repeats to
    reduce the chance of deleting legitimate emphasis.
    """
    tokens = (text or "").split()
    if len(tokens) < 2:
        return (text or "").strip()

    out: List[str] = []
    i = 0
    n = len(tokens)

    while i < n:
        collapsed = False

        # 1) Collapse repeated multi-token phrases, try longer first.
        max_l = min(max_phrase_len, n - i)
        for L in range(max_l, 1, -1):
            if i + L * min_repeats > n:
                continue
            phrase = tokens[i : i + L]
            r = 1
            while i + (r + 1) * L <= n and tokens[i + r * L : i + (r + 1) * L] == phrase:
                r += 1
            if r >= min_repeats:
                out.extend(phrase)
                i += r * L
                collapsed = True
                break

        if collapsed:
            continue

        # 2) Collapse repeated single tokens (e.g. 'y y y y ...') with a higher threshold.
        tok = tokens[i]
        r = 1
        while i + r < n and tokens[i + r] == tok:
            r += 1
        if r >= min_single_token_repeats:
            out.append(tok)
            i += r
            continue

        out.append(tok)
        i += 1

    result = " ".join(out)

    # Clean up spaces around punctuation.
    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    result = re.sub(r"\s+([)\]\}”’])", r"\1", result)
    result = re.sub(r"([(\[\{“‘])\s+", r"\1", result)
    result = re.sub(r"[ \t]+", " ", result)
    return result.strip()


def _looks_like_repeat_loop(segments: List[object]) -> bool:
    """Heuristic detector for the 'stuck repeating the same sentence' failure mode."""
    prev = None
    run = 0
    max_run = 0
    for seg in segments:
        norm = _normalize_text_for_dedupe(getattr(seg, "text", "") or "")
        if not norm:
            continue
        if prev == norm:
            run += 1
        else:
            prev = norm
            run = 1
        if run > max_run:
            max_run = run
        if max_run >= 6:
            return True

    # Also catch repetitions inside the final text (common in this bug).
    joined = smart_join_segments(segments)
    collapsed = collapse_repetitions_in_text(joined)
    # If we would delete a huge portion, it likely was a loop.
    if joined and len(collapsed) < int(len(joined) * 0.70):
        return True

    return False




###############################################################################
# Worker thread for running heavy transcription without freezing the UI
###############################################################################

class TranscriptionWorker(QObject):
    """Runs Whisper transcription in a background thread.

    Signals:
        progress(int): emitted periodically with percentage progress (0–100).
        finished(TranscriptEntry): emitted when transcription finishes.
        error(str): emitted if an error occurs during transcription.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)  # TranscriptEntry
    error = pyqtSignal(str)
    log_line = pyqtSignal(str)

    def __init__(self, paths: List[Path], language: str, model_size: str, models_dir: Optional[str] = None, translate_to_en: bool = False, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.paths = paths
        lang = (language or '').lower().strip()
        self.language = None if (not lang or lang == 'auto') else lang
        self.model_size = (model_size or DEFAULT_MODEL_SIZE).strip()
        self.models_dir = models_dir
        self.translate_to_en = bool(translate_to_en)
        self._stop_requested = False

    def request_stop(self) -> None:
        """Request cancellation of the transcription.

        The worker will attempt to stop at a safe point.
        """
        self._stop_requested = True

    def run(self) -> None:
        """Main entry point for the worker thread.

        Notes on progress reporting:
        - ``pywhispercpp`` returns the list of segments only after finishing the
          transcription call, so segment-based progress updates don't move the
          progress bar during inference.
        - whisper.cpp can print progress updates (e.g. "Progress: 17%") to stderr.
          We temporarily redirect whisper.cpp logs to a pipe and parse those
          messages to drive the GUI progress bar in real time.
        """
        import subprocess
        _ensure_std_streams()
        try:
            from pywhispercpp.utils import redirect_stderr as pw_redirect_stderr  # type: ignore
        except Exception:
            pw_redirect_stderr = None

        def log_line(msg: str) -> None:
            # Mirror logs to the in-app terminal.
            self.log_line.emit(msg)
            # In PyInstaller --windowed, stdout may be missing; keep this best-effort.
            _ensure_std_streams()
            try:
                print(msg, flush=True)
            except Exception:
                pass

        def _wav_is_16k_mono_pcm(p: Path) -> bool:
            """Return True if `p` is a 16 kHz mono PCM WAV.

            Prefer ffprobe (robust with WAVE_FORMAT_EXTENSIBLE). Fallback to Python's
            wave module when ffprobe is unavailable.
            """
            if p.suffix.lower() != ".wav":
                return False

            # Prefer ffprobe when available (handles more WAV variants than `wave`).
            try:
                if shutil.which("ffprobe"):
                    cmd = [
                        "ffprobe",
                        "-v", "error",
                        "-select_streams", "a:0",
                        "-show_entries", "stream=sample_rate,channels,codec_name",
                        "-of", "json",
                        str(p),
                    ]
                    creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    r = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="ignore",
                        creationflags=creation_flags,
                    )
                    if r.returncode == 0 and r.stdout:
                        try:
                            data = json.loads(r.stdout)
                            streams = data.get("streams") or []
                            if streams:
                                s0 = streams[0] or {}
                                sr = int(s0.get("sample_rate") or 0)
                                ch = int(s0.get("channels") or 0)
                                codec = str(s0.get("codec_name") or "").lower()
                                if sr == 16000 and ch == 1 and (codec.startswith("pcm_") or codec == "pcm_s16le"):
                                    return True
                        except Exception:
                            pass
            except Exception:
                pass

            # Fallback: Python wave (may fail on WAVE_FORMAT_EXTENSIBLE in Python < 3.12).
            try:
                with wave.open(str(p), "rb") as wf:
                    sr = wf.getframerate()
                    ch = wf.getnchannels()
                return sr == 16000 and ch == 1
            except Exception:
                return False

        try:
            model_size = self.model_size or DEFAULT_MODEL_SIZE
            model = get_whisper_model(model_size=model_size, models_dir=self.models_dir)

            total_files = len(self.paths)
            processed_files = 0

            log_line(
                f"[DEBUG] Starting transcription of {total_files} file(s) with language={self.language} translate_to_en={self.translate_to_en}"
            )

            for path in self.paths:
                if self._stop_requested:
                    log_line("[DEBUG] Stop requested, aborting transcription")
                    return

                log_line(f"[DEBUG] Processing file: {path}")

                # Determine if path is audio or video; if video, extract audio
                audio_path: Optional[Path] = None
                created_temp_audio = False

                if is_audio_file(path):
                    # whisper.cpp (via pywhispercpp) expects 16 kHz mono PCM WAV.
                    # Convert any audio input (e.g. OGG/MP3/M4A/WAV 48k) to a temp 16 kHz WAV.
                    # NOTE: For WAV inputs we prefer ffprobe-based detection to avoid false negatives
                    # (the Python `wave` module can fail on some valid WAV variants).
                    needs_convert = True
                    if path.suffix.lower() == ".wav":
                        needs_convert = not _wav_is_16k_mono_pcm(path)

                    if not needs_convert:
                        audio_path = path
                        log_line(f"[DEBUG] File is audio (already 16k WAV): {audio_path}")
                    else:
                        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".wav")
                        os.close(tmp_fd)
                        audio_path = Path(tmp_name)
                        created_temp_audio = True

                        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                        ffmpeg_cmd = [
                            "ffmpeg",
                            "-y",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-i",
                            str(path),
                            "-vn",
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            "-acodec",
                            "pcm_s16le",
                            str(audio_path),
                        ]
                        log_line(f"[DEBUG] Converting audio with command: {' '.join(ffmpeg_cmd)}")

                        last_out = ""
                        ok = False
                        for attempt in range(1, 4):
                            try:
                                r = subprocess.run(
                                    ffmpeg_cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True,
                                    encoding="utf-8",
                                    errors="ignore",
                                    creationflags=creation_flags,
                                )
                                last_out = (r.stdout or "").strip()
                                if r.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 44:
                                    ok = True
                                    break
                                time.sleep(0.25)
                            except Exception as exc:
                                last_out = str(exc)
                                time.sleep(0.25)

                        if not ok:
                            tail = "\n".join((last_out.splitlines()[-20:] if last_out else []))
                            error_msg = (
                                "Error converting audio with ffmpeg.\n"
                                f"Input: {path}\nOutput: {audio_path}\n\n"
                                + (f"ffmpeg output:\n{tail}" if tail else "")
                            )
                            log_line(f"[ERROR] {error_msg}")
                            self.error.emit(error_msg)
                            return

                        log_line(f"[DEBUG] Audio converted to: {audio_path}")

                elif is_video_file(path):
                    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".wav")
                    os.close(tmp_fd)
                    audio_path = Path(tmp_name)
                    created_temp_audio = True

                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(path),
                        "-vn",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-f",
                        "wav",
                        str(audio_path),
                    ]
                    log_line(f"[DEBUG] Extracting audio with command: {' '.join(cmd)}")

                    try:
                        subprocess.run(
                            cmd,
                            check=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT,
                        )
                        log_line(f"[DEBUG] Audio extracted to: {audio_path}")
                    except Exception as exc:
                        error_msg = f"Error extracting audio with ffmpeg: {exc}"
                        log_line(f"[ERROR] {error_msg}")
                        self.error.emit(error_msg)
                        return
                else:
                    log_line(f"[WARN] Unsupported file type: {path}")
                    continue

                if audio_path is None:
                    continue

                # Start reading whisper.cpp progress from stderr (redirected to a pipe)
                # so we can update the progress bar in real time.
                last_overall_pct = -1
                stop_evt = threading.Event()
                pipe_r, pipe_w = os.pipe()
                try:
                    os.set_blocking(pipe_r, False)
                except Exception:
                    # Older Python versions on some platforms; non-blocking is optional.
                    pass
                wfile = os.fdopen(pipe_w, "wb", buffering=0)

                progress_re = re.compile(rb"Progress:\s*(\d+)%")
                attempt_no = 1

                def reader_loop(file_index_done: int) -> None:
                    nonlocal last_overall_pct
                    buf = b""
                    while not stop_evt.is_set():
                        try:
                            chunk = os.read(pipe_r, 4096)
                            if not chunk:
                                break
                            buf += chunk
                            buf = buf.replace(b"\r", b"\n")
                            while b"\n" in buf:
                                raw, buf = buf.split(b"\n", 1)
                                raw_stripped = raw.strip()
                                if not raw_stripped:
                                    continue
                                try:
                                    line = raw_stripped.decode("utf-8", errors="replace")
                                except Exception:
                                    line = str(raw_stripped)
                                # Make progress logs less confusing when we retry.
                                display_line = line
                                if line.startswith("Progress:"):
                                    display_line = f"Intento {attempt_no}: {line}"
                                self.log_line.emit(display_line)

                                m = progress_re.search(raw_stripped)
                                if m:
                                    try:
                                        file_pct = int(m.group(1))
                                    except Exception:
                                        continue
                                    # Overall progress across multiple files
                                    denom = max(total_files, 1)
                                    overall = int(
                                        100
                                        * ((file_index_done + (file_pct / 100.0)) / denom)
                                    )
                                    if overall > 99:
                                        overall = 99
                                    if overall != last_overall_pct:
                                        last_overall_pct = overall
                                        self.progress.emit(overall)
                        except BlockingIOError:
                            time.sleep(0.05)
                        except Exception:
                            break

                t_reader = threading.Thread(
                    target=reader_loop, args=(processed_files,), daemon=True
                )
                t_reader.start()

                try:
                    # Best-effort: ask pywhispercpp to redirect whisper.cpp logs (if supported),
                    # but also wrap the call in an explicit stderr redirection context so we can
                    # parse progress lines in real time.
                    try:
                        setattr(model, "redirect_whispercpp_logs_to", wfile)
                    except Exception:
                        pass

                    language_param = None if self.language is None else self.language

                    log_line(f"[DEBUG] Starting transcription for {audio_path}")

                    def _transcribe_once(extra_params: Dict[str, object]) -> List[object]:
                        params: Dict[str, object] = {
                            "language": language_param,
                            "print_progress": True,
                            "print_realtime": False,
                        }
                        if self.translate_to_en:
                            params["translate"] = True
                        params.update(extra_params)

                        if pw_redirect_stderr is not None and sys.stderr is not None:
                            with pw_redirect_stderr(wfile):
                                return model.transcribe(str(audio_path), **params)
                        return model.transcribe(str(audio_path), **params)

                    # First pass (best accuracy, keeps context between windows by default).
                    segments = _transcribe_once({})

                    # Known Whisper failure mode: can get stuck repeating the same short phrase.
                    # If we detect it, retry with safer decoding settings.
                    if _looks_like_repeat_loop(segments):
                        log_line("[WARN] Detected repetition loop; retrying with no_context=True and temperature=0.2")
                        attempt_no = 2
                        log_line("[INFO] Reintento: iniciando intento 2 (el progreso se reinicia)")
                        try:
                            # Reset to baseline for this file so the progress bar doesn't look 'stuck at 99%'.
                            base = int(100 * (processed_files / max(total_files, 1)))
                            if base < 0:
                                base = 0
                            if base > 99:
                                base = 99
                            self.progress.emit(base)

                            segments = _transcribe_once({"no_context": True, "temperature": 0.2, "beam_size": 5})
                        except Exception as _exc2:
                            log_line(f"[WARN] Retry with beam_size failed ({_exc2}); retrying with no_context=True only")
                            segments = _transcribe_once({"no_context": True, "temperature": 0.2})

                    log_line(f"[DEBUG] Transcription completed for {audio_path}")

                except Exception as exc:
                    error_msg = f"Error during transcription: {exc}"
                    log_line(f"[ERROR] {error_msg}")
                    self.error.emit(error_msg)
                    return

                finally:
                    stop_evt.set()
                    try:
                        wfile.close()
                    except Exception:
                        pass
                    try:
                        t_reader.join(timeout=1.0)
                    except Exception:
                        pass
                    try:
                        os.close(pipe_r)
                    except Exception:
                        pass
                    # Reset redirection so future unrelated stderr prints behave normally.
                    try:
                        setattr(model, "redirect_whispercpp_logs_to", False)
                    except Exception:
                        pass

                # Detect language if auto was selected (do this before removing temp audio)
                detected_lang = self.language or "auto"
                try:
                    if language_param is None:
                        (lang_pred, _prob), _all_probs = model.auto_detect_language(
                            str(audio_path)
                        )
                        detected_lang = lang_pred
                    else:
                        detected_lang = language_param
                except Exception:
                    detected_lang = language_param or "unknown"

                # Build transcript text:
                # 1) Fix missing spaces between segments.
                # 2) Drop consecutive duplicate segments (rare Whisper repetition loop).
                # 3) Collapse pathological repeated phrases inside the text.
                segments = dedupe_repeated_segments(segments)
                transcript_text = smart_join_segments(segments)
                cleaned_text = collapse_repetitions_in_text(transcript_text)
                if cleaned_text != transcript_text:
                    log_line("[WARN] Collapsed repeated phrases in transcript output")
                    transcript_text = cleaned_text
                log_line(
                    f"[DEBUG] Transcript for {path.name} (length {len(transcript_text)} chars)"
                )

                # Remove temp file if created
                if created_temp_audio:
                    try:
                        audio_path.unlink(missing_ok=True)
                        log_line(f"[DEBUG] Removed temporary audio file: {audio_path}")
                    except Exception as exc:
                        log_line(f"[WARN] Failed to remove temporary file: {exc}")

                entry = TranscriptEntry(
                    filename=str(path.name),
                    language=detected_lang,
                    timestamp=time.time(),
                    transcript=transcript_text,
                )

                processed_files += 1
                # Make sure we reach 100% only when the last file is completed
                if total_files > 0:
                    self.progress.emit(min(int(100 * processed_files / total_files), 100))

                self.finished.emit(entry)
                log_line(f"[DEBUG] Finished processing file {path.name}")

        except Exception as exc:
            error_msg = f"Unhandled error in transcription thread: {exc}"
            print(f"[ERROR] {error_msg}", flush=True)
            self.error.emit(error_msg)




###############################################################################
# Dependency installation (pywhispercpp)
###############################################################################

class DependencyInstallWorker(QObject):
    """Instala dependencias Python (principalmente pywhispercpp) usando pip en segundo plano."""

    log_line = pyqtSignal(str)
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, package_spec: str = "pywhispercpp") -> None:
        super().__init__()
        self.package_spec = package_spec
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        import subprocess
        _ensure_std_streams()

        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", self.package_spec]
        self.log_line.emit(f"[DEBUG] Install cmd: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                creationflags=creation_flags,
            )
            assert proc.stdout is not None
            tail: List[str] = []
            for line in proc.stdout:
                if self._stop_requested:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    self.finished.emit(False, "Instalación cancelada.")
                    return
                msg = (line or "").rstrip()
                if msg:
                    self.log_line.emit(msg)
                    tail.append(msg)
                    tail = tail[-80:]
            rc = proc.wait()
            if rc == 0:
                self.finished.emit(True, "Dependencia instalada correctamente.")
            else:
                self.finished.emit(False, "pip terminó con código %s.\n\nÚltimas líneas:\n%s" % (rc, "\n".join(tail[-20:])))
        except Exception as exc:
            self.finished.emit(False, f"Error ejecutando pip: {exc}")

###############################################################################
# Custom widgets
###############################################################################


class UrlAudioDownloadWorker(QObject):
    """Download audio from a YouTube/Bilibili URL using yt-dlp, into a temp WAV.

    The worker emits progress updates when it can parse a percentage from yt-dlp output.
    It also emits log lines so the GUI can show what is happening.

    It uses `--extract-audio` and `--audio-format wav`, which requires ffmpeg.
    """

    progress = pyqtSignal(int)       # 0-100, or -1 when unknown/indeterminate
    log_line = pyqtSignal(str)
    finished = pyqtSignal(str)       # absolute path to downloaded wav
    error = pyqtSignal(str)

    def __init__(self, url: str, platform: str, cookies_path: str, temp_dir: str) -> None:
        super().__init__()
        self.url = url
        self.platform = platform
        self.cookies_path = cookies_path or ""
        self.temp_dir = temp_dir
        self._stop_requested = False


    def request_stop(self) -> None:
        self._stop_requested = True


    def run(self) -> None:
        import subprocess
        _ensure_std_streams()
        import shutil
        from typing import Tuple

        def _detect_js_runtimes() -> List[Tuple[str, str]]:
            """Detect supported JS runtimes for yt-dlp EJS (YouTube)."""
            candidates: List[Tuple[str, List[str]]] = [
                ("deno", ["deno"]),
                ("node", ["node", "nodejs"]),
                ("bun", ["bun"]),
                ("quickjs", ["qjs", "quickjs"]),
            ]
            found: List[Tuple[str, str]] = []
            for rt, exes in candidates:
                for exe in exes:
                    p = shutil.which(exe)
                    if p:
                        found.append((rt, p))
                        break
            return found

        def _build_cmd(extractor_args: Optional[str] = None, include_remote_components: bool = True) -> List[str]:
            out_tpl = str(Path(self.temp_dir) / "audio.%(ext)s")
            cmd: List[str] = [
                "yt-dlp",
                "--no-playlist",
                "--newline",
                "--extract-audio",
                "--audio-format", "wav",
                "--audio-quality", "0",
            ]

            # YouTube: enable EJS + JS runtime if available.
            if self.platform == "youtube":
                runtimes = _detect_js_runtimes()
                if runtimes:
                    # Enable and/or point yt-dlp to the runtime(s).
                    # Deno is enabled by default, but passing an explicit path helps on Windows.
                    for rt, path in runtimes:
                        cmd.extend(["--js-runtimes", f"{rt}:{path}"])

                    # Ensure EJS challenge solver scripts are available.
                    # Prefer npm auto-download when deno/bun are present; otherwise use GitHub.
                    if include_remote_components:
                        if any(rt in ("deno", "bun") for rt, _ in runtimes):
                            cmd.extend(["--remote-components", "ejs:npm"])
                        else:
                            cmd.extend(["--remote-components", "ejs:github"])
                else:
                    self.log_line.emit(
                        "[WARN] No se detectó un runtime JS (deno/node/bun/quickjs). "
                        "YouTube puede fallar sin esto."
                    )

                # Workaround for YouTube 403/SABR issues: prefer non-iOS clients
                # (we still allow overriding via retries).
                if extractor_args:
                    cmd.extend(["--extractor-args", extractor_args])

            cmd.extend(["-o", out_tpl, self.url])

            if self.cookies_path:
                cmd.extend(["--cookies", self.cookies_path])

            return cmd

        def _run_once(cmd: List[str], label: str) -> Tuple[int, bool, bool, List[str]]:
            """Run yt-dlp once, streaming logs and progress.

            Returns: (returncode, saw_percent, saw_js_runtime_warning, tail_lines)
            """
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

            self.log_line.emit(f"[DEBUG] {label} cmd: {' '.join(cmd)}")

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    creationflags=creation_flags,
                    bufsize=1,
                    universal_newlines=True,
                )
            except Exception as exc:
                self.error.emit(f"No se pudo iniciar yt-dlp: {exc}")
                return 1, False, False, [str(exc)]

            percent_re = re.compile(r"\[download\]\s+(\d{1,3}(?:\.\d+)?)%")
            saw_percent = False
            saw_js_runtime_warning = False
            tail: List[str] = []

            try:
                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    if self._stop_requested:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        self.error.emit("Descarga cancelada.")
                        return 1, saw_percent, saw_js_runtime_warning, tail[-30:]

                    line = (raw_line or "").strip()
                    if not line:
                        continue

                    tail.append(line)
                    if len(tail) > 200:
                        tail = tail[-200:]

                    # Detect common warnings for better UX
                    if "No supported JavaScript runtime could be found" in line:
                        saw_js_runtime_warning = True

                    self.log_line.emit(line)

                    m = percent_re.search(line)
                    if m:
                        saw_percent = True
                        try:
                            pct = float(m.group(1))
                            pct_int = max(0, min(100, int(round(pct))))
                            self.progress.emit(pct_int)
                        except Exception:
                            pass

                proc.wait()
            except Exception as exc:
                try:
                    proc.kill()
                except Exception:
                    pass
                self.error.emit(f"Error leyendo salida de yt-dlp: {exc}")
                return 1, saw_percent, saw_js_runtime_warning, tail[-30:]

            return proc.returncode or 0, saw_percent, saw_js_runtime_warning, tail[-30:]

        # Attempt ladder for YouTube.
        attempts: List[Tuple[str, Optional[str]]] = [
            ("Intento 1", "youtube:player_client=web_embedded,web,tv"),
            ("Intento 2", "youtube:player_client=tv_embedded,web_embedded,web"),
            ("Intento 3", "youtube:player_client=android,web"),
        ]

        if self.platform != "youtube":
            attempts = [("Intento 1", None)]

        # Some yt-dlp builds may not support --remote-components; retry once without it if needed.
        tried_without_remote_components = False
        last_tail: List[str] = []
        saw_percent_any = False
        saw_js_runtime_warning_any = False

        for label, extractor_args in attempts:
            cmd = _build_cmd(extractor_args=extractor_args, include_remote_components=not tried_without_remote_components)
            rc, saw_percent, saw_jswarn, tail = _run_once(cmd, label)
            saw_percent_any = saw_percent_any or saw_percent
            saw_js_runtime_warning_any = saw_js_runtime_warning_any or saw_jswarn
            last_tail = tail

            if rc == 0:
                break

            tail_text = "\n".join(tail).lower()
            if (not tried_without_remote_components) and ("--remote-components" in tail_text or "remote-components" in tail_text) and ("no such option" in tail_text or "unrecognized" in tail_text):
                # Older yt-dlp: try again without remote-components in the next loop iteration (same attempt label will advance, ok).
                tried_without_remote_components = True
                self.log_line.emit("[WARN] yt-dlp no reconoce --remote-components; reintentando sin esa opción.")
                continue

            if self.platform == "youtube" and "http error 403" in tail_text:
                self.log_line.emit("[WARN] 403 detectado; probando con otro player_client.")
                continue

            # Other failures: don't keep retrying blindly
            break

        # If the last run failed, emit a clearer error message
        # (include JS runtime hint when applicable).
        if rc != 0:
            msg = f"yt-dlp terminó con código {rc}."
            if self.platform == "youtube" and saw_js_runtime_warning_any:
                msg += (
                    " YouTube puede requerir un runtime JS (deno/node/bun/quickjs) "
                    "para resolver los desafíos JS."
                )
            if last_tail:
                msg += "\n\nÚltimas líneas:\n" + "\n".join(last_tail[-12:])
            self.error.emit(msg)
            return

        # Resolve output file
        wav_path = Path(self.temp_dir) / "audio.wav"
        if not wav_path.exists():
            wavs = sorted(Path(self.temp_dir).glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
            if wavs:
                wav_path = wavs[0]
        if not wav_path.exists():
            self.error.emit("No se encontró el audio WAV descargado.")
            return

        # Ensure 16 kHz mono PCM WAV for whisper.cpp/pywhispercpp
        # yt-dlp may produce WAV at 44.1/48 kHz; pywhispercpp requires 16 kHz.
        wav_16k = Path(self.temp_dir) / "audio_16k.wav"
        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i", str(wav_path),
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-acodec", "pcm_s16le",
                str(wav_16k),
            ]
            self.log_line.emit(f"[DEBUG] Resample cmd: {' '.join(ffmpeg_cmd)}")
            proc2 = subprocess.run(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                creationflags=creation_flags,
            )
            if proc2.returncode != 0 or (not wav_16k.exists()):
                out_tail = (proc2.stdout or "").strip().splitlines()[-20:]
                self.error.emit(
                    "No se pudo convertir el audio a WAV 16 kHz mono (requerido para transcripción)."
                    + ("\n\nSalida de ffmpeg:\n" + "\n".join(out_tail) if out_tail else "")
                )
                return
            # Prefer converted file for downstream transcription
            wav_path = wav_16k
        except Exception as exc:
            self.error.emit(f"Error al convertir a 16 kHz: {exc}")
            return

        if not saw_percent_any:
            self.progress.emit(100)

        self.finished.emit(str(wav_path.resolve()))


class DragDropWidget(QLabel):
    """Drag and drop zone that is also clickable to open a file dialog.

    This widget also supports translated placeholder text via templates:
      - default_text
      - selected_template (expects {name})
    """

    files_dropped = pyqtSignal(list)
    clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._default_text: str = ""
        self._selected_template: str = ""
        self._current_paths: List[Path] = []

        self.setStyleSheet("border: 2px dashed #aaa; padding: 20px; border-radius: 8px;")

    def set_text_templates(self, default_text: str, selected_template: str) -> None:
        self._default_text = default_text or ""
        self._selected_template = selected_template or ""
        if self._current_paths:
            self.set_selected_paths(self._current_paths)
        else:
            self.reset_text()

    def set_selected_paths(self, paths: List[Path]) -> None:
        self._current_paths = list(paths or [])
        if not self._current_paths:
            self.reset_text()
            return

        if len(self._current_paths) == 1:
            name = self._current_paths[0].name
        else:
            name = f"{self._current_paths[0].name} (+{len(self._current_paths) - 1} más)"

        if self._selected_template:
            self.setText(self._selected_template.format(name=name))
        else:
            self.setText(name)

    def reset_text(self) -> None:
        self._current_paths = []
        self.setText(self._default_text)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if is_audio_file(path) or is_video_file(path):
                        event.acceptProposedAction()
                        self.setStyleSheet(
                            "border: 2px dashed #00aaff; padding: 20px; border-radius: 8px;"
                        )
                        return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet("border: 2px dashed #aaa; padding: 20px; border-radius: 8px;")
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        self.setStyleSheet("border: 2px dashed #aaa; padding: 20px; border-radius: 8px;")
        if event.mimeData().hasUrls():
            paths: List[Path] = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = Path(url.toLocalFile())
                    if is_audio_file(path) or is_video_file(path):
                        paths.append(path)
            if paths:
                self.files_dropped.emit(paths)
        event.acceptProposedAction()


###############################################################################
# Model management dialog (install / delete / status)
###############################################################################


# Descarga de modelos ggml para Voxora
#
# Voxora gestiona modelos descargándolos en su propia carpeta dentro de AppData.
# Para evitar dependencias frágiles entre versiones, la instalación de modelos
# descarga directamente los archivos `ggml-*.bin` desde el repositorio oficial
# de whisper.cpp en Hugging Face.
#
# El formato de URL coincide con el usado por pywhispercpp (MODELS_BASE_URL y
# MODELS_PREFIX_URL) en su documentación oficial.
_HF_GGML_URL_TEMPLATE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model}.bin"


def _download_ggml_model_file(model_key: str, models_dir: Path, progress_cb=None, log_cb=None) -> Path:
    """Descarga (si hace falta) un modelo ggml a models_dir y devuelve la ruta final.

    progress_cb(pct): pct = -1 para indeterminado, o 0-100.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    final_path = models_dir / f"ggml-{model_key}.bin"
    if final_path.exists():
        if log_cb:
            log_cb(f"[INFO] El modelo ya existe: {final_path.name}")
        if progress_cb:
            progress_cb(100)
        return final_path

    url = _HF_GGML_URL_TEMPLATE.format(model=model_key)
    tmp_path = models_dir / f"ggml-{model_key}.bin.part"
    if log_cb:
        log_cb(f"[INFO] Descargando modelo: {model_key}")
        log_cb(f"[DEBUG] URL: {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "Voxora/1.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            total = resp.headers.get("Content-Length")
            total_bytes = int(total) if total and str(total).isdigit() else 0
            if progress_cb:
                progress_cb(-1 if total_bytes == 0 else 0)

            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0 and progress_cb:
                        pct = int(downloaded * 100 / total_bytes)
                        progress_cb(max(0, min(100, pct)))

        # Renombrar al final (reduce riesgo de archivos corruptos)
        try:
            if final_path.exists():
                final_path.unlink()
        except Exception:
            pass
        tmp_path.replace(final_path)

        if progress_cb:
            progress_cb(100)
        if log_cb:
            log_cb(f"[INFO] Modelo descargado: {final_path.name}")
        return final_path
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise RuntimeError(f"Error descargando el modelo '{model_key}': {exc}") from exc


def _guess_pywhispercpp_default_models_dir() -> Optional[Path]:
    """Ubica la carpeta por defecto de modelos de pywhispercpp (para migración única)."""
    try:
        from pywhispercpp.constants import MODELS_DIR  # type: ignore
        return Path(MODELS_DIR)
    except Exception:
        pass

    # Fallback aproximado en Windows
    local = os.environ.get("LOCALAPPDATA") or ""
    if local:
        return Path(local) / "pywhispercpp" / "models"
    return None



def _parse_installed_models(models_dir: Path) -> Dict[str, Path]:
    """Return {model_key -> file_path} for ggml models found in models_dir."""
    out: Dict[str, Path] = {}
    if not models_dir.exists():
        return out
    for p in models_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not (name.startswith("ggml-") and name.endswith(".bin")):
            continue
        key = name[len("ggml-"):-len(".bin")]
        out[key] = p
    return out


def _open_folder(path: Path) -> None:
    try:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
    except Exception:
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(path)])
        except Exception:
            pass


class ModelInstallWorker(QObject):
    log_line = pyqtSignal(str)
    progress = pyqtSignal(int)  # -1 indeterminado, 0-100 porcentaje
    finished = pyqtSignal(str)  # ruta del modelo descargado
    error = pyqtSignal(str)

    def __init__(self, model_key: str, models_dir: str) -> None:
        super().__init__()
        self.model_key = model_key
        self.models_dir = models_dir

    def run(self) -> None:
        try:
            models_dir_path = Path(self.models_dir)

            def _log(msg: str) -> None:
                self.log_line.emit(msg)

            def _pct(p: int) -> None:
                try:
                    self.progress.emit(int(p))
                except Exception:
                    pass

            p = _download_ggml_model_file(self.model_key, models_dir_path, progress_cb=_pct, log_cb=_log)
            self.finished.emit(str(p))
        except Exception as exc:
            self.error.emit(str(exc))

class ModelManagerDialog(QDialog):
    """UI to install/delete ggml models used by Voxora."""

    def __init__(self, parent: QWidget, models_dir: Path) -> None:
        super().__init__(parent)
        self.setWindowTitle("Voxora · Modelos")
        self.resize(820, 480)
        self.models_dir = models_dir

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Modelo", "Instalado", "Archivo", "Tamaño", "Notas"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)

        self.progress = QProgressBar()
        self.progress.setVisible(False)

        self.install_btn = QPushButton("Instalar")
        self.delete_btn = QPushButton("Borrar")
        self.refresh_btn = QPushButton("Actualizar")
        self.open_dir_btn = QPushButton("Abrir carpeta")
        self.close_btn = QPushButton("Cerrar")

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.install_btn)
        btn_row.addWidget(self.delete_btn)
        btn_row.addWidget(self.refresh_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.open_dir_btn)
        btn_row.addWidget(self.close_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.info_label)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.progress)
        layout.addLayout(btn_row)
        self.setLayout(layout)

        self._thread: Optional[QThread] = None
        self._worker: Optional[ModelInstallWorker] = None

        self.refresh_btn.clicked.connect(self.refresh)
        self.open_dir_btn.clicked.connect(lambda: _open_folder(self.models_dir))
        self.close_btn.clicked.connect(self.close)
        self.install_btn.clicked.connect(self.install_selected)
        self.delete_btn.clicked.connect(self.delete_selected)

        self.refresh()

    def _available_models(self) -> List[str]:
        # Prefer pywhispercpp's constant list.
        try:
            from pywhispercpp.constants import AVAILABLE_MODELS  # type: ignore
            return list(AVAILABLE_MODELS)
        except Exception:
            # Fallback: keys from selector.
            keys = [k for _lbl, k in MODEL_CHOICES]
            # unique preserving order
            seen = set()
            out = []
            for k in keys:
                if k not in seen:
                    out.append(k)
                    seen.add(k)
            return out

    def refresh(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        installed = _parse_installed_models(self.models_dir)
        avail = self._available_models()

        self.info_label.setText(f"Carpeta de modelos: {self.models_dir}")

        self.table.setRowCount(len(avail))
        for r, key in enumerate(avail):
            file_path = installed.get(key)
            installed_txt = "Sí" if file_path else "No"
            file_txt = file_path.name if file_path else ""
            size_txt = ""
            if file_path and file_path.exists():
                try:
                    mb = file_path.stat().st_size / (1024 * 1024)
                    size_txt = f"{mb:.0f} MB"
                except Exception:
                    size_txt = ""
            notes = MODEL_HW_SPECS.get(key, {}).get("notes", "")

            self.table.setItem(r, 0, QTableWidgetItem(key))
            self.table.setItem(r, 1, QTableWidgetItem(installed_txt))
            self.table.setItem(r, 2, QTableWidgetItem(file_txt))
            self.table.setItem(r, 3, QTableWidgetItem(size_txt))
            self.table.setItem(r, 4, QTableWidgetItem(notes))

        self.delete_btn.setEnabled(True)
        self.install_btn.setEnabled(True)

    def _selected_model_key(self) -> Optional[str]:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return item.text().strip() if item else None

    def install_selected(self) -> None:
        key = self._selected_model_key()
        if not key:
            QMessageBox.information(self, "Modelo", "Selecciona un modelo.")
            return

        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Start worker
        self._worker = ModelInstallWorker(model_key=key, models_dir=str(self.models_dir))
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._install_done)
        self._worker.error.connect(self._install_error)
        self._worker.log_line.connect(lambda s: None)
        self._worker.progress.connect(self._install_progress)

        self._worker.finished.connect(self._cleanup_worker)
        self._worker.error.connect(self._cleanup_worker)

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.install_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)

        self._thread.start()

    def _install_done(self, model_path: str) -> None:
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        QMessageBox.information(self, "Modelo instalado", f"Modelo instalado:\n{model_path}")
        self.refresh()

    def _install_error(self, msg: str) -> None:
        self.progress.setVisible(False)
        self.refresh_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)
        self.refresh()


    def _install_progress(self, pct: int) -> None:
        if pct < 0:
            if self.progress.maximum() != 0:
                self.progress.setRange(0, 0)
            return
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
        self.progress.setValue(max(0, min(100, int(pct))))

    def delete_selected(self) -> None:
        key = self._selected_model_key()
        if not key:
            QMessageBox.information(self, "Modelo", "Selecciona un modelo.")
            return

        installed = _parse_installed_models(self.models_dir)
        file_path = installed.get(key)
        if not file_path:
            QMessageBox.information(self, "Borrar", "Ese modelo no está instalado en la carpeta de Voxora.")
            return

        # Prevent deleting a model that is currently loaded in memory.
        cache_key = f"{str(self.models_dir)}::{key}"
        if cache_key in _WHISPER_MODELS:
            QMessageBox.warning(
                self,
                "Modelo en uso",
                "Ese modelo está cargado en memoria. Cierra Voxora para poder borrarlo con seguridad.",
            )
            return

        resp = QMessageBox.question(
            self,
            "Confirmar",
            f"¿Borrar el modelo '{key}'?\n\nArchivo:\n{file_path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

        try:
            file_path.unlink(missing_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"No se pudo borrar:\n{exc}")
            return

        QMessageBox.information(self, "Borrado", "Modelo borrado.")
        self.refresh()

    def _cleanup_worker(self, *_args) -> None:
        try:
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait()
        except Exception:
            pass
        self._thread = None
        self._worker = None
        self.install_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)


###############################################################################
# Main application window
###############################################################################

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        # Persisted settings (cookie file paths, etc.)
        self.settings = QSettings("neura", "Voxora")

        # -----------------------------
        # Idiomas
        # -----------------------------
        # UI language (app text). Default: English.
        # Migration: older versions used "language_code" for something else.
        self.ui_lang = (self.settings.value("ui_lang", "", type=str) or "").lower().strip()
        if not self.ui_lang:
            legacy = (self.settings.value("language_code", "", type=str) or "").lower().strip()
            self.ui_lang = legacy if legacy in ("en", "es", "zh") else "en"
            self.settings.setValue("ui_lang", self.ui_lang)
        if self.ui_lang not in ("en", "es", "zh"):
            self.ui_lang = "en"
            self.settings.setValue("ui_lang", self.ui_lang)

        # Transcription audio language hint. Default: auto.
        self.transcribe_lang = (self.settings.value("transcribe_lang", "auto", type=str) or "auto").lower().strip()
        if self.transcribe_lang not in ("auto", "en", "es", "zh"):
            self.transcribe_lang = "auto"
            self.settings.setValue("transcribe_lang", self.transcribe_lang)
        self.youtube_cookies_file = self.settings.value("youtube_cookies_file", "", type=str) or ""
        self.bilibili_cookies_file = self.settings.value("bilibili_cookies_file", "", type=str) or ""

        # Modelo seleccionado (persistente). Default: small.
        self.model_key = (self.settings.value("model_key", DEFAULT_MODEL_SIZE, type=str) or DEFAULT_MODEL_SIZE).strip()
        if not self.model_key:
            self.model_key = DEFAULT_MODEL_SIZE

        # Carpeta de modelos (ggml) para Voxora. Se usa SIEMPRE esta carpeta.
        # No se consulta la ubicación por defecto de pywhispercpp en tiempo de ejecución.
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
        if not base:
            base = str(Path.home() / ".voxora")
        self.models_dir = Path(base) / "models"
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Migración única: si antes ya descargaste modelos con pywhispercpp, los copiamos aquí.
        self._migrate_legacy_models_once()

        # External tools availability
        self._ytdlp_available = shutil.which("yt-dlp") is not None
        self._ffmpeg_available = shutil.which("ffmpeg") is not None

        # Temp cleanup for URL downloads
        self._temp_paths_to_cleanup: List[Path] = []
        self._temp_dirs_to_cleanup: List[Path] = []
        # Last audio sources for optional translation
        self._last_audio_paths: List[Path] = []
        # Keep the last URL-downloaded WAV available until the user starts a new job
        self._kept_url_audio_path: Optional[Path] = None
        self._kept_url_audio_dir: Optional[Path] = None


        # URL download thread management
        self._url_thread: Optional[QThread] = None
        self._url_worker: Optional[UrlAudioDownloadWorker] = None

        # Dependency (engine) install thread management
        self._dep_thread: Optional[QThread] = None
        self._dep_worker: Optional[DependencyInstallWorker] = None
        self._pending_after_engine: Optional[Callable[[], None]] = None
        self._engine_installing: bool = False

        # Status bar animation (e.g. 'Transcribiendo...')
        self._status_anim_timer = QTimer(self)
        self._status_anim_timer.setInterval(400)
        self._status_anim_timer.timeout.connect(self._tick_status_animation)
        self._status_anim_base = ""
        self._status_anim_dots = 0

        self.setWindowTitle("Voxora")
        self.resize(900, 600)

        # Window icon (assets/img/icon.ico). If missing, Qt will ignore it.
        try:
            icon_path = resource_path(os.path.join("assets", "img", "icon.ico"))
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass

        # Load history
        self.history_path = Path(HISTORY_FILENAME)
        self.history_entries: List[TranscriptEntry] = load_history(self.history_path)

        # Create UI
        self._create_widgets()
        self._create_layout()
        self._create_connections()
        self._create_menu()
        self._apply_ui_language()

        # Status bar
        self._stop_status_animation(ui_tr(self.ui_lang, "status_ready"))
        self._cleanup_temp_downloads()

        # Thread management
        self._thread: Optional[QThread] = None
        self._worker: Optional[TranscriptionWorker] = None


    def _create_menu(self) -> None:
        bar = self.menuBar()
        bar.clear()

        # Models menu
        self.menu_models = bar.addMenu(ui_tr(self.ui_lang, "menu_models"))
        self.act_manage_models = QAction(ui_tr(self.ui_lang, "act_manage_models"), self)
        self.act_manage_models.triggered.connect(self._open_model_manager)
        self.menu_models.addAction(self.act_manage_models)

        self.act_open_models_dir = QAction(ui_tr(self.ui_lang, "act_open_models_dir"), self)
        self.act_open_models_dir.triggered.connect(lambda: _open_folder(self.models_dir))
        self.menu_models.addAction(self.act_open_models_dir)

        self.act_install_engine = QAction(ui_tr(self.ui_lang, "act_install_engine"), self)
        self.act_install_engine.triggered.connect(self._install_engine_now)
        self.menu_models.addSeparator()
        self.menu_models.addAction(self.act_install_engine)

        # Settings menu (UI language)
        self.menu_settings = bar.addMenu(ui_tr(self.ui_lang, "menu_settings"))
        self.menu_app_lang = self.menu_settings.addMenu(ui_tr(self.ui_lang, "menu_app_lang"))

        lang_group = QActionGroup(self)
        lang_group.setExclusive(True)

        def _add_lang_action(code: str, label: str) -> None:
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(self.ui_lang == code)
            act.triggered.connect(lambda _checked=False, c=code: self._set_ui_language(c))
            lang_group.addAction(act)
            self.menu_app_lang.addAction(act)

        # Show language names in their own language for clarity
        _add_lang_action("en", "English")
        _add_lang_action("es", "Español")
        _add_lang_action("zh", "中文")

        # Help menu
        self.menu_help = bar.addMenu(ui_tr(self.ui_lang, "menu_help"))
        self.act_about = QAction(ui_tr(self.ui_lang, "act_about"), self)
        self.act_about.triggered.connect(self._about)
        self.menu_help.addAction(self.act_about)




    def _set_ui_language(self, code: str) -> None:
        code = (code or "").lower().strip()
        if code not in ("en", "es", "zh"):
            code = "en"
        if self.ui_lang == code:
            return
        self.ui_lang = code
        try:
            self.settings.setValue("ui_lang", code)
        except Exception:
            pass
        self._apply_ui_language()

    def _apply_ui_language(self) -> None:
        """Apply current UI language to visible texts."""
        # Tabs
        try:
            self.tabs.setTabText(0, ui_tr(self.ui_lang, "tab_files"))
            self.tabs.setTabText(1, ui_tr(self.ui_lang, "tab_links"))
            self.tabs.setTabText(2, ui_tr(self.ui_lang, "tab_history"))
            self.tabs.setTabText(3, ui_tr(self.ui_lang, "tab_req"))
        except Exception:
            pass

        # Toolbar
        try:
            self.toolbar.setWindowTitle("Voxora")
            self.cancel_action.setText(ui_tr(self.ui_lang, "toolbar_cancel"))
            self.toggle_terminal_action.setText(ui_tr(self.ui_lang, "toolbar_terminal"))
        except Exception:
            pass

        # Drag/drop placeholder
        try:
            self.drop_widget.set_text_templates(
                ui_tr(self.ui_lang, "drop_default"),
                ui_tr(self.ui_lang, "drop_selected", name="{name}"),
            )
        except Exception:
            pass

        # Audio language row
        try:
            self.audio_lang_label.setText(ui_tr(self.ui_lang, "label_audio_lang"))
            self.model_label.setText(ui_tr(self.ui_lang, "label_model"))
        except Exception:
            pass

        # Combos and buttons
        try:
            self._rebuild_audio_lang_combo_items()
        except Exception:
            pass
        try:
            self.copy_button.setText(ui_tr(self.ui_lang, "btn_copy"))
            self.translate_button.setText(ui_tr(self.ui_lang, "btn_translate"))
            self.translate_button.setToolTip(ui_tr(self.ui_lang, "msg_translate_body"))
        except Exception:
            pass

        # Transcript + terminal placeholders
        try:
            self.text_edit.setPlaceholderText(ui_tr(self.ui_lang, "placeholder_transcript"))
            self.terminal.setPlaceholderText(ui_tr(self.ui_lang, "placeholder_terminal"))
        except Exception:
            pass

        # Find bar
        try:
            self.find_label_widget.setText(ui_tr(self.ui_lang, "find_label"))
            self.find_input.setPlaceholderText(ui_tr(self.ui_lang, "find_placeholder"))
            self.find_prev_btn.setText(ui_tr(self.ui_lang, "find_prev"))
            self.find_next_btn.setText(ui_tr(self.ui_lang, "find_next"))
            self.find_close_btn.setText(ui_tr(self.ui_lang, "find_close"))
        except Exception:
            pass

        # Links tab
        try:
            self.url_group.setTitle(ui_tr(self.ui_lang, "group_url"))
            self.url_label.setText(ui_tr(self.ui_lang, "label_url"))
            self.url_input.setPlaceholderText(ui_tr(self.ui_lang, "placeholder_url"))
            self.url_transcribe_button.setText(ui_tr(self.ui_lang, "btn_transcribe_url"))
            self.youtube_cookies_label.setText(ui_tr(self.ui_lang, "label_youtube_cookies"))
            self.bilibili_cookies_label.setText(ui_tr(self.ui_lang, "label_bilibili_cookies"))
            self.youtube_cookies_btn.setText(ui_tr(self.ui_lang, "btn_load"))
            self.youtube_cookies_clear_btn.setText(ui_tr(self.ui_lang, "btn_remove"))
            self.bilibili_cookies_btn.setText(ui_tr(self.ui_lang, "btn_load"))
            self.bilibili_cookies_clear_btn.setText(ui_tr(self.ui_lang, "btn_remove"))
        except Exception:
            pass

        # History tab buttons
        try:
            self.copy_history_button.setText(ui_tr(self.ui_lang, "btn_copy_selected"))
            self.delete_history_button.setText(ui_tr(self.ui_lang, "btn_delete_selected"))
        except Exception:
            pass

        # Rebuild menubar in the new language
        try:
            self._create_menu()
        except Exception:
            pass

        # Status bar
        try:
            self._stop_status_animation(ui_tr(self.ui_lang, "status_ready"))
        except Exception:
            pass

    def _is_engine_available(self) -> bool:
        """Returns True if the transcription engine (pywhispercpp) is importable."""
        return WhisperCppModel is not None


    def _refresh_engine_import(self) -> bool:
        """Try to (re)import pywhispercpp after installation."""
        global WhisperCppModel
        try:
            importlib.invalidate_caches()
            from pywhispercpp.model import Model as _Model  # type: ignore
            WhisperCppModel = _Model  # type: ignore
            return True
        except Exception:
            WhisperCppModel = None  # type: ignore
            return False


    def _ensure_engine_and_then(self, callback: Callable[[], None]) -> None:
        """Ensure pywhispercpp is installed; then run callback."""
        if self._is_engine_available():
            callback()
            return

        # If already installing, just remember the last requested action.
        self._pending_after_engine = callback
        if self._engine_installing:
            self._start_status_animation(ui_tr(self.ui_lang, "status_installing_engine"))
            return

        # Start installation in background
        self._install_engine_now()


    def _install_engine_now(self) -> None:
        if self._engine_installing:
            return
        self._engine_installing = True
        self._start_status_animation(ui_tr(self.ui_lang, "status_installing_engine"))
        self._append_terminal_line("[INFO] Instalando dependencia: pywhispercpp")

        # Disable inputs while installing
        try:
            self.drop_widget.setEnabled(False)
        except Exception:
            pass
        try:
            self.url_transcribe_button.setEnabled(False)
        except Exception:
            pass

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # indeterminate

        self._dep_worker = DependencyInstallWorker("pywhispercpp")
        self._dep_thread = QThread()
        self._dep_worker.moveToThread(self._dep_thread)
        self._dep_thread.started.connect(self._dep_worker.run)
        self._dep_worker.log_line.connect(self._append_terminal_line)
        self._dep_worker.finished.connect(self._on_engine_install_finished)
        self._dep_worker.finished.connect(self._cleanup_engine_install_worker)
        self._dep_thread.start()


    def _on_engine_install_finished(self, success: bool, message: str) -> None:
        if success:
            ok = self._refresh_engine_import()
            if ok:
                self._append_terminal_line("[INFO] Motor instalado y listo.")
            else:
                success = False
                message = "Se instaló pywhispercpp, pero no se pudo importar.\n" + message
        if not success:
            self._append_terminal_line("[ERROR] " + message)
            QMessageBox.critical(
                self,
                "No se pudo instalar el motor",
                message + "\n\nPuedes intentar manualmente: pip install pywhispercpp",
            )
            self._engine_installing = False
            self.progress_bar.setVisible(False)
            self._stop_status_animation(ui_tr(self.ui_lang, "status_ready"))
            try:
                self.drop_widget.setEnabled(True)
            except Exception:
                pass
            try:
                self.url_transcribe_button.setEnabled(True)
            except Exception:
                pass
            return

        # Success path
        self._engine_installing = False
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self._stop_status_animation(ui_tr(self.ui_lang, "status_ready"))
        try:
            self.drop_widget.setEnabled(True)
        except Exception:
            pass
        try:
            self.url_transcribe_button.setEnabled(True)
        except Exception:
            pass

        cb = self._pending_after_engine
        self._pending_after_engine = None
        if cb:
            try:
                cb()
            except Exception as exc:
                self._append_terminal_line(f"[ERROR] Error al continuar después de instalar el motor: {exc}")


    def _cleanup_engine_install_worker(self, *args) -> None:
        try:
            if self._dep_thread is not None:
                self._dep_thread.quit()
                self._dep_thread.wait()
        except Exception:
            pass
        self._dep_thread = None
        self._dep_worker = None

    def _open_model_manager(self) -> None:
        dlg = ModelManagerDialog(self, self.models_dir)
        dlg.exec()
        # Refresh model combo labels after installs/deletes
        try:
            self._refresh_model_combo_labels()
        except Exception:
            pass

    def _about(self) -> None:
        QMessageBox.information(
            self,
            "Voxora",
            "Voxora\n\nTranscripción local con whisper.cpp (pywhispercpp).",
        )

    def _migrate_legacy_models_once(self) -> None:
        """Copia modelos ggml existentes (pywhispercpp) a la carpeta de Voxora una sola vez."""
        try:
            done = self.settings.value("migrated_pywhispercpp_models", False, bool)
        except Exception:
            done = False
        if done:
            return

        src = _guess_pywhispercpp_default_models_dir()
        if not src or not src.exists() or not src.is_dir():
            try:
                self.settings.setValue("migrated_pywhispercpp_models", True)
            except Exception:
                pass
            return

        # Si ya es la misma carpeta, no hacemos nada.
        try:
            if src.resolve() == self.models_dir.resolve():
                self.settings.setValue("migrated_pywhispercpp_models", True)
                return
        except Exception:
            pass

        copied = 0
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        for f in src.glob("ggml-*.bin"):
            try:
                dst = self.models_dir / f.name
                if dst.exists():
                    continue
                shutil.copy2(str(f), str(dst))
                copied += 1
            except Exception:
                continue

        try:
            self.settings.setValue("migrated_pywhispercpp_models", True)
        except Exception:
            pass

        if copied > 0:
            try:
                self._append_terminal_line(
                    f"[INFO] Se importaron {copied} modelo(s) desde pywhispercpp a la carpeta de Voxora."
                )
            except Exception:
                pass

    def _create_widgets(self) -> None:
        # Toolbar with actions
        self.toolbar = QToolBar("Barra de herramientas")
        self.addToolBar(self.toolbar)        # Cancel action
        self.cancel_action = QAction("Cancelar", self)
        self.cancel_action.setStatusTip("Cancelar la transcripción en curso")
        self.cancel_action.setEnabled(False)
        self.toolbar.addAction(self.cancel_action)
        # Toggle terminal action
        self.toggle_terminal_action = QAction("Terminal", self)
        self.toggle_terminal_action.setCheckable(True)
        self.toggle_terminal_action.setChecked(False)
        self.toggle_terminal_action.setStatusTip("Mostrar u ocultar la terminal de logs")
        self.toolbar.addAction(self.toggle_terminal_action)

        # Tabs: one for transcription, one for history
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # Tab 1: Transcription
        self.transcription_tab = QWidget()
        self.tabs.addTab(self.transcription_tab, "Archivo")
        # Drag/drop area
        self.drop_widget = DragDropWidget()
        self.drop_widget.set_text_templates(
            ui_tr(self.ui_lang, "drop_default"),
            ui_tr(self.ui_lang, "drop_selected", name="{name}"),
        )

        # Tab 2: Enlaces (YouTube / Bilibili)
        self.links_tab = QWidget()
        self.tabs.addTab(self.links_tab, "Enlaces")

        # URL transcription (YouTube / Bilibili)
        self.url_group = QGroupBox(ui_tr(self.ui_lang, "group_url"))
        url_layout = QVBoxLayout(self.url_group)

        url_row = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText(ui_tr(self.ui_lang, "placeholder_url"))
        self.url_input.setClearButtonEnabled(True)
        self.url_transcribe_button = QPushButton(ui_tr(self.ui_lang, "btn_transcribe_url"))
        self.url_label = QLabel(ui_tr(self.ui_lang, "label_url"))
        url_row.addWidget(self.url_label)
        url_row.addWidget(self.url_input, 1)
        url_row.addWidget(self.url_transcribe_button)
        url_layout.addLayout(url_row)

        cookies_row_yt = QHBoxLayout()
        self.youtube_cookies_display = QLineEdit()
        self.youtube_cookies_display.setReadOnly(True)
        self.youtube_cookies_display.setPlaceholderText("Cookies de YouTube (opcional)")
        if self.youtube_cookies_file:
            self.youtube_cookies_display.setText(os.path.basename(self.youtube_cookies_file))
            self.youtube_cookies_display.setToolTip(self.youtube_cookies_file)
        self.youtube_cookies_btn = QPushButton(ui_tr(self.ui_lang, "btn_load"))
        self.youtube_cookies_clear_btn = QPushButton(ui_tr(self.ui_lang, "btn_remove"))
        self.youtube_cookies_label = QLabel(ui_tr(self.ui_lang, "label_youtube_cookies"))
        cookies_row_yt.addWidget(self.youtube_cookies_label)
        cookies_row_yt.addWidget(self.youtube_cookies_display, 1)
        cookies_row_yt.addWidget(self.youtube_cookies_btn)
        cookies_row_yt.addWidget(self.youtube_cookies_clear_btn)
        url_layout.addLayout(cookies_row_yt)

        cookies_row_bi = QHBoxLayout()
        self.bilibili_cookies_display = QLineEdit()
        self.bilibili_cookies_display.setReadOnly(True)
        self.bilibili_cookies_display.setPlaceholderText("Cookies de Bilibili (opcional)")
        if self.bilibili_cookies_file:
            self.bilibili_cookies_display.setText(os.path.basename(self.bilibili_cookies_file))
            self.bilibili_cookies_display.setToolTip(self.bilibili_cookies_file)
        self.bilibili_cookies_btn = QPushButton(ui_tr(self.ui_lang, "btn_load"))
        self.bilibili_cookies_clear_btn = QPushButton(ui_tr(self.ui_lang, "btn_remove"))
        self.bilibili_cookies_label = QLabel(ui_tr(self.ui_lang, "label_bilibili_cookies"))
        cookies_row_bi.addWidget(self.bilibili_cookies_label)
        cookies_row_bi.addWidget(self.bilibili_cookies_display, 1)
        cookies_row_bi.addWidget(self.bilibili_cookies_btn)
        cookies_row_bi.addWidget(self.bilibili_cookies_clear_btn)
        url_layout.addLayout(cookies_row_bi)

        # Disable URL section if yt-dlp or ffmpeg is missing
        if not self._ytdlp_available or not self._ffmpeg_available:
            self.url_group.setEnabled(False)
            missing = []
            if not self._ytdlp_available:
                missing.append("yt-dlp")
            if not self._ffmpeg_available:
                missing.append("ffmpeg")
            self.url_group.setToolTip("Faltan dependencias: " + ", ".join(missing))


        links_layout = QVBoxLayout()
        links_layout.addWidget(self.url_group)
        links_layout.addStretch()
        self.links_tab.setLayout(links_layout)

                # Audio language hint (for transcription). Default: auto detect.
        self.audio_lang_combo = QComboBox()
        self._rebuild_audio_lang_combo_items()

        # Model selector
        self.model_combo = QComboBox()
        for label, key in MODEL_CHOICES:
            self.model_combo.addItem(label, userData=key)
        # Select persisted model (fallback to DEFAULT_MODEL_SIZE)
        saved_model = (getattr(self, "model_key", "") or DEFAULT_MODEL_SIZE).strip()
        default_index = 0
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == saved_model:
                default_index = i
                break
        else:
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == DEFAULT_MODEL_SIZE:
                    default_index = i
                    break
        self.model_combo.setCurrentIndex(default_index)
        self.model_combo.setToolTip("Selecciona el modelo Whisper (tamaño/velocidad/precisión)")

        try:
            self._refresh_model_combo_labels()
        except Exception:
            pass

        # Translate button (reprocesses the same audio using Whisper's translate task to English)
        self.translate_button = QPushButton(ui_tr(self.ui_lang, "btn_translate"))
        self.translate_button.setEnabled(False)
        self.translate_button.setToolTip(ui_tr(self.ui_lang, "msg_translate_body"))

        # Find bar (Ctrl+F)
        self.find_bar = QWidget()
        find_layout = QHBoxLayout(self.find_bar)
        find_layout.setContentsMargins(0, 0, 0, 0)
        self.find_input = QLineEdit()
        self.find_input.setPlaceholderText(ui_tr(self.ui_lang, "find_placeholder"))
        self.find_case = QCheckBox("Aa")
        self.find_case.setToolTip("Coincidir mayúsculas/minúsculas")
        self.find_prev_btn = QPushButton(ui_tr(self.ui_lang, "find_prev"))
        self.find_next_btn = QPushButton(ui_tr(self.ui_lang, "find_next"))
        self.find_close_btn = QToolButton()
        self.find_close_btn.setText(ui_tr(self.ui_lang, "find_close"))
        self.find_label_widget = QLabel(ui_tr(self.ui_lang, "find_label"))
        find_layout.addWidget(self.find_label_widget)
        find_layout.addWidget(self.find_input, 1)
        find_layout.addWidget(self.find_case)
        find_layout.addWidget(self.find_prev_btn)
        find_layout.addWidget(self.find_next_btn)
        find_layout.addWidget(self.find_close_btn)
        self.find_bar.setVisible(False)

        # Transcript text area
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText(ui_tr(self.ui_lang, "placeholder_transcript"))
        self.text_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        # Terminal (logs)
        self.terminal = QPlainTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setPlaceholderText(ui_tr(self.ui_lang, "placeholder_terminal"))
        self.terminal.setVisible(False)
        # Keep memory bounded even if the terminal is hidden
        self.terminal.setMaximumBlockCount(3000)
        # Copy button
        self.copy_button = QPushButton(ui_tr(self.ui_lang, "btn_copy"))
        self.copy_button.setEnabled(False)
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        # Layout for transcription tab
        trans_layout = QVBoxLayout()
        trans_layout.addWidget(self.drop_widget)
        # Horizontal layout: language selector and copy button
        hlayout = QHBoxLayout()
        self.audio_lang_label = QLabel(ui_tr(self.ui_lang, "label_audio_lang"))
        hlayout.addWidget(self.audio_lang_label)
        hlayout.addWidget(self.audio_lang_combo)
        self.model_label = QLabel(ui_tr(self.ui_lang, "label_model"))
        hlayout.addWidget(self.model_label)
        hlayout.addWidget(self.model_combo)
        hlayout.addStretch()
        hlayout.addWidget(self.translate_button)
        hlayout.addWidget(self.copy_button)
        trans_layout.addLayout(hlayout)
        trans_layout.addWidget(self.find_bar)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.text_edit)
        splitter.addWidget(self.terminal)
        splitter.setSizes([400, 200])
        trans_layout.addWidget(splitter, 1)
        trans_layout.addWidget(self.progress_bar)
        self.transcription_tab.setLayout(trans_layout)

        # Tab 2: History
        self.history_tab = QWidget()
        self.tabs.addTab(self.history_tab, "Historial")
        self.history_list = QListWidget()
        self._populate_history_list()
        self.history_text_edit = QTextEdit()
        self.history_text_edit.setReadOnly(True)
        # Buttons for history operations
        self.copy_history_button = QPushButton(ui_tr(self.ui_lang, "btn_copy_selected"))
        self.copy_history_button.setEnabled(False)
        self.delete_history_button = QPushButton(ui_tr(self.ui_lang, "btn_delete_selected"))
        self.delete_history_button.setEnabled(False)
        # Layout for history tab
        hsplit = QSplitter(Qt.Orientation.Horizontal)
        hsplit.addWidget(self.history_list)
        hsplit.addWidget(self.history_text_edit)
        hsplit.setSizes([200, 700])
        hist_layout = QVBoxLayout()
        hist_layout.addWidget(hsplit)
        hist_buttons_layout = QHBoxLayout()
        hist_buttons_layout.addWidget(self.copy_history_button)
        hist_buttons_layout.addWidget(self.delete_history_button)
        hist_buttons_layout.addStretch()
        hist_layout.addLayout(hist_buttons_layout)
        self.history_tab.setLayout(hist_layout)

        # Tab 3: Modelos y requisitos
        self.models_tab = QWidget()
        self.tabs.addTab(self.models_tab, "Requisitos")

        self.models_info = QLabel(
            "Especificaciones orientativas por modelo.\n"
            "CPU y RAM: mínimo / recomendado para uso cómodo.\n"
            "VRAM: mínimo / recomendado si usas un backend con GPU (referencia)."
        )
        self.models_info.setWordWrap(True)

        self.models_table = QTableWidget()
        self.models_table.setColumnCount(5)
        self.models_table.setHorizontalHeaderLabels([
            "Modelo",
            "CPU (mín / rec)",
            "RAM (mín / rec)",
            "VRAM (mín / rec)",
            "Notas",
        ])
        self.models_table.verticalHeader().setVisible(False)
        self.models_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.models_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.models_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.models_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self._populate_models_table()

        models_layout = QVBoxLayout()
        models_layout.addWidget(self.models_info)
        models_layout.addWidget(self.models_table)
        self.models_tab.setLayout(models_layout)


    def _create_layout(self) -> None:
        """Reserved for future extension: currently handled in _create_widgets."""
        pass

    def _create_connections(self) -> None:        # Cancel action
        self.cancel_action.triggered.connect(self._cancel_transcription)
        # Terminal toggle
        self.toggle_terminal_action.toggled.connect(self._set_terminal_visible)
        # Drag/drop events
        self.drop_widget.files_dropped.connect(self._handle_dropped_files)
        self.drop_widget.clicked.connect(self._open_file_dialog)
        # URL transcription
        self.url_transcribe_button.clicked.connect(self._transcribe_from_url)
        self.url_input.returnPressed.connect(self._transcribe_from_url)
        self.youtube_cookies_btn.clicked.connect(self._load_youtube_cookies)
        self.youtube_cookies_clear_btn.clicked.connect(self._clear_youtube_cookies)
        self.bilibili_cookies_btn.clicked.connect(self._load_bilibili_cookies)
        self.bilibili_cookies_clear_btn.clicked.connect(self._clear_bilibili_cookies)

        # Ctrl+F find in transcript
        self.find_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        self.find_shortcut.activated.connect(self._show_find_bar)
        self.find_input.returnPressed.connect(self._find_next)
        self.find_next_btn.clicked.connect(self._find_next)
        self.find_prev_btn.clicked.connect(self._find_prev)
        self.find_close_btn.clicked.connect(self._hide_find_bar)
        # Escape hides the find bar
        self.find_escape_shortcut = QShortcut(QKeySequence("Escape"), self.find_bar)
        self.find_escape_shortcut.activated.connect(self._hide_find_bar)

        # Copy buttons
        self.copy_button.clicked.connect(self._copy_current_transcript)
        self.translate_button.clicked.connect(self._translate_last_audio)
        self.copy_history_button.clicked.connect(self._copy_selected_history)
        # Delete history
        self.delete_history_button.clicked.connect(self._delete_selected_history)
        # History list selection
        self.history_list.currentRowChanged.connect(self._display_history_entry)

        # Persist transcription audio language hint
        self.audio_lang_combo.currentIndexChanged.connect(self._persist_transcribe_language_setting)
        # Persist model selection
        self.model_combo.currentIndexChanged.connect(self._persist_model_setting)


    # ---------------------------------------------------------------------
    # Find bar (Ctrl+F)
    # ---------------------------------------------------------------------

    def _show_find_bar(self) -> None:
        self.find_bar.setVisible(True)
        self.find_input.setFocus()
        self.find_input.selectAll()

    def _hide_find_bar(self) -> None:
        self.find_bar.setVisible(False)
        self.text_edit.setFocus()

    def _find_next(self) -> None:
        needle = (self.find_input.text() or "").strip()
        if not needle:
            return
        flags = QTextDocument.FindFlag(0)
        if self.find_case.isChecked():
            flags |= QTextDocument.FindFlag.FindCaseSensitively
        found = self.text_edit.find(needle, flags)
        if not found:
            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            self.text_edit.setTextCursor(cursor)
            self.text_edit.find(needle, flags)

    def _find_prev(self) -> None:
        needle = (self.find_input.text() or "").strip()
        if not needle:
            return
        flags = QTextDocument.FindFlag.FindBackward
        if self.find_case.isChecked():
            flags |= QTextDocument.FindFlag.FindCaseSensitively
        found = self.text_edit.find(needle, flags)
        if not found:
            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.text_edit.setTextCursor(cursor)
            self.text_edit.find(needle, flags)

    # ---------------------------------------------------------------------
    # Status bar animation helpers
    # ---------------------------------------------------------------------

    def _start_status_animation(self, base_text: str) -> None:
        self._status_anim_base = base_text
        self._status_anim_dots = 0
        self.statusBar().showMessage(base_text)
        # show first dot immediately
        self._tick_status_animation()
        if not self._status_anim_timer.isActive():
            self._status_anim_timer.start()

    def _tick_status_animation(self) -> None:
        if not self._status_anim_base:
            return
        self._status_anim_dots = (self._status_anim_dots % 3) + 1
        dots = "." * self._status_anim_dots
        self.statusBar().showMessage(f"{self._status_anim_base}{dots}")

    def _stop_status_animation(self, final_text: str = "Listo") -> None:
        if self._status_anim_timer.isActive():
            self._status_anim_timer.stop()
        self._status_anim_base = ""
        self._status_anim_dots = 0
        self.statusBar().showMessage(final_text)

    # ---------------------------------------------------------------------
    # URL download and transcription
    # ---------------------------------------------------------------------

    def _determine_platform(self, url: str) -> Optional[str]:
        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower()
        except Exception:
            host = ""
        if "youtu.be" in host or "youtube.com" in host or "m.youtube.com" in host:
            return "youtube"
        if "bilibili.com" in host:
            return "bilibili"
        return None

    def _load_youtube_cookies(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo de cookies de YouTube",
            "",
            "Cookies (*.txt);;Todos los archivos (*)",
        )
        if not file_path:
            return
        self.youtube_cookies_file = file_path
        self.youtube_cookies_display.setText(os.path.basename(file_path))
        self.youtube_cookies_display.setToolTip(file_path)
        self.settings.setValue("youtube_cookies_file", file_path)

    def _clear_youtube_cookies(self) -> None:
        self.youtube_cookies_file = ""
        self.youtube_cookies_display.clear()
        self.youtube_cookies_display.setToolTip("")
        self.settings.setValue("youtube_cookies_file", "")

    def _load_bilibili_cookies(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo de cookies de Bilibili",
            "",
            "Cookies (*.txt);;Todos los archivos (*)",
        )
        if not file_path:
            return
        self.bilibili_cookies_file = file_path
        self.bilibili_cookies_display.setText(os.path.basename(file_path))
        self.bilibili_cookies_display.setToolTip(file_path)
        self.settings.setValue("bilibili_cookies_file", file_path)

    def _clear_bilibili_cookies(self) -> None:
        self.bilibili_cookies_file = ""
        self.bilibili_cookies_display.clear()
        self.bilibili_cookies_display.setToolTip("")
        self.settings.setValue("bilibili_cookies_file", "")

    def _transcribe_from_url(self) -> None:
        if self._thread is not None or self._url_thread is not None:
            QMessageBox.warning(
                self,
                ui_tr(self.ui_lang, "msg_busy_title"),
                ui_tr(self.ui_lang, "msg_busy_body"),
            )
            return

        # Starting a new URL job: drop any previously kept URL audio
        self._cleanup_kept_url_audio()

        url = (self.url_input.text() or "").strip().strip('"')
        if not url:
            QMessageBox.information(self, "URL vacía", "Pega una URL de YouTube o Bilibili.")
            return

        platform = self._determine_platform(url)
        if platform is None:
            QMessageBox.warning(self, "URL no soportada", "Solo se soportan URLs de YouTube y Bilibili.")
            return

        if not self._ytdlp_available:
            QMessageBox.critical(self, "Falta yt-dlp", "No se encontró yt-dlp en PATH. Instálalo para usar transcripción por URL.")
            return

        if not self._ffmpeg_available:
            QMessageBox.critical(self, "Falta ffmpeg", "No se encontró ffmpeg en PATH. Es necesario para extraer audio.")
            return

        # Show main tab so you can see progress/logs
        try:
            self.tabs.setCurrentIndex(0)
        except Exception:
            pass

        # Prepare temp dir
        tmp_dir = tempfile.mkdtemp(prefix="pytranscriber_url_")
        tmp_dir_path = Path(tmp_dir)
        self._temp_dirs_to_cleanup.append(tmp_dir_path)

        # Progress bar starts indeterminate, can switch to determinate if we parse percent
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)

        self._start_status_animation(ui_tr(self.ui_lang, "status_downloading_audio"))

        cookies = ""
        if platform == "youtube":
            cookies = self.youtube_cookies_file or ""
        elif platform == "bilibili":
            cookies = self.bilibili_cookies_file or ""

        self._url_worker = UrlAudioDownloadWorker(url=url, platform=platform, cookies_path=cookies, temp_dir=tmp_dir)
        self._url_thread = QThread()
        self._url_worker.moveToThread(self._url_thread)

        self._url_thread.started.connect(self._url_worker.run)
        self._url_worker.log_line.connect(self._append_terminal_line)
        self._url_worker.progress.connect(self._url_download_progress)
        self._url_worker.finished.connect(self._url_download_finished)
        self._url_worker.error.connect(self._url_download_error)

        # ensure cleanup of thread object
        self._url_worker.finished.connect(self._cleanup_url_worker)
        self._url_worker.error.connect(self._cleanup_url_worker)

        self.cancel_action.setEnabled(True)
        self._url_thread.start()

    def _url_download_progress(self, pct: int) -> None:
        if pct < 0:
            if self.progress_bar.maximum() != 0:
                self.progress_bar.setRange(0, 0)
            return
        # Switch to determinate mode
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(max(0, min(100, int(pct))))

    def _url_download_finished(self, wav_path_str: str) -> None:
        # Mark URL worker done early so Cancel applies to transcription stage
        self._url_worker = None
        wav_path = Path(wav_path_str)
        # Some antivirus/cleanup tools can briefly delay file visibility on Windows.
        # If the file is not visible yet, wait a moment and/or pick the newest WAV in the folder.
        if not wav_path.exists():
            try:
                time.sleep(0.15)
            except Exception:
                pass
        if not wav_path.exists():
            try:
                parent = wav_path.parent
                if parent.exists():
                    cands = sorted(parent.glob('*.wav'), key=lambda p: p.stat().st_mtime, reverse=True)
                    if cands:
                        wav_path = cands[0]
            except Exception:
                pass
        if not wav_path.exists():
            self._append_terminal_line(f"[ERROR] El archivo descargado no existe: {wav_path}")
            self._stop_status_animation(ui_tr(self.ui_lang, "status_ready"))
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", "No se encontró el archivo de audio descargado. Revisa los logs de yt-dlp/ffmpeg.")
            self._cleanup_temp_downloads()
            return
        # Keep this file available for the optional Translate button.
        self._kept_url_audio_path = wav_path
        self._kept_url_audio_dir = wav_path.parent

        # Show selected file name in the drop zone placeholder
        try:
            self.drop_widget.set_selected_paths([wav_path])
        except Exception:
            pass

        try:
            self.tabs.setCurrentIndex(0)
        except Exception:
            pass

        # Reset progress bar for transcription stage
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self._append_terminal_line("[INFO] Descarga completa. Iniciando transcripción...")

        # Now start transcription using existing pipeline
        # IMPORTANT: preserve the just-downloaded URL audio; otherwise _start_transcription would
        # delete it via _cleanup_kept_url_audio(), causing 'No such file or directory' in ffmpeg.
        self._start_transcription([wav_path], preserve_kept_url=True)

    def _url_download_error(self, msg: str) -> None:
        self._append_terminal_line(f"[ERROR] {msg}")
        self._stop_status_animation(ui_tr(self.ui_lang, "status_ready"))
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error en descarga", msg)
        self._cleanup_temp_downloads()

    def _cleanup_url_worker(self) -> None:
        # Stop and dispose URL download thread/worker
        if self._url_thread is not None:
            self._url_thread.quit()
            self._url_thread.wait()
            self._url_thread = None
        self._url_worker = None
        # Cancel action might still be used by transcription worker
        if self._thread is None:
            self.cancel_action.setEnabled(False)


    def _cleanup_temp_downloads(self) -> None:
        """Delete temporary files/directories from URL downloads.

        Note: Voxora keeps the most recent URL-downloaded WAV temporarily so the
        user can use the Translate button. That kept file and its directory are
        skipped here.
        """
        for p in list(self._temp_paths_to_cleanup):
            if self._kept_url_audio_path is not None and p == self._kept_url_audio_path:
                continue
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
            finally:
                try:
                    self._temp_paths_to_cleanup.remove(p)
                except ValueError:
                    pass

        for d in list(self._temp_dirs_to_cleanup):
            if self._kept_url_audio_dir is not None and d == self._kept_url_audio_dir:
                continue
            try:
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
            finally:
                try:
                    self._temp_dirs_to_cleanup.remove(d)
                except ValueError:
                    pass

    def _cleanup_kept_url_audio(self) -> None:
        """Remove the last URL-downloaded audio that we kept for translation."""
        p = self._kept_url_audio_path
        d = self._kept_url_audio_dir
        self._kept_url_audio_path = None
        self._kept_url_audio_dir = None

        if p is not None:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
            try:
                if p in self._temp_paths_to_cleanup:
                    self._temp_paths_to_cleanup.remove(p)
            except Exception:
                pass

        if d is not None:
            try:
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
            try:
                if d in self._temp_dirs_to_cleanup:
                    self._temp_dirs_to_cleanup.remove(d)
            except Exception:
                pass

    def _rebuild_audio_lang_combo_items(self) -> None:
        """(Re)build the items of the audio language combo based on current UI language."""
        if not hasattr(self, "audio_lang_combo") or self.audio_lang_combo is None:
            return
        current = (self.transcribe_lang or "auto").lower().strip()
        if current not in ("auto", "en", "es", "zh"):
            current = "auto"
        self.audio_lang_combo.blockSignals(True)
        try:
            self.audio_lang_combo.clear()
            self.audio_lang_combo.addItem(ui_tr(self.ui_lang, "audio_auto"), userData="auto")
            self.audio_lang_combo.addItem(ui_tr(self.ui_lang, "lang_en"), userData="en")
            self.audio_lang_combo.addItem(ui_tr(self.ui_lang, "lang_es"), userData="es")
            self.audio_lang_combo.addItem(ui_tr(self.ui_lang, "lang_zh"), userData="zh")
            idx = self.audio_lang_combo.findData(current)
            self.audio_lang_combo.setCurrentIndex(idx if idx >= 0 else 0)
        finally:
            self.audio_lang_combo.blockSignals(False)

    def _refresh_model_combo_labels(self) -> None:
        installed = _parse_installed_models(self.models_dir)
        for i in range(self.model_combo.count()):
            key = self.model_combo.itemData(i)
            if not key:
                continue
            base_text = self.model_combo.itemText(i)
            # Remove prior marker
            base_text = base_text.replace(" [instalado]", "")
            if key in installed:
                self.model_combo.setItemText(i, base_text + " [instalado]")
            else:
                self.model_combo.setItemText(i, base_text)

    def _populate_models_table(self) -> None:
        rows = []
        for label, key in MODEL_CHOICES:
            spec = MODEL_HW_SPECS.get(key, {"cpu": "", "ram": "", "vram": "", "notes": ""})
            rows.append((label, spec.get("cpu", ""), spec.get("ram", ""), spec.get("vram", ""), spec.get("notes", "")))

        self.models_table.setRowCount(len(rows))
        for r, (model_label, cpu, ram, vram, notes) in enumerate(rows):
            self.models_table.setItem(r, 0, QTableWidgetItem(model_label))
            self.models_table.setItem(r, 1, QTableWidgetItem(cpu))
            self.models_table.setItem(r, 2, QTableWidgetItem(ram))
            self.models_table.setItem(r, 3, QTableWidgetItem(vram))
            self.models_table.setItem(r, 4, QTableWidgetItem(notes))




    def _open_file_dialog(self) -> None:
        """Open file selection dialog to choose audio/video files."""
        options = QFileDialog.Option.ReadOnly | QFileDialog.Option.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleccionar archivos de audio o video",
            "",
            "Audio/Video Files (*.wav *.mp3 *.ogg *.flac *.m4a *.opus *.aac *.caf *.webm *.mp4 *.mkv *.avi *.mov *.wmv *.flv *.m4v)",
            options=options,
        )
        if files:
            paths = [Path(f) for f in files]
            self._start_transcription(paths)

    def _handle_dropped_files(self, paths: List[Path]) -> None:
        """Called when files are dropped onto the drag-and-drop widget."""
        self._start_transcription(paths)

    def _start_transcription(self, paths: List[Path], *, translate_to_en: bool = False, preserve_kept_url: bool = False) -> None:
        """Start transcription of selected or dropped files."""
        if self._thread is not None:
            QMessageBox.warning(
                self,
                ui_tr(self.ui_lang, "msg_busy_title"),
                ui_tr(self.ui_lang, "msg_busy_body"),
            )
            return

        if not preserve_kept_url:
            self._cleanup_kept_url_audio()

        # Remember last audio source(s) for optional translation.
        self._last_audio_paths = list(paths)

        def _do_start() -> None:
            # Update drop zone placeholder with selected file name(s)
            try:
                self.drop_widget.set_selected_paths(paths)
            except Exception:
                pass
            self.text_edit.clear()
            self.terminal.clear()
            self.copy_button.setEnabled(False)
            self.translate_button.setEnabled(False)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

            language = self.audio_lang_combo.currentData()
            model_size = self.model_combo.currentData() or DEFAULT_MODEL_SIZE

            self._worker = TranscriptionWorker(paths, language, model_size, models_dir=str(self.models_dir), translate_to_en=translate_to_en)
            self._thread = QThread()
            self._worker.moveToThread(self._thread)

            self._thread.started.connect(self._worker.run)
            self._worker.progress.connect(self._update_progress)
            self._worker.finished.connect(self._transcription_finished)
            self._worker.error.connect(self._transcription_error)
            self._worker.log_line.connect(self._append_terminal_line)
            self._worker.finished.connect(self._cleanup_worker)
            self._worker.error.connect(self._cleanup_worker)

            self.cancel_action.setEnabled(True)
            self._thread.start()
            if translate_to_en:
                self._start_status_animation(ui_tr(self.ui_lang, "status_translating"))
            else:
                self._start_status_animation(ui_tr(self.ui_lang, "status_transcribing"))

            self._gui_log(
                f"[DEBUG] GUI: Started transcription for paths: {[str(p) for p in paths]} "
                f"with language={language} model={model_size}"
            )

        # Ensure the engine is installed before starting a transcription.
        self._ensure_engine_and_then(_do_start)
    def _cancel_transcription(self) -> None:
        """Cancel an ongoing transcription or URL download."""
        if self._url_worker is not None:
            self._url_worker.request_stop()
            self._start_status_animation(ui_tr(self.ui_lang, "status_cancelling_download"))
            return
        if self._worker is not None:
            self._worker.request_stop()
            self._start_status_animation(ui_tr(self.ui_lang, "status_cancelling"))

    def _cleanup_worker(self) -> None:
        """Clean up thread and worker after finishing or error."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self.cancel_action.setEnabled(False)
        self.progress_bar.setVisible(False)
        self._stop_status_animation(ui_tr(self.ui_lang, "status_ready"))
        self._cleanup_temp_downloads()

    def _gui_log(self, line: str) -> None:
        # Mirror GUI-side logs to the optional terminal and to stdout.
        try:
            self._append_terminal_line(line)
        except Exception:
            pass
        print(line, flush=True)

    def _set_terminal_visible(self, visible: bool) -> None:
        self.terminal.setVisible(visible)

    def _persist_transcribe_language_setting(self) -> None:
        code = self.audio_lang_combo.currentData() or "auto"
        code = (str(code).lower().strip() or "auto")
        if code not in ("auto", "en", "es", "zh"):
            code = "auto"
        self.transcribe_lang = code
        try:
            self.settings.setValue("transcribe_lang", code)
        except Exception:
            pass

    def _persist_model_setting(self) -> None:
        key = self.model_combo.currentData() or DEFAULT_MODEL_SIZE
        key = (str(key).strip() or DEFAULT_MODEL_SIZE)
        self.model_key = key
        try:
            self.settings.setValue("model_key", key)
        except Exception:
            pass


    def _append_terminal_line(self, line: str) -> None:
        # QPlainTextEdit is efficient for large logs and provides a terminal-like feel.
        self.terminal.appendPlainText(line)

        # Make long operations clearer via the status bar.
        # This only changes the UI message, not the processing logic.
        try:
            l = (line or "")
            if "Extracting audio with command" in l:
                self._start_status_animation(ui_tr(self.ui_lang, "status_extracting_audio"))
            elif "Starting transcription" in l:
                self._start_status_animation(ui_tr(self.ui_lang, "status_transcribing"))
            elif "Downloading" in l and "model" in l.lower():
                self._start_status_animation(ui_tr(self.ui_lang, "status_downloading_model"))
            elif "Detected repetition loop" in l:
                self._start_status_animation(ui_tr(self.ui_lang, "status_retrying"))
        except Exception:
            pass

    def _update_progress(self, value: int) -> None:
        self.progress_bar.setValue(value)

    def _transcription_finished(self, entry: TranscriptEntry) -> None:
        """Handle successful transcription results."""
        # Display transcript in the editor (overwriting previous if multiple files)
        self.text_edit.setPlainText(entry.transcript)
        self.copy_button.setEnabled(True)
        try:
            self.translate_button.setEnabled(bool(getattr(self, '_last_audio_paths', None)))
        except Exception:
            pass
        # Append to history and save
        self.history_entries.append(entry)
        save_history(self.history_path, self.history_entries)
        # Update history list UI
        self._populate_history_list()
        # Switch to transcribe tab to show output
        self.tabs.setCurrentIndex(0)
        # Debug output
        self._gui_log(f"[DEBUG] GUI: Transcription finished for {entry.filename}")

    def _transcription_error(self, message: str) -> None:
        # Debug output
        self._gui_log(f"[ERROR] GUI: {message}")
        QMessageBox.critical(self, ui_tr(self.ui_lang, "err_transcription"), message)


    def _translate_last_audio(self) -> None:
        """Translate the last transcribed audio to English (Whisper translate task)."""
        if self._thread is not None or self._url_thread is not None:
            QMessageBox.warning(self, ui_tr(self.ui_lang, "msg_busy_title"), ui_tr(self.ui_lang, "msg_busy_body"))
            return

        if not getattr(self, "_last_audio_paths", None):
            QMessageBox.information(self, ui_tr(self.ui_lang, "msg_translate_title"), ui_tr(self.ui_lang, "err_no_audio_to_translate"))
            return

        paths = [Path(p) for p in self._last_audio_paths if Path(p).exists()]
        if not paths:
            QMessageBox.information(self, ui_tr(self.ui_lang, "msg_translate_title"), ui_tr(self.ui_lang, "err_no_audio_to_translate"))
            return

        # Keep URL temp audio (if any) until the user starts a new job.
        self._start_transcription(paths, translate_to_en=True, preserve_kept_url=True)

    def _copy_current_transcript(self) -> None:
        """Copy current transcript to clipboard."""
        QGuiApplication.clipboard().setText(self.text_edit.toPlainText())

    def _populate_history_list(self) -> None:
        """Refresh the QListWidget with the current history entries."""
        self.history_list.clear()
        for idx, entry in enumerate(self.history_entries):
            dt = datetime.fromtimestamp(entry.timestamp)
            display_text = f"{entry.filename}  ({dt.strftime('%Y-%m-%d %H:%M')})"
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.history_list.addItem(item)

    def _display_history_entry(self, row: int) -> None:
        """Show the transcript of the selected history item."""
        if 0 <= row < len(self.history_entries):
            entry = self.history_entries[row]
            self.history_text_edit.setPlainText(entry.transcript)
            self.copy_history_button.setEnabled(True)
            self.delete_history_button.setEnabled(True)
        else:
            self.history_text_edit.clear()
            self.copy_history_button.setEnabled(False)
            self.delete_history_button.setEnabled(False)

    def _copy_selected_history(self) -> None:
        row = self.history_list.currentRow()
        if 0 <= row < len(self.history_entries):
            entry = self.history_entries[row]
            QGuiApplication.clipboard().setText(entry.transcript)

    def _delete_selected_history(self) -> None:
        row = self.history_list.currentRow()
        if 0 <= row < len(self.history_entries):
            entry = self.history_entries.pop(row)
            # Update storage
            save_history(self.history_path, self.history_entries)
            # Refresh list and selection
            self._populate_history_list()
            self.history_text_edit.clear()
            self.copy_history_button.setEnabled(False)
            self.delete_history_button.setEnabled(False)



    def closeEvent(self, event) -> None:
        # Ensure settings are persisted
        try:
            self.settings.setValue("youtube_cookies_file", self.youtube_cookies_file or "")
            self.settings.setValue("bilibili_cookies_file", self.bilibili_cookies_file or "")
            self.settings.setValue("models_dir", str(self.models_dir))
            self.settings.setValue("model_key", getattr(self, "model_key", "") or (self.model_combo.currentData() or DEFAULT_MODEL_SIZE))
            self.settings.sync()
        except Exception:
            pass
        try:
            # Always remove the kept URL audio on exit to avoid leaving temp folders behind.
            self._cleanup_kept_url_audio()
        except Exception:
            pass
        try:
            self._cleanup_temp_downloads()
        except Exception:
            pass
        super().closeEvent(event)


###############################################################################
# Application entry point
###############################################################################


def apply_dark_theme(app: QApplication) -> None:
    """Apply a dark Fusion theme to the application.

    Qt's native style on Windows does not fully support dark mode for widgets.
    By switching to the ``Fusion`` style and providing a dark palette, we get a
    consistent dark appearance across platforms.  If the user prefers to use
    system colors, this function can be omitted.
    """
    app.setStyle('Fusion')
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(142, 45, 197))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)
    # Optional: set a font size appropriate for readability
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)





def main() -> int:
    _ensure_std_streams()
    # Create the Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Voxora")
    app.setOrganizationName("neura")
    # App icon (assets/img/icon.ico)
    try:
        icon_path = resource_path(os.path.join("assets", "img", "icon.ico"))
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
    except Exception:
        pass
    # Apply dark theme (comment out the next line if you prefer system theme)
    apply_dark_theme(app)
    # Create and show main window
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())

r"""
README / Windows Setup
----------------------

To run this application on Windows you need to create a Python virtual
environment, install the required packages and ensure that ffmpeg is available
on your ``PATH``.  The following commands assume you have Python 3.10+ installed.

1. Open a command prompt and navigate to the directory containing this script.

2. Create a virtual environment (replace ``transcriber-env`` with a name you
   prefer):

   ```
   py -3 -m venv transcriber-env
   transcriber-env\Scripts\activate
   ````

3. Upgrade pip and install dependencies:

   ````
   python -m pip install --upgrade pip
   # Install PyQt6 for the GUI
   pip install pyqt6
   # Install pywhispercpp.  This package provides Python bindings to
   # whisper.cpp and will download the chosen Whisper model on first run.
   pip install pywhispercpp
   # Ensure ffmpeg is available: either install from https://ffmpeg.org/download.html
   # and add it to your PATH, or use a Python binding such as ffmpeg-python if you
   # prefer (you'll need to adjust the code accordingly).
   ````

4. (Optional) Whisper.cpp can be compiled with CUDA or other backends for
   hardware acceleration.  See the `pywhispercpp` documentation if you wish
   to build the library with GPU support.  By default the CPU‑only build is
   sufficient for moderate workloads.

5. Run the application:

   ````
   python transcriber_app.py
   ````

When the program starts you will see varias pestañas: "Archivo" for performing
new transcriptions and "Historial" to review past results.  Drag an audio or
video file into the drop zone or choose a file by clicking the drag-and-drop area
to begin.  The language selector defaults to automatic detection but you can
manually select Spanish, English or Chinese if needed.  While transcription
runs a progress bar will show the approximate completion percentage; you can
cancel at any time.  After completion the transcript appears in the editor
where you can correct mistakes and copy the result.  Each finished transcript
is saved in a JSON file (``transcripts_history.json``) so you can revisit it
later from the Historial tab.

Additional enhancements you might consider implementing include:

* Offering a choice of Whisper model sizes (tiny, base, medium, large) via a
  configuration dialog.  Larger models yield higher accuracy but require
  significantly more memory and computation.
* Enabling translation tasks, e.g. transcribing non‑English audio directly
  to English text, by passing ``translate=True`` when creating the
  `pywhispercpp` model or in the `transcribe()` call.
* Exporting transcripts to plain text or subtitle (SRT) files directly from
  the application.
* Displaying timestamps or speaker diarization to facilitate editing of long
  recordings.
* Integrating a spell checker to further refine output quality.
"""