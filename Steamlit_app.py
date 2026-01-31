import streamlit as st
import pandas as pd
# from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import os
from openai import OpenAI
import sounddevice as sd
import numpy as np
import wave
import json
import csv
from pathlib import Path
from queue import Queue, Empty, Full
import threading
import time
from datetime import datetime
import contextlib

# ----------------------------- APP & SETTINGS PATHS
st.set_page_config(page_title="TTS Dataset Manager", layout="wide")  # must be first Streamlit call
APP_DIR = Path.home() / ".tts_dataset_manager"
APP_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = APP_DIR / "settings.json"
RECORDS_INDEX_FILE = APP_DIR / "records_index.json"  # (row_idx, lang) -> list of takes

def _load_settings() -> dict:
    """Load persistent settings (language, dirs, last index)."""
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "lang": "EN",
        "base_dir": os.getcwd(),
        "csv_base_dir": "",
        "recordings_dir": "recordings",
        "last_index": 0,
    }

def _save_settings(d: dict) -> None:
    """Save persistent settings atomically."""
    try:
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(SETTINGS_FILE)
    except Exception:
        pass

SET = _load_settings()

# ----------------------------- Records index (takes)
def _load_records_index() -> dict:
    """Load takes index: { "<row_idx>|<lang>": [ {path,label,ts} ] }"""
    if RECORDS_INDEX_FILE.exists():
        try:
            return json.loads(RECORDS_INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _save_records_index(d: dict) -> None:
    try:
        tmp = RECORDS_INDEX_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(RECORDS_INDEX_FILE)
    except Exception:
        pass

RECORDS_INDEX = _load_records_index()

# ----------------------------- I18N (single API = tr())
def load_language(lang_code="EN"):
    try:
        with open(f"i18n/{lang_code}.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Missing language file for '{lang_code}', using English fallback.")
        try:
            with open("i18n/EN.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            st.error("English fallback file missing. Please add 'i18n/EN.json'.")
            return {}

# INIT T BEFORE first tr()
T = load_language(SET.get("lang", "EN"))

def tr(key: str, default: str | None = None, **kwargs) -> str:
    """Translate by key, with optional default and .format(**kwargs)."""
    raw = T.get(key, default if default is not None else key)
    try:
        return str(raw).format(**kwargs)
    except Exception:
        return str(raw)

lang_codes = ["HR", "DE", "EN", "FR", "PL"]
try:
    lang_index = lang_codes.index(SET.get("lang", "EN"))
except ValueError:
    lang_index = 2
lang_choice = st.sidebar.selectbox("🌐 " + tr("language", default="Language"), lang_codes, index=lang_index)
if lang_choice != SET.get("lang"):
    SET["lang"] = lang_choice
    _save_settings(SET)

T = load_language(lang_choice)

st.title(tr("title", default="TTS Dataset Manager"))

# ----------------------------- PATH HELPERS
def resolve_audio_path(p: str, csv_base_dir: str | None = None, base_dir: str | None = None) -> str | None:
    """Resolve audio path; return absolute path if exists, else None."""
    if not isinstance(p, str):
        return None
    p = p.strip()
    if not p:
        return None

    # expand ~
    p1 = os.path.expanduser(p)

    # 1) absolute path
    if os.path.isabs(p1) and os.path.isfile(p1):
        return os.path.normpath(p1)

    # try helper to join and check
    def _try_join(dir_candidate: str | None, rel: str) -> str | None:
        if not dir_candidate:
            return None
        candidate = os.path.join(dir_candidate, rel)
        return os.path.normpath(candidate) if os.path.isfile(candidate) else None

    # 2) relative to csv base dir
    found = _try_join(csv_base_dir, p1) or _try_join(base_dir, p1)
    if found:
        return found

    # 3) relative to script directory
    try:
        script_dir = Path(__file__).resolve().parent
        candidate2 = script_dir / p1
        if candidate2.is_file():
            return str(candidate2)
    except NameError:
        # __file__ may be undefined in some environments
        pass

    # 4) relative to current working directory
    candidate3 = Path.cwd() / p1
    if candidate3.is_file():
        return str(candidate3.resolve())

    return None

def _to_relative_if_possible(abs_path: str, csv_base_dir: str | None) -> str:
    """Return path relative to csv_base_dir if possible; otherwise filename."""
    if not abs_path:
        return abs_path
    try:
        if csv_base_dir:
            base = Path(csv_base_dir).resolve()
            p = Path(abs_path).resolve()
            if str(p).startswith(str(base)):
                return str(p.relative_to(base)).replace("\\", "/")
    except Exception:
        pass
    # as fallback keep original name (relative) or filename
    try:
        return str(Path(abs_path).name)
    except Exception:
        return abs_path

def _compute_save_target_for_record(row, sr: int, csv_base_dir: str | None, base_dir: str | None,
                                    recordings_root: str, lang_code: str) -> Path:
    """
    Keep same subdirectory structure as originals + separate per language:
    <recordings_root>/<LANG_CODE>/<RELATIVE_ORIG_DIR>/recorded_entry{row}_{sr}Hz_{ts}.wav
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recorded_entry{row.name}_{sr}Hz_{ts}.wav"

    original_rel = None
    orig_val = row.get("audio_path") if hasattr(row, "get") else (row["audio_path"] if "audio_path" in row else None)
    if isinstance(orig_val, str) and orig_val.strip():
        p = Path(orig_val)
        if not p.is_absolute():
            original_rel = p.parent
        else:
            try:
                if csv_base_dir and str(p.resolve()).startswith(str(Path(csv_base_dir).resolve())):
                    original_rel = p.resolve().relative_to(Path(csv_base_dir).resolve()).parent
                elif base_dir and str(p.resolve()).startswith(str(Path(base_dir).resolve())):
                    original_rel = p.resolve().relative_to(Path(base_dir).resolve()).parent
            except Exception:
                original_rel = None

    # root is <recordings_root>/<LANG_CODE>
    lang_root = Path(recordings_root) / lang_code
    subdir = (lang_root / original_rel) if original_rel else lang_root
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / filename

# ----------------------------- CACHING LAYERS
@st.cache_data(show_spinner=False)
def build_search_index(df_in: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a lowercased transcript column for fast search."""
    df = df_in.copy()
    if "transcript" in df.columns:
        df["transcript_lc"] = df["transcript"].astype(str).str.lower()
    return df

@st.cache_data(show_spinner=False)
def cached_waveform_png_bytes(audio_path: str, mtime: float) -> bytes:
    """Render waveform to PNG bytes (cache key uses mtime to bust cache on change)."""
    y, sr = librosa.load(audio_path, sr=None)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="cyan")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return buf.getvalue()

@st.cache_resource(show_spinner=False)
def cached_list_input_devices():
    """List audio input devices once."""
    try:
        devices = sd.query_devices()
        return [(i, d['name']) for i, d in enumerate(devices) if d.get('max_input_channels', 0) > 0]
    except Exception:
        return []

# ----------------------------- SESSION STATE (RECORDER)
def _init_state():
    """Initialize state for translation + recorder."""
    ss = st.session_state
    # Translation cache: {(row_idx, target_lang): "translation"}
    ss.setdefault("translations", {})
    ss.setdefault("rec_active", False)          # is recording on
    ss.setdefault("rec_thread", None)           # background consumer thread
    ss.setdefault("rec_stream", None)           # sounddevice.InputStream
    ss.setdefault("rec_channels", 1)
    ss.setdefault("rec_sr", 22050)
    ss.setdefault("rec_started_at", None)       # timestamp for elapsed
    ss.setdefault("rec_last_file", None)        # saved wav path
    ss.setdefault("rec_row_index", int(SET.get("last_index", 0)))
    ss.setdefault("rec_target_lang", "English") # last chosen target lang
    ss.setdefault("rec_device_index", None)
    # Persistent recorder context that survives reruns
    if "rec_ctx" not in ss:
        ss["rec_ctx"] = {
            # queue and buffer will be reused across reruns
            "queue": None,                   # type: Queue | None
            "buffer": [],                    # type: list[np.ndarray] (int16)
            "lock": threading.Lock(),        # single persistent lock
            "active": False,                 # recording flag
            "cb_count": 0,                   # callback counter
            "channels": 1,                   # cached channels during recording
        }
    # Search cache (avoid recompute for each rerun)
    ss.setdefault("last_query", "")
    ss.setdefault("last_results_idx", None)  # numpy array or list of indices
    ss.setdefault("page_size", 50)
    ss.setdefault("page_num", 1)  # 1-based
    # DataFrame holder
    ss.setdefault("df", None)
    ss.setdefault("df_indexed", None)

_init_state()

# Aliases to persistent recorder context (same objects every rerun)
REC_CTX = st.session_state["rec_ctx"]

# ----------------------------- RECORDING: CALLBACK/WORKER
def _raw_callback(indata, frames, time_info, status):
    """PortAudio callback — push raw bytes to queue (no Streamlit calls here)."""
    REC_CTX["cb_count"] += 1
    q = REC_CTX["queue"]
    if q is not None:
        try:
            q.put_nowait(bytes(indata))
        except Full:
            pass

def _consumer_loop():
    """Convert raw bytes to int16 frames and append to persistent buffer."""
    sample_width_bytes = 2
    channels = REC_CTX["channels"]
    frame_bytes = channels * sample_width_bytes
    pending = bytearray()
    while REC_CTX["active"]:
        q = REC_CTX["queue"]
        if q is None:
            time.sleep(0.05)
            continue
        try:
            chunk = q.get(timeout=0.2)
            pending.extend(chunk)
            usable = len(pending) - (len(pending) % frame_bytes)
            if usable > 0:
                data = np.frombuffer(pending[:usable], dtype=np.int16).reshape(-1, channels)
                with REC_CTX["lock"]:
                    REC_CTX["buffer"].append(data.copy())
                del pending[:usable]
        except Empty:
            continue

def start_recording(sr: int, channels: int = 1):
    """Start recording and background consumer."""
    ss = st.session_state
    if ss.get("rec_active", False):
        return

    # reset counters and buffer but DO NOT replace objects
    if REC_CTX["queue"] is None:
        REC_CTX["queue"] = Queue(maxsize=256)
    else:
        # drain old queue
        try:
            while True:
                REC_CTX["queue"].get_nowait()
        except Exception:
            pass
    with REC_CTX["lock"]:
        REC_CTX["buffer"].clear()
    REC_CTX["active"] = True
    REC_CTX["cb_count"] = 0
    REC_CTX["channels"] = channels

    ss["rec_sr"] = sr
    ss["rec_channels"] = channels
    ss["rec_started_at"] = time.time()
    ss["rec_active"] = True

    device_idx = ss.get("rec_device_index", None)
    if device_idx is not None:
        cur_out = None
        try:
            cur_out = sd.default.device[1]
        except Exception:
            pass
        sd.default.device = (device_idx, cur_out)

    stream = sd.RawInputStream(
        samplerate=sr,
        channels=channels,
        dtype="int16",
        callback=_raw_callback,
        device=device_idx,
        blocksize=2048,
        latency='low'
    )
    stream.start()
    ss["rec_stream"] = stream

    th = threading.Thread(target=_consumer_loop, daemon=True)
    th.start()
    ss["rec_thread"] = th

def _add_take_to_index(row_index: int, lang_code: str, file_path: str, label: str | None = None):
    """Append take metadata to records index and persist."""
    key = f"{row_index}|{lang_code}"
    lst = RECORDS_INDEX.get(key, [])
    lst.append({
        "path": str(Path(file_path).resolve()),
        "label": label or f"take #{len(lst)+1}",
        "ts": datetime.now().isoformat(timespec="seconds")
    })
    RECORDS_INDEX[key] = lst
    _save_records_index(RECORDS_INDEX)

# ---- Trim silence helper (to remove blank start/end in takes)
def _trim_silence_int16(stacked_int16: np.ndarray, threshold: float = 0.005) -> np.ndarray:
    """
    Trim leading/trailing silence using simple amplitude threshold on mono mix.
    threshold ~ 0.005 ≈ -46 dBFS. Adjust if needed.
    """
    if stacked_int16.size == 0:
        return stacked_int16
    arr = stacked_int16
    mono = arr if arr.ndim == 1 else arr.mean(axis=1)
    mono_f = np.abs(mono.astype(np.float32)) / 32768.0
    mask = mono_f > threshold
    if not np.any(mask):
        return arr  # all silence
    start = int(np.argmax(mask))
    end = int(len(mask) - np.argmax(mask[::-1]))
    return arr[start:end, :] if arr.ndim == 2 else arr[start:end]

def stop_recording_and_save(row_index: int, sr: int, out_dir_fallback: str, lang_code: str) -> str | None:
    """Stop recording and write WAV; return absolute path."""
    ss = st.session_state
    if not ss.get("rec_active", False):
        return ss.get("rec_last_file")

    ss["rec_active"] = False
    REC_CTX["active"] = False

    stream: sd.RawInputStream = ss.get("rec_stream")
    try:
        if stream:
            stream.stop()
            stream.close()
    finally:
        ss["rec_stream"] = None

    th: threading.Thread = ss.get("rec_thread")
    if th and th.is_alive():
        th.join(timeout=1.5)
    ss["rec_thread"] = None

    # Choose save path to keep original structure
    df_local = st.session_state.get("df")
    row_local = df_local.iloc[int(row_index)] if (df_local is not None and len(df_local) > row_index) else pd.Series(dtype=object)
    target_path = _compute_save_target_for_record(
        row=row_local,
        sr=sr,
        csv_base_dir=SET.get("csv_base_dir") or None,
        base_dir=SET.get("base_dir") or None,
        recordings_root=out_dir_fallback,
        lang_code=lang_code
    )
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    with REC_CTX["lock"]:
        frames_list = list(REC_CTX["buffer"])
        REC_CTX["buffer"].clear()

    try:
        with contextlib.closing(wave.open(str(target_path), "wb")) as wf:
            wf.setnchannels(ss["rec_channels"])
            wf.setsampwidth(2)  # int16
            wf.setframerate(sr)
            if frames_list:
                data = np.vstack(frames_list)
                # --- trim leading/trailing silence to remove blank parts ---
                data = _trim_silence_int16(data, threshold=0.005)
                wf.writeframes(data.tobytes())
    except Exception as e:
        st.error(tr("parse_error_csv", default="Could not parse CSV: {error}", error=e))
        return None

    ss["rec_last_file"] = str(target_path.resolve())
    ss["rec_started_at"] = None

    if os.path.exists(ss["rec_last_file"]):
        st.toast(tr("record_saved", default="Saved new audio: {path}", path=ss["rec_last_file"]))
        # Add to takes index
        _add_take_to_index(int(row_index), lang_code, ss["rec_last_file"])
    else:
        st.error(tr("audio_not_found", default="Audio file not found"))

    if not frames_list:
        st.warning(tr("record_empty_warning", default="The recording is empty (0 frames). Check microphone permissions and/or the input device in the Settings tab."))
    return ss["rec_last_file"]


def elapsed_seconds() -> float:
    """Elapsed seconds since start of recording."""
    ss = st.session_state
    if ss["rec_started_at"] is None:
        return 0.0
    return max(0.0, time.time() - ss["rec_started_at"])

# ----------------------------- MIC PROBE (sync, main thread)
def probe_mic(duration_sec: float = 1.0, save_dir: str | Path | None = None) -> tuple[str | None, float, float]:
    ss = st.session_state
    if ss.get("rec_active", False):
        st.warning(tr("record_rehearsal_in_progress", default="It is not possible to do rehearsals while recording is in progress."))
        return None, 0.0, -np.inf

    sr = int(ss.get("rec_sr", 22050))
    ch = int(ss.get("rec_channels", 1))
    dev = ss.get("rec_device_index", None)

    # save short audio
    frames = int(duration_sec * sr)
    try:
        data = sd.rec(frames=frames, samplerate=sr, channels=ch, dtype="float32", device=dev)
        sd.wait()
    except Exception as e:
        st.error(tr("probe_mic_error", default=f"Probe mic error: {e}", error=e))
        return None, 0.0, -np.inf
    mono = np.mean(data, axis=1) if (data.ndim == 2 and data.shape[1] > 1) else data.reshape(-1)

    rms = float(np.sqrt(np.mean(mono**2)))
    peak = float(np.max(np.abs(mono)))
    # dBFS: 0 dBFS ≡ full scale (±1.0);
    peak_dbfs = 20.0 * np.log10(max(peak, 1e-12))

    # save WAV recordings
    if save_dir is None:
        save_dir = SET.get("recordings_dir") or "recordings"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"probe_{sr}Hz_{ts}.wav"
    fullpath = str((Path(save_dir) / filename).resolve())

    # float32 [-1,1] -> int16
    data_i16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
    # return channels ch>1
    if ch > 1:
        data_i16 = np.tile(data_i16[:, None], (1, ch))

    try:
        with contextlib.closing(wave.open(fullpath, "wb")) as wf:
            wf.setnchannels(ch)
            wf.setsampwidth(2)  # int16
            wf.setframerate(sr)
            wf.writeframes(data_i16.tobytes())
    except Exception as e:
        st.error(tr("error_save_wav", default="Error save WAV: {error}", error=e))
        return None, rms, peak_dbfs

    return fullpath, rms, peak_dbfs

# ----------------------------- SIDEBAR – PERSISTENT DIRS
default_base_dir = SET.get("base_dir", os.getcwd())
base_dir = st.sidebar.text_input("🎧 " + tr("base_audio_dir", default="Base audio dir"), value=default_base_dir)
if base_dir != SET.get("base_dir"):
    SET["base_dir"] = base_dir
    _save_settings(SET)

csv_base_dir = st.sidebar.text_input("📁 " + tr("csv_base_dir", default="CSV base dir (relative paths resolved against this)"),
                                     value=SET.get("csv_base_dir", ""))
if csv_base_dir != SET.get("csv_base_dir"):
    SET["csv_base_dir"] = csv_base_dir
    _save_settings(SET)

recordings_dir = st.sidebar.text_input("💾 " + tr("recordings_dir", default="Recordings dir"),
                                       value=SET.get("recordings_dir", "recordings"))
Path(recordings_dir).mkdir(parents=True, exist_ok=True)
if recordings_dir != SET.get("recordings_dir"):
    SET["recordings_dir"] = recordings_dir
    _save_settings(SET)

st.sidebar.caption(tr("save_csv", default="Saved as annotated_dataset.csv") + f" → `{Path(recordings_dir).resolve()}`")

# ----------------------------- TABS
tab_upload, tab_search, tab_translate, tab_annotate, tab_export, tab_settings = st.tabs(
    ["📥 " + tr("load_dataset", default="Load Dataset"),
     "🔎 " + tr("search_filter", default="Search and Filter"),
     "🌍 " + tr("translate", default="Translate Text"),
     "🏷️ " + tr("annotation", default="Annotation"),
     "📤 " + tr("export", default="Export Dataset"),
     "⚙️ " + tr("settings", default="Settings")]
)

# ----------------------------- DATA UPLOAD
with tab_upload:
    st.subheader(tr("load_dataset", default="Load Dataset"))
    uploaded_file = st.file_uploader(tr("upload", default="Upload CSV or JSON"), type=["csv", "json"], key="uploader")
    df = None
    if uploaded_file:
        # load dataset depending on file type
        if uploaded_file.name.endswith(".csv"):
            # detect delimiter
            try:
                sample = uploaded_file.read(2048).decode("utf-8", errors="ignore")
                uploaded_file.seek(0)
                sniffer = csv.Sniffer()
                detect_delim = sniffer.sniff(sample).delimiter
            except Exception:
                detect_delim = None

            # Internal definition (language-independent)
            common_delims = {
                "comma": ",",
                "pipe": "|",
                "semicolon": ";",
                "tab": "\t",
                "space": " "
            }

            # Build translated display names
            display_names = {
                "auto_detect": tr("auto_detect", default="Auto-detect"),
                **{k: tr(f"delim_{k}", default=k) for k in common_delims.keys()}
            }

            # Reverse map: from delimiter character to internal key
            reverse_map = {v: k for k, v in common_delims.items()}

            # Map detected delimiter to internal key or default
            default_key = reverse_map.get(detect_delim, "auto_detect")

            # Dropdown options (translated)
            options = [display_names[k] for k in ["auto_detect"] + list(common_delims.keys())]

            # Default index
            default_index = ["auto_detect"] + list(common_delims.keys())
            default_index = default_index.index(default_key)

            # Show selectbox
            delim_display = st.selectbox(
                tr("delimiter_select", default="Choose a common delimiter (or leave on Auto-detect)"),
                options,
                index=default_index
            )

            # Get back internal key from display text
            delim_key = [k for k, v in display_names.items() if v == delim_display][0]

            # Get actual character (if applicable)
            selected_delim = common_delims.get(delim_key, None)

            custom_delim = st.text_input(tr("delimiter_custom", default="Or enter custom delimiter (overrides choice)"), "")

            try:
                if custom_delim:
                    df = pd.read_csv(uploaded_file, sep=custom_delim)
                    st.info(tr("using_custom_delim", default="Using custom delimiter: '{delim}'", delim=custom_delim))

                elif delim_key != "auto_detect":
                    chosen_symbol = common_delims[delim_key]
                    st.info(tr("using_delim", default="Using delimiter: '{delim}'", delim=tr(f"delim_{delim_key}", default=delim_key)))
                    df = pd.read_csv(uploaded_file, sep=chosen_symbol)

                elif detect_delim:
                    df = pd.read_csv(uploaded_file, sep=detect_delim)
                    st.info(tr("using_auto_delim", default="Auto-detected delimiter: '{delim}'", delim=detect_delim))

                else:
                    st.error(tr("delim_warning", default="No delimiter detected or selected!"))
                    st.stop()

            except Exception as e:
                st.error(tr("parse_error_csv", default="Could not parse CSV: {error}", error=e))
                st.stop()
        else:
            try:
                df = pd.read_json(uploaded_file)
            except Exception as e:
                st.error(tr("parse_error_json", default="Could not parse JSON: {error}", error=e))
                st.stop()

        if df is not None and df.shape[1] == 1:
            st.warning(tr("dataset_warning", default="The dataset seems to have only one column. Try a different delimiter"))

        # ------------------------- FLEXIBLE COLUMN MAPPING
        st.subheader(tr("column_mapping", default="Column Mapping"))

        if df is not None:
            # default columns
            def_audio_col = next((c for c in df.columns if c.lower() in ["audio_path", "wav_filename", "file", "path"]), None)
            def_text_col = next((c for c in df.columns if c.lower() in ["transcript", "text", "sentence"]), None)
            def_speaker_col = next((c for c in df.columns if c.lower() in ["speaker", "spk", "spk_id"]), None)
            def_emotion_col = next((c for c in df.columns if c.lower() in ["emotion", "emo", "mood"]), None)
            def_category_col = next((c for c in df.columns if c.lower() in ["category", "tag", "class", "cat"]), None)

            # let user pick
            audio_col = st.selectbox(tr("col_audio", default="Select audio column"), ["<None>"] + list(df.columns),
                                     index=(list(df.columns).index(def_audio_col) + 1) if def_audio_col else 0)
            text_col = st.selectbox(tr("col_text", default="Select text/transcript column"), ["<None>"] + list(df.columns),
                                    index=(list(df.columns).index(def_text_col) + 1) if def_text_col else 0)
            speaker_col = st.selectbox(tr("col_speaker", default="Select speaker column"), ["<None>"] + list(df.columns),
                                       index=(list(df.columns).index(def_speaker_col) + 1) if def_speaker_col else 0)
            emotion_col = st.selectbox(tr("col_emotion", default="Select emotion column"), ["<None>"] + list(df.columns),
                                       index=(list(df.columns).index(def_emotion_col) + 1) if def_emotion_col else 0)
            category_col = st.selectbox(tr("col_category", default="Select category column"), ["<None>"] + list(df.columns),
                                        index=(list(df.columns).index(def_category_col) + 1) if def_category_col else 0)

            # normalize the column names internally
            rename_map = {}
            if audio_col != "<None>": rename_map[audio_col] = "audio_path"
            if text_col != "<None>": rename_map[text_col] = "transcript"
            if speaker_col != "<None>": rename_map[speaker_col] = "speaker"
            if emotion_col != "<None>": rename_map[emotion_col] = "emotion"
            if category_col != "<None>": rename_map[category_col] = "category"

            st.markdown(f"""
**{tr('column_mapping_applied', default='Column mapping applied:')}**

- **{tr('column_audio', default='audio_path')}** → `{audio_col}`
- **{tr('column_transcript', default='transcript')}** → `{text_col}`
- **{tr('column_speaker', default='speaker')}** → `{speaker_col}`
- **{tr('column_emotion', default='emotion')}** → `{emotion_col}`
- **{tr('column_category', default='category')}** → `{category_col}`
            """)

            # remove duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]

            # rename according to mapping
            df = df.rename(columns=rename_map)

            # verify no duplicates remain
            if df.columns.duplicated().any():
                st.error(f"{tr('duplicate_column', default='Duplicate column after mapping')}: {df.columns[df.columns.duplicated()].tolist()}")
                st.stop()

        # verify audio files
        if "audio_path" in (df.columns if df is not None else []):
            df["audio_resolved"] = df["audio_path"].apply(
                lambda x: resolve_audio_path(x, csv_base_dir=csv_base_dir or None, base_dir=base_dir or None)
            )
            df["audio_exists"] = df["audio_resolved"].apply(lambda p: isinstance(p, str) and os.path.isfile(p))

        # highlight missing audio in DataFrame (avoid slow styling on big data)
        if df is not None and "audio_exists" in df.columns:
            if len(df) <= 500:
                def highlight_missing(val):
                    return "background-color: salmon" if val is False else ""
                st.dataframe(df.style.map(highlight_missing, subset=['audio_exists']), width="stretch")
            else:
                st.caption(tr("large_dataset_no_style", default="Large dataset detected; showing first 500 rows without styling for performance."))
                st.dataframe(df.head(500), width="stretch")
            st.success(tr("loaded_entries", default="Loaded {n} entries", n=len(df)))
        elif df is not None:
            if len(df) > 2000:
                st.caption(tr("large_dataset_trimmed", default="Large dataset detected; showing first 500 rows."))
                st.dataframe(df.head(500), width="stretch")
            else:
                st.dataframe(df, width="stretch")
            st.success(tr("loaded_entries", default="Loaded {n} entries", n=len(df)))

        # Persist df in session for other tabs
        st.session_state["df"] = df
        st.session_state["df_indexed"] = build_search_index(df) if df is not None else None
        # reset search cache
        st.session_state["last_query"] = ""
        st.session_state["last_results_idx"] = None

# Ensure df for other tabs
df = st.session_state.get("df")
df_indexed = st.session_state.get("df_indexed")

# ----------------------------- SEARCH & PREVIEW
with tab_search:
    if df_indexed is not None and "transcript" in df_indexed.columns:
        st.subheader(tr("search_filter", default="Search & filter"))
        cols = st.columns([4,1])
        with cols[0]:
            query = st.text_input(tr("search_input", default="Search text"), st.session_state.get("last_query",""))
        with cols[1]:
            page_size = st.number_input(tr("page_number", default="Page"), min_value=10, max_value=500, value=st.session_state.get("page_size",50), step=10)
        
        # trigger search
        do_search = st.button(tr("search_button", default="Search"))

        # Run search only if query changed or user clicked Search
        if do_search or query != st.session_state.get("last_query","") or st.session_state.get("last_results_idx") is None:
            if query:
                mask = df_indexed["transcript_lc"].str.contains(query.lower(), na=False)
                results_idx = list(df_indexed[mask].index.values)
            else:
                results_idx = list(df_indexed.index.values)
            st.session_state["last_query"] = query
            st.session_state["last_results_idx"] = results_idx
            st.session_state["page_size"] = page_size
            st.session_state["page_num"] = 1  # reset to first page

        results_idx = st.session_state.get("last_results_idx") or []
        total = len(results_idx)
        if total == 0:
            st.info(tr("search_results", default="Found {n} matches", n=0))
        else:
            st.success(tr("search_results", default="Found {n} matches", n=total))

            # pagination
            page_size = st.session_state.get("page_size", 50)
            total_pages = max(1, (total + page_size - 1) // page_size)
            page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=st.session_state.get("page_num",1))
            st.session_state["page_num"] = int(page_num)

            start = (int(page_num)-1)*page_size
            end = min(start+page_size, total)
            page_slice = results_idx[start:end]
            page_df = df_indexed.loc[page_slice, ["transcript"]].copy()
            page_df["row_index"] = page_slice
            st.dataframe(page_df, width="stretch")

            # choose entry within page
            sel_idx = st.number_input(tr("entry_index", default="Entry index"),
                                      min_value=int(page_slice[0]) if page_slice else 0,
                                      max_value=int(page_slice[-1]) if page_slice else 0,
                                      value=int(page_slice[0]) if page_slice else 0)
            if page_slice:
                entry = df_indexed.loc[int(sel_idx)]
                st.markdown(f"**{tr('transcript_label','Transcript')}:** {entry['transcript']}")
                # audio playback of active (resolved)
                if "audio_resolved" in df_indexed.columns and pd.notna(entry.get("audio_resolved")):
                    ap = entry["audio_resolved"]
                    try:
                        with open(ap, "rb") as f:
                            st.audio(f.read(), format="audio/wav")
                    except FileNotFoundError:
                        st.error(f"{tr('audio_not_found','Audio not found')}: {ap}")
                else:
                    st.info(tr("audio_not_found", "Audio file not found."))

                # waveform (on-demand; cached by mtime)
                show_wf = st.checkbox(tr("show_waveform", default="Show waveform"), value=False, key=f"wf_{sel_idx}")
                if show_wf:
                    audio_path_use = entry.get("audio_resolved")
                    try:
                        if audio_path_use and os.path.isfile(audio_path_use):
                            mtime = os.path.getmtime(audio_path_use)
                            png_bytes = cached_waveform_png_bytes(audio_path_use, mtime)
                            st.image(io.BytesIO(png_bytes), width="stretch")
                    except Exception as e:
                        st.warning(f"{tr('waveform_fail','Waveform failed')}: {e}")

                # history of takes for this row + current UI language
                key_hr = f"{int(sel_idx)}|{lang_choice}"
                takes = RECORDS_INDEX.get(key_hr, [])
                st.markdown("### 🎧 " + tr("translation_label", default="Translation") + " / Takes")
                if takes:
                    for i, tinfo in enumerate(reversed(takes), 1):
                        pth = tinfo["path"]
                        label = tinfo.get("label", f"take #{i}")
                        st.write(f"• {label} — {tinfo.get('ts','')}")
                        if os.path.exists(pth):
                            with open(pth, "rb") as f:
                                st.audio(f.read(), format="audio/wav")
                            # Activate this take as current audio (links into df)
                            if st.button(tr("activate_take_button", default="Activate this take ({label})", label=label), key=f"act_{key_hr}_{i}"):
                                df.loc[int(sel_idx), "audio_path"] = pth
                                df.loc[int(sel_idx), "audio_resolved"] = pth
                                df.loc[int(sel_idx), "audio_exists"] = True
                                st.session_state["df"] = df
                                st.session_state["df_indexed"] = build_search_index(df)
                                st.success(tr("take_activated", default="Take activated for this entry."))
                else:
                    st.info(tr("no_previous_takes", default="No previous takes for this row/language."))
    else:
        st.info(tr("load_dataset_first", default="Load the dataset in the first tab."))

    # ----------------------------- TRANSLATION (OpenAI)
with tab_translate:
    df = st.session_state.get("df")
    if df is not None and "transcript" in df.columns:
        st.subheader(tr("translate", "Translate"))

        # safer client init (prevents KeyError if secret missing)
        client = None
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        except Exception:
            st.warning(tr("openai_key_missing", default="OPENAI_API_KEY is not configured in Streamlit secrets."))

        def translate_text(text, target_lang):
            """Translate and return output only."""
            prompt = f"Detect the language of the following text and translate it into {target_lang}. Reply with only the translation.\n\n{text}"
            if client is None:
                return tr("openai_key_not_configured", default="(OpenAI key is not configured)")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional translator. Always reply with the translated text only, without explanations or quotes"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()

        # Choose row and target language
        colA, colB = st.columns([2, 1])
        with colA:
            row_index = st.number_input(tr("entry_index", "Entry index"),
                                        min_value=0, max_value=len(df)-1, key="tr_row_idx",
                                        value=int(st.session_state.get("rec_row_index", 0)))
            base_text = df.loc[int(row_index), "transcript"]
        with colB:
            target_lang = st.selectbox(tr("target_language", "Target language"),
                                       ["Croatian", "English", "French", "German", "Polish"],
                                       index=1, key="tr_target_lang")

        st.session_state["rec_row_index"] = int(row_index)
        SET["last_index"] = int(row_index)
        _save_settings(SET)

        st.markdown(f"**{tr('transcript_label','Transcript')}:** {base_text}")

        # Translate button — store into session to avoid losing on rerun
        key_tuple = (int(row_index), target_lang)
        if st.button(tr("translate_button","Translate"), key="btn_translate_single"):
            translated = translate_text(base_text, target_lang)
            st.session_state["translations"][key_tuple] = translated
            st.session_state["rec_target_lang"] = target_lang
            st.toast(tr("translation_ready", default="Translation ready"), icon="🌍")

        # Always show last known translation for this row/lang (persisted)
        last_translation = st.session_state["translations"].get(key_tuple)
        if last_translation:
            with st.expander(f"{tr('translation_header','Translation')} – {target_lang}", expanded=True):
                st.write(last_translation)

        st.divider()
        st.markdown("### 🎙️ " + tr("record_audio", "Record Audio"))

        # Recorder controls
        colL, colR = st.columns([3, 2])
        with colL:
            sr_choice = st.slider(
                tr("record_samplerate","Sample rate"),
                8000, 48000,
                st.session_state.get("rec_sr", 22050),
                step=1000,
                key="rec_sr_slider"
            )
            st.session_state["rec_sr"] = sr_choice

            if st.session_state["rec_active"]:
                # compute chunks & RMS only while recording
                with REC_CTX["lock"]:
                    chunks = len(REC_CTX["buffer"])
                    if chunks:
                        last_block = REC_CTX["buffer"][-1].astype(np.float32) / 32768.0
                        rms = float(np.sqrt(np.mean(last_block**2)))
                    else:
                        rms = 0.0
                st.info(tr("recording_status_live",
                    default="🔴 Recording... {elapsed:.1f}s @ {sr} Hz  •  chunks: {chunks}  •  cb: {cb_count}  •  RMS: {rms:.4f}",
                    elapsed=elapsed_seconds(), sr=sr_choice, chunks=chunks,
                    cb_count=REC_CTX["cb_count"], rms=rms))
            else:
                last = st.session_state.get("rec_last_file")
                if last:
                    st.success(tr("last_recording", default="✅ Last recording: `{last}`", last=last))

        with colR:
            c1, c2, c3 = st.columns(3)
            with c1:
                if not st.session_state["rec_active"]:
                    if st.button(tr("record_start_button", default="🎙️ Start"), key="btn_rec_start"):
                        try:
                            start_recording(sr=st.session_state["rec_sr"], channels=1)
                            st.toast(tr("record_started_toast", default="Recording started"), icon="🎙️")
                        except Exception as e:
                            st.error(tr("audio_device_error", default="Audio device error: {error}", error=e))
            with c2:
                if st.session_state["rec_active"]:
                    if st.button(tr("record_stop_button", default="⏹️ Stop"), key="btn_rec_stop"):
                        path = stop_recording_and_save(
                            st.session_state["rec_row_index"],
                            st.session_state["rec_sr"],
                            recordings_dir,
                            lang_choice
                        )
                        if path:
                            st.toast(tr("record_saved_toast", default="Recording saved (see below)"), icon="💾")
            with c3:
                if st.button(tr("record_reset_button", default="🗑️ Reset"), key="btn_rec_reset"):
                    st.session_state["rec_last_file"] = None
                    st.session_state["rec_started_at"] = None
                    with REC_CTX["lock"]:
                        REC_CTX["buffer"].clear()
                    st.toast(tr("record_cleared_toast", default="Cleared"), icon="🧹")

        # If we have a saved file, show player + link-to-row
        saved_path = st.session_state.get("rec_last_file")
        if saved_path:
            st.markdown(tr("recorded_in", default="📁 Recorded in: `{path}`", path=saved_path))
            if os.path.exists(saved_path):
                with open(saved_path, "rb") as f:
                    st.audio(f.read(), format="audio/wav")
            else:
                st.error(tr("record_file_missing",
                default="The file was not found on the disk even though a save was attempted (check write permissions)."))

            if os.path.exists(saved_path):
                if st.button(tr("record_replace","Replace entry {entry}").format(entry=int(row_index)), key="btn_link_last"):
                    df.loc[int(row_index), "audio_path"] = saved_path
                    resolved = resolve_audio_path(saved_path, csv_base_dir=csv_base_dir or None, base_dir=base_dir or None) or saved_path
                    df.loc[int(row_index), "audio_resolved"] = resolved
                    df.loc[int(row_index), "audio_exists"] = bool(resolved and os.path.isfile(resolved))
                    st.success(tr("record_linked","Linked to entry {entry}").format(entry=int(row_index)))
                    st.session_state["df"] = df
                    st.session_state["df_indexed"] = build_search_index(df)

        # Takes history for this row/lang
        st.markdown("### " + tr("takes_history", default="📜 Takes / History"))
        key_hr = f"{int(st.session_state['rec_row_index'])}|{lang_choice}"
        takes = RECORDS_INDEX.get(key_hr, [])
        if takes:
            for i, tinfo in enumerate(reversed(takes), 1):
                pth = tinfo["path"]
                label = tinfo.get("label", f"take #{i}")
                st.write(f"• {label} — {tinfo.get('ts','')}")
                if os.path.exists(pth):
                    with open(pth, "rb") as f:
                        st.audio(f.read(), format="audio/wav")
                    c1, c2 = st.columns([1,1])
                    with c1:
                        if st.button(tr("activate_take", default=f"Activate ({label})"), key=f"act_tr_{key_hr}_{i}"):
                            df.loc[int(st.session_state["rec_row_index"]), "audio_path"] = pth
                            df.loc[int(st.session_state["rec_row_index"]), "audio_resolved"] = pth
                            df.loc[int(st.session_state["rec_row_index"]), "audio_exists"] = True
                            st.session_state["df"] = df
                            st.session_state["df_indexed"] = build_search_index(df)
                            st.success(tr("activated", default="Activated."))
                    with c2:
                        new_label = st.text_input(tr("label_input", default="Label"), value=label, key=f"lbl_{key_hr}_{i}")
                        if st.button(tr("rename_button", default="Rename"), key=f"ren_{key_hr}_{i}"):
                            # rename in index
                            idx_key = key_hr
                            lst = RECORDS_INDEX.get(idx_key, [])
                            # reverse mapping index
                            pos = len(lst) - i  # because we iterated reversed
                            if 0 <= pos < len(lst):
                                lst[pos]["label"] = new_label or label
                                RECORDS_INDEX[idx_key] = lst
                                _save_records_index(RECORDS_INDEX)
                                st.toast(tr("label_renamed", default="Renamed label."), icon="✏️")
        else:
            st.info(tr("no_previous_takes", default="There are no previous recordings for this row/language."))

    else:
        st.info(tr("load_dataset_first", default="Load the dataset in the first tab."))

# ----------------------------- ANNOTATION
with tab_annotate:
    st.subheader(tr("annotation", "Annotation"))
    if df is not None:
        row_index = st.number_input(tr("annotate_row","Row to annotate"), min_value=0, max_value=len(df)-1,
                                    value=int(SET.get("last_index",0)))
        exist_speaker = df.loc[row_index, "speaker"] if "speaker" in df.columns else None
        exist_emotion = df.loc[row_index, "emotion"] if "emotion" in df.columns else None
        exist_category = df.loc[row_index, "category"] if "category" in df.columns else None

        if exist_speaker or exist_emotion or exist_category:
            st.warning(
                tr(
                    "annotation_warning",
                    "Row {row} has existing annotations: Speaker = '{speaker}', Emotion = '{emotion}', Category = '{category}'"
                ).format(row=row_index, speaker=exist_speaker, emotion=exist_emotion, category=exist_category)
            )

        speaker = st.text_input(tr("speaker_name","Speaker"), exist_speaker if exist_speaker else "")
        # Emotion options (internal keys)
        emotion_keys = ["neutral", "happy", "sad", "angry"]

        # Translated display names
        emotion_display = [tr(f"emotion_{e}", e) for e in emotion_keys]

        # Determine default index
        default_index = emotion_keys.index(exist_emotion) if exist_emotion in emotion_keys else 0

        # Selectbox with translated options
        emotion_display_selected = st.selectbox(
            tr("emotion", "Emotion"),
            emotion_display,
            index=default_index
        )

        # Map back to internal key
        emotion = emotion_keys[emotion_display.index(emotion_display_selected)]
        category = st.text_input(tr("category","Category"), exist_category if exist_category else "")

        if st.button(tr("annotate_button","Save annotation")):
            df.loc[row_index, "speaker"] = speaker
            df.loc[row_index, "emotion"] = emotion
            df.loc[row_index, "category"] = category
            st.success(tr("annotated","Annotated row {row}").format(row=row_index))
            st.session_state["df"] = df
            st.session_state["df_indexed"] = build_search_index(df)
            SET["last_index"] = int(row_index)
            _save_settings(SET)
            st.dataframe(df.head(), width="stretch")
    else:
        st.info(tr("load_dataset_first", default="Load the dataset in the first tab."))

# ----------------------------- EXPORT
with tab_export:
    st.subheader(tr("export","Export"))
    if df is not None:
        export_format = st.radio(tr("choose_format","Choose format"), ["CSV", "JSON"])
        dataset_name = st.text_input(
            tr("dataset_name", default="Dataset name"),
            value="annotated_dataset",
            help=tr("dataset_name_help", default="Enter a name for the exported file (no extension needed).")
        )


        # Build export view with audio_path relative to csv_base_dir (when possible)
        def build_export_df(df_in: pd.DataFrame) -> pd.DataFrame:
            df_out = df_in.copy()
            if "audio_resolved" in df_out.columns:
                df_out["audio_path"] = df_out["audio_resolved"].fillna(df_out.get("audio_path"))
            if "audio_path" in df_out.columns:
                df_out["audio_path"] = df_out["audio_path"].apply(
                    lambda p: _to_relative_if_possible(p, SET.get("csv_base_dir") or None) if isinstance(p, str) else p
                )
            # drop helper columns
            for col in ["audio_resolved", "audio_exists", "transcript_lc"]:
                if col in df_out.columns:
                    df_out = df_out.drop(columns=[col])
            return df_out

        if st.button(tr("export_button","Export now")):
            dataset_name = dataset_name.strip() or "annotated_dataset"
            edf = build_export_df(df)
            if export_format == "CSV":
                file_path = Path(f"{dataset_name}.csv")
                edf.to_csv(file_path, index=False)
            else:
                file_path = Path(f"{dataset_name}.json")
                edf.to_json(file_path, orient="records", indent=2, force_ascii=False)
            
            st.success(tr("export_success", f"Exported as {file_path.name}"))
            st.caption(f"📁 {file_path.resolve()}")
    else:
        st.info(tr("load_dataset_first", default="Load the dataset in the first tab."))

# ----------------------------- SETTINGS / DIAGNOSTICS
with tab_settings:
    st.subheader(tr("audio_input_device", default="Audio input device"))
    try:
        input_devices = cached_list_input_devices()
        if not input_devices:
            st.warning(tr("no_input_device", default="No input audio device was found."))
        else:
            names = [f"[{i}] {name}" for i, name in input_devices]
            pre_idx = st.session_state.get("rec_device_index", input_devices[0][0])
            try:
                default_idx = [k for k,(dev_i,_) in enumerate(input_devices) if dev_i == pre_idx][0]
            except Exception:
                default_idx = 0
            sel = st.selectbox(tr("input_device", default="Input device"), options=names, index=default_idx)
            chosen_idx = int(sel.split("]")[0].strip("["))
            st.session_state["rec_device_index"] = chosen_idx
            st.caption(tr("selected_device_index", default="Selected device index: {index}").format(index=chosen_idx))
            try:
                sd.check_input_settings(device=chosen_idx, samplerate=st.session_state["rec_sr"])
            except Exception as e:
                st.warning(tr("device_warning", default="Device warning: {error}").format(error=e))
    except Exception as e:
        st.error(tr("device_error", default="Error while retrieving device: {error}").format(error=e))

    st.subheader(tr("probe_mic", default="Probe mic"))
    colp1, colp2 = st.columns([2, 1])
    with colp1:
        probe_dur = st.slider(tr("probe_duration", default="Probe duration (s)"), 0.5, 3.0, 1.0, 0.5)
    with colp2:
        probe_disabled = st.session_state.get("rec_active", False)
        if st.button(tr("probe_button", default="Probe mic"), disabled=probe_disabled):
            path, rms, peak_dbfs = probe_mic(duration_sec=probe_dur, save_dir=recordings_dir)
            if path:
                st.success(tr("probe_recorded", default="Probe recorded: `{path}`").format(path=path))
                st.caption(tr("probe_info", default="RMS: {rms:.4f}  •  Peak: {peak_dbfs:.1f} dBFS").format(rms=rms, peak_dbfs=peak_dbfs))
                try:
                    with open(path, "rb") as f:
                        st.audio(f.read(), format="audio/wav")
                except Exception:
                    pass
            else:
                st.warning(tr("probe_failed", default="Probe failed."))

    st.subheader(tr("diagnostics", default="Diagnostics"))
    st.write(f"**{tr('working_directory', default='Working directory')}:**", os.getcwd())
    st.write(f"**{tr('recordings_dir', default='Recordings dir')}:**", str(Path(recordings_dir).resolve()))
    st.write(f"**{tr('input_device_index', default='Input device index')}:**", st.session_state.get("rec_device_index"))
    st.write(f"**{tr('callback_count', default='Callback count')}:**", REC_CTX["cb_count"])
    with REC_CTX["lock"]:
        st.write(f"**{tr('buffered_chunks', default='Buffered chunks')}:**", len(REC_CTX["buffer"]))
    df = st.session_state.get("df")
    if df is not None:
        st.write(f"**{tr('dataframe_shape', default='DataFrame shape')}:**", df.shape)
    st.write(f"**{tr('recording_active', default='Recording active')}:**", st.session_state["rec_active"])
    st.write(f"**{tr('last_recording_file', default='Last recording file')}:**", st.session_state.get("rec_last_file"))
