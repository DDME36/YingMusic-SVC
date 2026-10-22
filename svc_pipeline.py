"""Shared pipeline used by gradio_app.py and the Colab notebook.

fetch input (file / microphone / YouTube) -> separate vocals -> convert -> remix.
Models are loaded once per process and reused, and separation results are cached
per input file, so converting the same song with another voice is fast.
"""
import gc
import hashlib
import os
import re
import shutil
import subprocess
import time
import types
from typing import Callable, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
# my_inference.py / hf_utils.py resolve ./checkpoints relative to the working directory
os.chdir(ROOT)

SEP_CFG = os.path.join(ROOT, "accom_separation", "ckpt", "bs_roformer", "config_bd_roformer.yaml")
SEP_CKPT = os.path.join(ROOT, "accom_separation", "ckpt", "bs_roformer", "bs_roformer.ckpt")
SVC_CFG = os.path.join(ROOT, "configs", "YingMusic-SVC.yml")
SVC_CKPT = os.environ.get("YINGMUSIC_SVC_CKPT", os.path.join(ROOT, "checkpoints", "YingMusic-SVC-full.pt"))
WORK_ROOT = os.path.join(ROOT, "outputs", "runs")
SAMPLE_TIMBRES = {
    os.path.splitext(f)[0]: os.path.join(ROOT, "prompts", f)
    for f in sorted(os.listdir(os.path.join(ROOT, "prompts")))
    if f.lower().endswith((".wav", ".mp3", ".flac"))
}
TIMBRE_MAX_SEC = 25  # run_inference only uses the first 25 s of the reference

Log = Callable[[str], None]


class PipelineError(RuntimeError):
    """Error with a message that is safe to show to the user."""


# ---------------- setup ----------------
def ensure_checkpoints(token: Optional[str] = None, log: Log = print) -> None:
    """Download every checkpoint the pipeline needs (skips files that already exist)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    token = token or None
    os.makedirs(os.path.dirname(SEP_CKPT), exist_ok=True)
    if not os.path.isfile(SVC_CKPT):
        log("⬇️ YingMusic-SVC-full.pt")
        hf_hub_download("GiantAILab/YingMusic-SVC", "YingMusic-SVC-full.pt", local_dir="checkpoints", token=token)
    if not os.path.isfile(SEP_CKPT):
        log("⬇️ bs_roformer.ckpt")
        hf_hub_download("GiantAILab/YingMusic-SVC", "bs_roformer.ckpt", local_dir=os.path.dirname(SEP_CKPT), token=token)
    # same locations my_inference.py looks in
    log("⬇️ rmvpe / campplus / bigvgan / whisper")
    hf_hub_download("lj1995/VoiceConversionWebUI", "rmvpe.pt", cache_dir="checkpoints", token=token)
    hf_hub_download("funasr/campplus", "campplus_cn_common.bin", cache_dir="checkpoints", token=token)
    snapshot_download("nvidia/bigvgan_v2_44khz_128band_512x", cache_dir="checkpoints/hf_cache",
                      allow_patterns=["config.json", "bigvgan_generator.pt"], token=token)
    snapshot_download("openai/whisper-small", cache_dir="checkpoints/hf_cache",
                      allow_patterns=["*.json", "*.safetensors", "*.txt"], token=token)


# ---------------- helpers ----------------
def safe_name(text: str, default: str = "audio") -> str:
    """Filename-safe version of a title; keeps Thai and other non-ASCII letters."""
    text = re.sub(r"[^\w\-]+", "_", text or "", flags=re.UNICODE).strip("_")
    return text[:60] or default


def new_workdir(tag: str = "run") -> str:
    path = os.path.join(WORK_ROOT, f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}")
    os.makedirs(path, exist_ok=True)
    return path


def _run(cmd: list) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise PipelineError(f"{cmd[0]} failed: {res.stderr.strip()[-500:]}")


def to_wav(src: str, dst: str, mono: bool = False, start: float = 0.0, duration: float = 0.0) -> str:
    """Convert any audio/video file to 44.1 kHz wav, optionally trimmed to [start, start + duration]."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start > 0:
        cmd += ["-ss", str(start)]
    cmd += ["-i", src]
    if duration > 0:
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-ar", "44100"] + (["-ac", "1"] if mono else []) + [dst]
    _run(cmd)
    return dst


def to_mp3(src: str) -> str:
    """Small preview copy for the browser player (a 4 min float wav is ~80 MB)."""
    dst = os.path.splitext(src)[0] + ".mp3"
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-b:a", "192k", dst])
    return dst


def is_url(text: str) -> bool:
    return bool(re.match(r"^\s*https?://", text or ""))


def download_youtube(url: str, out_dir: str, cookies: Optional[str] = None, log: Log = print) -> tuple:
    """Download the audio of a YouTube (or any yt-dlp supported) link. Returns (path, title)."""
    try:
        import yt_dlp
    except ImportError:
        raise PipelineError("yt-dlp is not installed: pip install yt-dlp")

    log(f"⬇️ {url.strip()}")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "yt_%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if cookies:
        opts["cookiefile"] = cookies
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url.strip(), download=True)
            path = ydl.prepare_filename(info)
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        if "Sign in" in msg or "bot" in msg:
            raise PipelineError("YOUTUBE_BOT_CHECK")
        raise PipelineError(f"YouTube: {msg[-300:]}")
    return path, info.get("title") or info.get("id") or "youtube"


def fetch_audio(src: Optional[str], url: str, out_dir: str, name: str, mono: bool = False,
                start: float = 0.0, duration: float = 0.0, cookies: Optional[str] = None, log: Log = print) -> tuple:
    """Turn an uploaded/recorded file or a link into <out_dir>/<name>.wav. Returns (wav, title)."""
    if url and url.strip():
        if not is_url(url):
            raise PipelineError("BAD_URL")
        src, title = download_youtube(url, out_dir, cookies=cookies, log=log)
    elif src:
        title = os.path.splitext(os.path.basename(src))[0]
    else:
        return None, None
    wav = to_wav(src, os.path.join(out_dir, f"{name}.wav"), mono=mono, start=start, duration=duration)
    return wav, title


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _file_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


# ---------------- separation ----------------
def separate(song_wav: str, log: Log = print) -> dict:
    """Split a song into vocals / backing_vocal / instrumental. Cached by file content."""
    cache_dir = os.path.join(ROOT, "outputs", "sep_cache", _file_hash(song_wav))
    stems_dir = os.path.join(cache_dir, "out", "song")
    stems = {k: os.path.join(stems_dir, f"{k}.wav") for k in ("vocals", "backing_vocal", "instrumental")}
    if os.path.isfile(stems["vocals"]) and os.path.isfile(stems["instrumental"]):
        log("♻️ separation cache hit")
        return {k: v for k, v in stems.items() if os.path.isfile(v)}

    in_dir = os.path.join(cache_dir, "in")
    os.makedirs(in_dir, exist_ok=True)
    shutil.copy(song_wav, os.path.join(in_dir, "song.wav"))

    import torch
    from accom_separation.inference import proc_folder

    log("🎚️ separating vocals / instrumental ...")
    proc_folder({
        "model_type": "bs_roformer",
        "config_path": SEP_CFG,
        "start_check_point": SEP_CKPT,
        "input_folder": in_dir,
        "store_dir": os.path.join(cache_dir, "out"),
        "device_ids": [0],
        "disable_detailed_pbar": False,
        "force_cpu": not torch.cuda.is_available(),
    })
    _free_gpu()
    if not os.path.isfile(stems["vocals"]):
        raise PipelineError("Separation produced no vocals")
    return {k: v for k, v in stems.items() if os.path.isfile(v)}


def accompaniment(stems: dict, keep_backing: bool) -> str:
    """Instrumental, optionally with the original backing vocals mixed back in."""
    acc = stems["instrumental"]
    backing = stems.get("backing_vocal")
    if not (keep_backing and backing):
        return acc
    mixed = os.path.join(os.path.dirname(acc), "instrumental_with_backing.wav")
    if not os.path.isfile(mixed):
        _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", acc, "-i", backing,
              "-filter_complex", "amix=inputs=2:normalize=0", mixed])
    return mixed


# ---------------- conversion ----------------
_MODELS = {}


def _svc_args(**kw):
    import torch

    args = types.SimpleNamespace(
        checkpoint=SVC_CKPT, config=SVC_CFG, expname="svc", output=os.path.join(ROOT, "outputs"),
        cuda=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
        fp16=torch.cuda.is_available(), length_adjust=1.0, inference_cfg_rate=0.7,
        f0_condition=True, semi_tone_shift=None, diffusion_steps=100,
    )
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def _models(args, log: Log = print):
    if "svc" not in _MODELS:
        from my_inference import load_models_api

        log("🧠 loading SVC models (first time only) ...")
        _MODELS["svc"] = load_models_api(args, device=args.cuda)
    return _MODELS["svc"]


def convert(source_vocal: str, timbre_wav: str, out_dir: str, steps: int = 100,
            pitch_shift: Optional[int] = None, accompany: Optional[str] = None, log: Log = print) -> tuple:
    """Convert source_vocal to the timbre of timbre_wav. Returns (vocal_wav, mixed_wav or None)."""
    from my_inference import run_inference
    from Remix.auger import echo_then_reverb_save

    if not os.path.isfile(SVC_CKPT):
        raise PipelineError(f"SVC checkpoint not found: {SVC_CKPT}")
    args = _svc_args(source=source_vocal, target=timbre_wav, diffusion_steps=int(steps),
                     semi_tone_shift=pitch_shift, output=out_dir, expname=".")
    bundle = _models(args, log)
    log(f"🎤 converting ({int(steps)} steps) ...")
    vocal = run_inference(args, bundle, device=args.cuda)
    _free_gpu()

    mixed = None
    if accompany:
        log("🎛️ mixing with instrumental ...")
        mixed = os.path.join(out_dir, "cover.wav")
        echo_then_reverb_save(vocal, mixed, accompany)
    final_vocal = os.path.join(out_dir, "vocal.wav")
    shutil.move(vocal, final_vocal)
    return final_vocal, mixed


def cover(song_wav: str, timbre_wav: str, out_dir: str, has_accompaniment: bool = True,
          keep_backing: bool = False, steps: int = 100, pitch_shift: Optional[int] = None,
          log: Log = print) -> dict:
    """Full pipeline. Returns dict with vocal, cover (if accompanied) and stems."""
    stems = {}
    source, acc = song_wav, None
    if has_accompaniment:
        stems = separate(song_wav, log=log)
        source, acc = stems["vocals"], accompaniment(stems, keep_backing)
    vocal, mixed = convert(source, timbre_wav, out_dir, steps=steps, pitch_shift=pitch_shift,
                           accompany=acc, log=log)
    return {"vocal": vocal, "cover": mixed, "stems": stems}
