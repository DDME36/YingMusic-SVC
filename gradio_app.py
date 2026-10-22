import argparse
import os
from typing import Optional

import gradio as gr

import svc_pipeline as P

# ---------------- i18n ----------------
LANGS = {"ไทย": "th", "English": "en"}
T = {
    "title": {
        "th": "# 🎤 YingMusic-SVC\nเปลี่ยนเสียงร้องในเพลงให้เป็นเสียงของคนอื่น ใช้แค่ตัวอย่างเสียง 10–25 วินาที",
        "en": "# 🎤 YingMusic-SVC\nMake any song sung in another voice — all it needs is a 10–25 s voice sample",
    },
    "tab_cover": {"th": "⚡ ทำคัฟเวอร์", "en": "⚡ Cover"},
    "tab_sep": {"th": "🎚️ แยกเสียงร้อง", "en": "🎚️ Separate"},
    "tab_svc": {"th": "🎤 แปลงเสียงร้องล้วน", "en": "🎤 Convert vocal only"},
    "tab_help": {"th": "❓ วิธีใช้", "en": "❓ Help"},
    "song_hdr": {"th": "### 1. เพลงต้นฉบับ", "en": "### 1. Source song"},
    "timbre_hdr": {"th": "### 2. เสียงเป้าหมาย", "en": "### 2. Target voice"},
    "opt_hdr": {"th": "### 3. ตั้งค่า", "en": "### 3. Options"},
    "song": {"th": "อัปโหลดหรืออัดเสียงเพลง", "en": "Upload or record the song"},
    "song_url": {"th": "…หรือวางลิงก์ YouTube", "en": "…or paste a YouTube link"},
    "url_ph": {"th": "https://www.youtube.com/watch?v=...", "en": "https://www.youtube.com/watch?v=..."},
    "clip": {"th": "ตัดช่วงเพลง (วินาที)", "en": "Clip range (seconds)"},
    "clip_start": {"th": "เริ่มที่", "en": "Start"},
    "clip_len": {"th": "ความยาว (0 = ทั้งเพลง)", "en": "Length (0 = whole song)"},
    "clip_info": {"th": "ลองแปลงแค่ 30 วิแรกก่อนจะเร็วกว่ามาก", "en": "Try a 30 s clip first, it is much faster"},
    "timbre": {"th": "อัปโหลดหรืออัดเสียงเป้าหมาย (ร้อง/พูดสะอาด ไม่มีดนตรี)",
               "en": "Upload or record the target voice (clean singing/speech, no music)"},
    "timbre_url": {"th": "…หรือลิงก์ YouTube ของเสียงเป้าหมาย", "en": "…or a YouTube link of the target voice"},
    "timbre_start": {"th": "เริ่มตัดเสียงเป้าหมายที่ (วินาที)", "en": "Take the target voice from (seconds)"},
    "timbre_sep": {"th": "เสียงเป้าหมายมีดนตรี → แยกเอาแต่เสียงร้อง",
                   "en": "Target voice has music → keep vocals only"},
    "sample": {"th": "…หรือใช้เสียงตัวอย่าง", "en": "…or use a sample voice"},
    "sample_none": {"th": "(ไม่ใช้)", "en": "(none)"},
    "has_acc": {"th": "เพลงมีดนตรีประกอบ (แยกเสียงร้องก่อน)", "en": "Song has accompaniment (separate vocals first)"},
    "keep_backing": {"th": "เก็บเสียงคอรัสต้นฉบับไว้ในดนตรี", "en": "Keep the original backing vocals"},
    "steps": {"th": "คุณภาพ ↔ ความเร็ว (diffusion steps)", "en": "Quality ↔ speed (diffusion steps)"},
    "steps_info": {"th": "100 = แนะนำ · 30–50 = เร็วขึ้น", "en": "100 = recommended · 30–50 = faster"},
    "pitch_auto": {"th": "ปรับคีย์อัตโนมัติ", "en": "Auto key shift"},
    "pitch_auto_info": {"th": "ปรับคีย์ให้เข้ากับช่วงเสียงของเสียงเป้าหมาย",
                        "en": "Matches the key to the target voice range"},
    "pitch": {"th": "ปรับคีย์เอง (ครึ่งเสียง)", "en": "Manual key shift (semitones)"},
    "pitch_info": {"th": "+12 = สูงขึ้น 1 ออกเทฟ · ชาย→หญิง ลอง +12, หญิง→ชาย ลอง -12",
                   "en": "+12 = one octave up · male→female try +12, female→male try -12"},
    "advanced": {"th": "⚙️ ขั้นสูง", "en": "⚙️ Advanced"},
    "cookies": {"th": "cookies.txt ของ YouTube (ใช้เมื่อ YouTube บล็อก)",
                "en": "YouTube cookies.txt (use when YouTube blocks the download)"},
    "run_cover": {"th": "🚀 เริ่มทำคัฟเวอร์", "en": "🚀 Make cover"},
    "run_sep": {"th": "🎚️ แยกเสียง", "en": "🎚️ Separate"},
    "run_svc": {"th": "🎤 แปลงเสียง", "en": "🎤 Convert"},
    "out_cover": {"th": "🎧 เพลงคัฟเวอร์ (มิกซ์กับดนตรีแล้ว)", "en": "🎧 Cover (mixed with music)"},
    "out_vocal": {"th": "🗣️ เสียงร้องที่แปลงแล้ว", "en": "🗣️ Converted vocal"},
    "out_files": {"th": "💾 ดาวน์โหลดไฟล์ WAV คุณภาพเต็ม", "en": "💾 Download full-quality WAV"},
    "out_stems": {"th": "🎚️ เสียงที่แยกได้", "en": "🎚️ Separated stems"},
    "out_lead": {"th": "เสียงร้องหลัก", "en": "Lead vocal"},
    "out_back": {"th": "เสียงคอรัส", "en": "Backing vocal"},
    "out_inst": {"th": "ดนตรี", "en": "Instrumental"},
    "log": {"th": "สถานะ", "en": "Status"},
    "svc_src": {"th": "เสียงร้องล้วน (ไม่มีดนตรี)", "en": "Dry vocal (no music)"},
    "svc_acc": {"th": "ดนตรีสำหรับมิกซ์ (ไม่ใส่ก็ได้)", "en": "Instrumental to mix with (optional)"},
    "help": {
        "th": """### วิธีใช้
1. **เพลงต้นฉบับ**: อัปโหลดไฟล์ (mp3/wav/flac/m4a/mp4) กดไมค์เพื่ออัดเสียง หรือวางลิงก์ YouTube
2. **เสียงเป้าหมาย**: เสียงร้องหรือเสียงพูดที่ชัด ไม่มีดนตรี ยาว 10–25 วินาที (ใช้แค่ 25 วิแรก)
   ถ้าเอามาจากเพลงใน YouTube ให้ติ๊ก *แยกเอาแต่เสียงร้อง* และตั้งเวลาเริ่มให้ตรงท่อนที่ร้องชัด ๆ
3. กด **เริ่มทำคัฟเวอร์** รอบแรกต้องโหลดโมเดลราว 1 นาที รอบต่อไปจะเร็วขึ้น
   ถ้าใช้เพลงเดิมแต่เปลี่ยนเสียงเป้าหมาย จะข้ามขั้นแยกเสียงไปเลย

### เคล็ดลับ
- ลองตัดแค่ 30 วิก่อน ฟังว่าโอเคแล้วค่อยแปลงทั้งเพลง
- เสียงเพี้ยนหรือแหบ ลองปิด *ปรับคีย์อัตโนมัติ* แล้วตั้งคีย์เอง
- เปลี่ยนเสียงชายเป็นหญิง ลอง +12 · หญิงเป็นชาย ลอง -12

### YouTube โหลดไม่ได้?
บน Colab บางครั้ง YouTube จะขึ้นว่า *"Sign in to confirm you're not a bot"*
ให้ export `cookies.txt` จากเบราว์เซอร์ที่ล็อกอิน YouTube อยู่ (เช่น extension *Get cookies.txt LOCALLY*)
แล้วอัปโหลดในส่วน **⚙️ ขั้นสูง** หรือจะโหลดเพลงมาเองแล้วอัปโหลดไฟล์แทนก็ได้

⚠️ ใช้กับเสียงที่ได้รับอนุญาตเท่านั้น อย่าเลียนแบบเสียงคนอื่นเพื่อหลอกลวง""",
        "en": """### How to use
1. **Source song**: upload a file (mp3/wav/flac/m4a/mp4), record with the mic, or paste a YouTube link
2. **Target voice**: clean singing or speech without music, 10–25 s (only the first 25 s is used).
   If it comes from a song on YouTube, tick *keep vocals only* and set the start time to a clear vocal part.
3. Press **Make cover**. The first run loads the models (~1 min); later runs are faster.
   Re-using the same song with another voice skips separation.

### Tips
- Try a 30 s clip first, then convert the whole song
- If the result sounds off-key, turn off *auto key shift* and set it manually
- Male → female: try +12 · female → male: try -12

### YouTube download fails?
On Colab YouTube sometimes answers *"Sign in to confirm you're not a bot"*.
Export `cookies.txt` from a browser logged into YouTube (e.g. the *Get cookies.txt LOCALLY* extension)
and upload it under **⚙️ Advanced**, or download the song yourself and upload the file.

⚠️ Only use voices you have permission to use. Do not impersonate people to deceive.""",
    },
}
ERRORS = {
    "need_song": {"th": "กรุณาอัปโหลด อัดเสียง หรือวางลิงก์เพลงก่อน", "en": "Please upload, record or link a song first"},
    "need_timbre": {"th": "กรุณาเลือกเสียงเป้าหมาย", "en": "Please provide a target voice"},
    "need_vocal": {"th": "กรุณาใส่เสียงร้อง", "en": "Please provide a vocal"},
    "BAD_URL": {"th": "ลิงก์ไม่ถูกต้อง ต้องขึ้นต้นด้วย https://", "en": "Invalid link, it must start with https://"},
    "YOUTUBE_BOT_CHECK": {
        "th": "YouTube บล็อกการดาวน์โหลด (bot check) ให้อัปโหลด cookies.txt ใน ⚙️ ขั้นสูง หรืออัปโหลดไฟล์เพลงเอง",
        "en": "YouTube blocked the download (bot check). Upload cookies.txt under ⚙️ Advanced, or upload the file yourself",
    },
}


def tr(key: str, lang: str) -> str:
    table = T.get(key) or ERRORS.get(key)
    return table[lang] if table else key


def fail(e: Exception, lang: str):
    msg = str(e)
    raise gr.Error(ERRORS[msg][lang] if msg in ERRORS else msg)


# ---------------- handlers ----------------
class Logger:
    def __init__(self, progress: gr.Progress):
        self.lines, self.progress = [], progress

    def __call__(self, msg: str):
        print(msg)
        self.lines.append(msg)
        self.progress(None, desc=msg)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _cookies(f) -> Optional[str]:
    return f if isinstance(f, str) else getattr(f, "name", None)


def _timbre(work, file, url, sample, start, sep, cookies, log):
    """Target voice -> mono wav, at most TIMBRE_MAX_SEC long (trimmed before separation to keep it fast)."""
    if file or (url and url.strip()):
        wav, _ = P.fetch_audio(file, url, work, "timbre_raw", start=start, duration=P.TIMBRE_MAX_SEC,
                               cookies=cookies, log=log)
        if sep:
            wav = P.separate(wav, log=log)["vocals"]
        return P.to_wav(wav, os.path.join(work, "timbre.wav"), mono=True)
    if sample in P.SAMPLE_TIMBRES:
        return P.SAMPLE_TIMBRES[sample]
    raise P.PipelineError("need_timbre")


def do_cover(lang_name, song, song_url, clip_start, clip_len, timbre, timbre_url, sample, timbre_start, timbre_sep,
             has_acc, keep_backing, steps, pitch_auto, pitch, cookies, progress=gr.Progress(track_tqdm=True)):
    lang, log = LANGS[lang_name], Logger(progress)
    try:
        work = P.new_workdir("cover")
        song_wav, title = P.fetch_audio(song, song_url, work, "song", start=clip_start or 0,
                                        duration=clip_len or 0, cookies=_cookies(cookies), log=log)
        if not song_wav:
            raise P.PipelineError("need_song")
        timbre_wav = _timbre(work, timbre, timbre_url, sample, timbre_start or 0, timbre_sep,
                             _cookies(cookies), log)
        res = P.cover(song_wav, timbre_wav, work, has_accompaniment=has_acc, keep_backing=keep_backing,
                      steps=steps, pitch_shift=None if pitch_auto else int(pitch), log=log)
    except P.PipelineError as e:
        fail(e, lang)

    name = P.safe_name(title, "song")
    files = []
    for key in ("cover", "vocal"):
        if res[key]:
            nice = os.path.join(work, f"{name}_{key}.wav")
            os.replace(res[key], nice)
            res[key] = nice
            files.append(nice)
    log("✅ done")
    stems = res["stems"]
    return (P.to_mp3(res["cover"]) if res["cover"] else None, P.to_mp3(res["vocal"]), files,
            stems.get("vocals"), stems.get("backing_vocal"), stems.get("instrumental"), log.text)


def do_separate(lang_name, song, song_url, clip_start, clip_len, cookies, progress=gr.Progress(track_tqdm=True)):
    lang, log = LANGS[lang_name], Logger(progress)
    try:
        work = P.new_workdir("sep")
        song_wav, _ = P.fetch_audio(song, song_url, work, "song", start=clip_start or 0,
                                    duration=clip_len or 0, cookies=_cookies(cookies), log=log)
        if not song_wav:
            raise P.PipelineError("need_song")
        stems = P.separate(song_wav, log=log)
    except P.PipelineError as e:
        fail(e, lang)
    log("✅ done")
    return stems.get("vocals"), stems.get("backing_vocal"), stems.get("instrumental"), log.text


def do_convert(lang_name, vocal, timbre, timbre_url, sample, timbre_start, timbre_sep, acc, steps, pitch_auto, pitch,
               cookies, progress=gr.Progress(track_tqdm=True)):
    lang, log = LANGS[lang_name], Logger(progress)
    try:
        if not vocal:
            raise P.PipelineError("need_vocal")
        work = P.new_workdir("svc")
        src = P.to_wav(vocal, os.path.join(work, "source.wav"))
        timbre_wav = _timbre(work, timbre, timbre_url, sample, timbre_start or 0, timbre_sep,
                             _cookies(cookies), log)
        vc, mixed = P.convert(src, timbre_wav, work, steps=steps,
                              pitch_shift=None if pitch_auto else int(pitch), accompany=acc, log=log)
    except P.PipelineError as e:
        fail(e, lang)
    log("✅ done")
    return P.to_mp3(vc), P.to_mp3(mixed) if mixed else None, [f for f in (vc, mixed) if f], log.text


# ---------------- UI ----------------
CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
#title h1 {margin-bottom: 0;}
.run-btn {min-height: 56px; font-size: 1.1em !important;}
footer {display: none !important;}
"""


def build_ui(lang_name: str = "ไทย"):
    lang = LANGS.get(lang_name, "th")
    reg = []  # (component, key, attrs) relabelled when the language changes

    def L(comp, key, *, attr="label", info=None, placeholder=None):
        reg.append((comp, key, attr, info, placeholder))
        return comp

    def kw(key, info=None, placeholder=None):
        d = {"label": tr(key, lang)}
        if info:
            d["info"] = tr(info, lang)
        if placeholder:
            d["placeholder"] = tr(placeholder, lang)
        return d

    def audio_in(key):
        return L(gr.Audio(sources=["upload", "microphone"], type="filepath", **kw(key)), key)

    def url_in(key):
        return L(gr.Textbox(lines=1, max_lines=1, **kw(key, placeholder="url_ph")), key, placeholder="url_ph")

    def timbre_inputs():
        t = audio_in("timbre")
        url = url_in("timbre_url")
        with gr.Row():
            start = L(gr.Number(value=0, minimum=0, **kw("timbre_start")), "timbre_start")
            sep = L(gr.Checkbox(value=False, **kw("timbre_sep")), "timbre_sep")
        samples = [tr("sample_none", lang)] + list(P.SAMPLE_TIMBRES)
        sample = L(gr.Dropdown(choices=samples, value=samples[0], **kw("sample")), "sample")
        return t, url, sample, start, sep

    def pitch_inputs():
        with gr.Row():
            auto = L(gr.Checkbox(value=True, **kw("pitch_auto", "pitch_auto_info")), "pitch_auto", info="pitch_auto_info")
            pitch = L(gr.Slider(-24, 24, value=0, step=1, interactive=False, **kw("pitch", "pitch_info")),
                      "pitch", info="pitch_info")
        auto.change(lambda a: gr.update(interactive=not a), auto, pitch)
        return auto, pitch

    def steps_input():
        return L(gr.Slider(10, 200, value=100, step=10, **kw("steps", "steps_info")), "steps", info="steps_info")

    def clip_inputs():
        with gr.Row():
            start = L(gr.Number(value=0, minimum=0, **kw("clip_start")), "clip_start")
            length = L(gr.Number(value=0, minimum=0, **kw("clip_len", "clip_info")), "clip_len", info="clip_info")
        return start, length

    with gr.Blocks(title="YingMusic-SVC", theme=gr.themes.Soft(primary_hue="violet", secondary_hue="pink"),
                   css=CSS) as demo:
        with gr.Row(equal_height=True):
            title = L(gr.Markdown(tr("title", lang), elem_id="title"), "title", attr="value")
            lang_sel = gr.Radio(list(LANGS), value=lang_name, show_label=False, scale=0, min_width=180)

        with gr.Tabs():
            # ---------- one click ----------
            with L(gr.Tab(tr("tab_cover", lang)), "tab_cover"):
                with gr.Row():
                    with gr.Column():
                        L(gr.Markdown(tr("song_hdr", lang)), "song_hdr", attr="value")
                        song = audio_in("song")
                        song_url = url_in("song_url")
                        clip_start, clip_len = clip_inputs()
                        L(gr.Markdown(tr("timbre_hdr", lang)), "timbre_hdr", attr="value")
                        t_file, t_url, t_sample, t_start, t_sep = timbre_inputs()
                    with gr.Column():
                        L(gr.Markdown(tr("opt_hdr", lang)), "opt_hdr", attr="value")
                        has_acc = L(gr.Checkbox(value=True, **kw("has_acc")), "has_acc")
                        keep_backing = L(gr.Checkbox(value=False, **kw("keep_backing")), "keep_backing")
                        steps = steps_input()
                        p_auto, p_val = pitch_inputs()
                        with L(gr.Accordion(tr("advanced", lang), open=False), "advanced"):
                            cookies = L(gr.File(file_types=[".txt"], **kw("cookies")), "cookies")
                        run_cover = L(gr.Button(tr("run_cover", lang), variant="primary", elem_classes="run-btn"),
                                      "run_cover", attr="value")
                        o_cover = L(gr.Audio(type="filepath", interactive=False, **kw("out_cover")), "out_cover")
                        o_vocal = L(gr.Audio(type="filepath", interactive=False, **kw("out_vocal")), "out_vocal")
                        o_files = L(gr.File(file_count="multiple", interactive=False, **kw("out_files")), "out_files")
                        with L(gr.Accordion(tr("out_stems", lang), open=False), "out_stems"):
                            o_lead = L(gr.Audio(type="filepath", interactive=False, **kw("out_lead")), "out_lead")
                            o_back = L(gr.Audio(type="filepath", interactive=False, **kw("out_back")), "out_back")
                            o_inst = L(gr.Audio(type="filepath", interactive=False, **kw("out_inst")), "out_inst")
                        o_log = L(gr.Textbox(lines=4, interactive=False, **kw("log")), "log")
                has_acc.change(lambda a: gr.update(interactive=a), has_acc, keep_backing)
                run_cover.click(
                    do_cover,
                    [lang_sel, song, song_url, clip_start, clip_len, t_file, t_url, t_sample, t_start, t_sep,
                     has_acc, keep_backing, steps, p_auto, p_val, cookies],
                    [o_cover, o_vocal, o_files, o_lead, o_back, o_inst, o_log],
                )

            # ---------- separation only ----------
            with L(gr.Tab(tr("tab_sep", lang)), "tab_sep"):
                with gr.Row():
                    with gr.Column():
                        s_song = audio_in("song")
                        s_url = url_in("song_url")
                        s_start, s_len = clip_inputs()
                        run_sep = L(gr.Button(tr("run_sep", lang), variant="primary", elem_classes="run-btn"),
                                    "run_sep", attr="value")
                    with gr.Column():
                        s_lead = L(gr.Audio(type="filepath", interactive=False, **kw("out_lead")), "out_lead")
                        s_back = L(gr.Audio(type="filepath", interactive=False, **kw("out_back")), "out_back")
                        s_inst = L(gr.Audio(type="filepath", interactive=False, **kw("out_inst")), "out_inst")
                        s_log = L(gr.Textbox(lines=3, interactive=False, **kw("log")), "log")
                run_sep.click(do_separate, [lang_sel, s_song, s_url, s_start, s_len, cookies],
                              [s_lead, s_back, s_inst, s_log])

            # ---------- conversion only ----------
            with L(gr.Tab(tr("tab_svc", lang)), "tab_svc"):
                with gr.Row():
                    with gr.Column():
                        v_src = audio_in("svc_src")
                        v_file, v_url, v_sample, v_start, v_sep = timbre_inputs()
                        v_acc = L(gr.Audio(sources=["upload"], type="filepath", **kw("svc_acc")), "svc_acc")
                    with gr.Column():
                        v_steps = steps_input()
                        v_auto, v_pitch = pitch_inputs()
                        run_svc = L(gr.Button(tr("run_svc", lang), variant="primary", elem_classes="run-btn"),
                                    "run_svc", attr="value")
                        v_out = L(gr.Audio(type="filepath", interactive=False, **kw("out_vocal")), "out_vocal")
                        v_mix = L(gr.Audio(type="filepath", interactive=False, **kw("out_cover")), "out_cover")
                        v_files = L(gr.File(file_count="multiple", interactive=False, **kw("out_files")), "out_files")
                        v_log = L(gr.Textbox(lines=3, interactive=False, **kw("log")), "log")
                run_svc.click(do_convert,
                              [lang_sel, v_src, v_file, v_url, v_sample, v_start, v_sep, v_acc, v_steps, v_auto,
                               v_pitch, cookies],
                              [v_out, v_mix, v_files, v_log])

            with L(gr.Tab(tr("tab_help", lang)), "tab_help"):
                L(gr.Markdown(tr("help", lang)), "help", attr="value")

        def relabel(name):
            lg = LANGS[name]
            updates = []
            for comp, key, attr, info, placeholder in reg:
                if isinstance(comp, gr.Dropdown):  # the "(none)" choice is translated too
                    choices = [tr("sample_none", lg)] + list(P.SAMPLE_TIMBRES)
                    updates.append(gr.update(label=tr(key, lg), choices=choices, value=choices[0]))
                    continue
                u = {attr: tr(key, lg)}
                if info:
                    u["info"] = tr(info, lg)
                if placeholder:
                    u["placeholder"] = tr(placeholder, lg)
                updates.append(gr.update(**u))
            return updates

        lang_sel.change(relabel, lang_sel, [c for c, *_ in reg])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=list(LANGS) + list(LANGS.values()), default="ไทย")
    parser.add_argument("--share", action="store_true", help="create a public gradio.live link")
    parser.add_argument("--port", type=int, default=30911)
    parser.add_argument("--download", action="store_true", help="download checkpoints before starting")
    a = parser.parse_args()
    if a.download:
        P.ensure_checkpoints()
    lang_name = {v: k for k, v in LANGS.items()}.get(a.lang, a.lang)
    build_ui(lang_name).queue().launch(server_name="0.0.0.0", server_port=a.port, share=a.share, inbrowser=not a.share)
