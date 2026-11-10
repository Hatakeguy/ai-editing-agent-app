#!/usr/bin/env python3
"""
reel_pipeline.py - local, CPU-only jump-cut + caption + reframe pipeline.

Steps
  1. Probe the source with ffprobe (nothing is assumed about the file).
  2. Detect silences with FFmpeg `silencedetect`.
  3. Build an Edit Decision List (EDL): keep-segments with millisecond timestamps.
  4. (Optional) get word timestamps from local Whisper (faster-whisper, CPU/int8),
     a words.json cache, or an existing SRT; remap them onto the edited timeline.
  5. Build an ASS caption file (word-by-word highlight, or plain cues for SRT input).
  6. Render in ONE streaming FFmpeg pass (libx264, CPU): cuts, alternating punch-in,
     9:16 reframe, captions, loudness normalisation.
  7. Validate the output with ffprobe and report honestly.

The source file is never modified. The output refuses to overwrite anything
unless --force is given.

Examples
  python reel_pipeline.py talk.mp4 -o talk_reel.mp4 --dry-run          # EDL only
  python reel_pipeline.py talk.mp4 -o talk_reel.mp4 --transcribe       # Whisper captions
  python reel_pipeline.py talk.mp4 -o talk_reel.mp4 --srt talk.srt     # existing SRT
  python reel_pipeline.py talk.mp4 -o talk_reel.mp4 --edl talk_reel_work/edl.json --words talk_reel_work/words.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class PipelineError(Exception):
    """Any user-facing failure. The CLI prints it; the GUI shows it in a dialog."""


def _find_tool(name):
    """Prefer an ffmpeg/ffprobe shipped next to the app (PyInstaller bundle or
    script folder); otherwise fall back to PATH."""
    exe = name + (".exe" if os.name == "nt" else "")
    bases = []
    if getattr(sys, "frozen", False):
        bases += [Path(getattr(sys, "_MEIPASS", sys.executable)), Path(sys.executable).parent]
    bases.append(Path(__file__).resolve().parent)
    for b in bases:
        if (b / exe).is_file():
            return str(b / exe)
    return name


FFMPEG = _find_tool("ffmpeg")
FFPROBE = _find_tool("ffprobe")

# --------------------------------------------------------------------------- #
# Templates. Add more by copying a block.
# --------------------------------------------------------------------------- #
PRESETS = {
    "Reel_FastPaced": dict(
        width=1080, height=1920, fit="cover",   # crop-to-fill 9:16
        zoom=1.08,                              # alternate 1.00 / 1.08 punch-in per cut
        pad=0.10,                               # seconds of silence kept each side of a cut
        cap_size=84, cap_outline=7, cap_shadow=2,
        cap_align=5, cap_margin_v=0,            # 5 = middle-centre
        cap_words=3, cap_upper=True, cap_highlight="&H0000FFFF",  # yellow (BGR)
    ),
    "YouTube_Long_Form": dict(
        width=1920, height=1080, fit="contain",  # letterbox, never crop
        zoom=1.0,
        pad=0.25,
        cap_size=54, cap_outline=4, cap_shadow=1,
        cap_align=2, cap_margin_v=70,            # 2 = bottom-centre
        cap_words=6, cap_upper=False, cap_highlight="&H0000FFFF",
    ),
}


@dataclass
class Segment:
    start: float          # source seconds
    end: float
    out_start: float = 0.0

    @property
    def dur(self):
        return self.end - self.start


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def fmt(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ass_time(t):
    cs = int(round(max(t, 0) * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def run(cmd, cwd=None):
    # stdin=DEVNULL + CREATE_NO_WINDOW: a windowed (console=False) Windows app has
    # no valid stdin handle, and subprocess.run then fails with WinError 6.
    kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
    try:
        return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=cwd, stdin=subprocess.DEVNULL, **kw)
    except FileNotFoundError:
        raise PipelineError(
            f"'{cmd[0]}' was not found. Install FFmpeg and add it to PATH, or put "
            f"ffmpeg and ffprobe next to the app.")


def die(msg):
    raise PipelineError(msg)


def probe(path):
    p = run([FFPROBE, "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,width,height", "-of", "json", str(path)])
    if p.returncode != 0:
        die(f"ffprobe could not read {path}:\n{p.stderr.strip()}")
    j = json.loads(p.stdout)
    streams = j.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    dur = float(j.get("format", {}).get("duration", 0) or 0)
    return dict(duration=dur, has_audio=has_audio,
                width=(v or {}).get("width"), height=(v or {}).get("height"),
                has_video=v is not None)


# --------------------------------------------------------------------------- #
# Analysis + EDL
# --------------------------------------------------------------------------- #
def detect_silences(src, noise_db, min_dur):
    p = run([FFMPEG, "-hide_banner", "-nostats", "-i", str(src), "-vn",
             "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}", "-f", "null", "-"])
    if p.returncode != 0:
        die(f"silence detection failed:\n{p.stderr[-1500:]}")
    silences, start = [], None
    for line in p.stderr.splitlines():
        m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
        if m:
            start = max(0.0, float(m.group(1)))
        m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
        if m and start is not None:
            silences.append((start, float(m.group(1))))
            start = None
    return silences, start  # `start` not None => file ends inside a silence


def build_segments(duration, silences, trailing_start, min_silence, pad):
    cuts = []
    allsil = list(silences) + ([(trailing_start, duration)] if trailing_start is not None else [])
    for s, e in allsil:
        e = min(e, duration)
        if e - s < min_silence:
            continue
        cs = 0.0 if s <= 0.01 else s + pad           # leading silence: cut from 0
        ce = duration if e >= duration - 0.01 else e - pad  # trailing: cut to the end
        if ce - cs > 0.05:
            cuts.append((round(cs, 3), round(ce, 3)))
    segs, cursor = [], 0.0
    for cs, ce in cuts:
        if cs - cursor >= 0.04:
            segs.append(Segment(round(cursor, 3), cs))
        cursor = ce
    if duration - cursor >= 0.04:
        segs.append(Segment(round(cursor, 3), round(duration, 3)))
    return segs, cuts


def finalize(segs):
    t = 0.0
    for sg in segs:
        sg.out_start = round(t, 3)
        t += sg.dur
    return t


def to_out(t, segs, prefer):
    """Map a source time to the edited timeline. If t fell in a cut, snap
    forward ('next') or backward ('prev') to the nearest kept edge."""
    for sg in segs:
        if sg.start <= t <= sg.end:
            return sg.out_start + (t - sg.start)
    if prefer == "next":
        for sg in segs:
            if sg.start > t:
                return sg.out_start
        return segs[-1].out_start + segs[-1].dur
    prev = 0.0
    for sg in segs:
        if sg.end < t:
            prev = sg.out_start + sg.dur
        else:
            break
    return prev


def print_edl(segs, cuts, src_dur, out_dur):
    print("\nEDIT DECISION LIST (source time -> output time)")
    print(f"{'#':>3}  {'src in':<13} {'src out':<13} {'dur':>8}  {'out in':<13} zoom")
    for i, sg in enumerate(segs):
        print(f"{i:>3}  {fmt(sg.start):<13} {fmt(sg.end):<13} {sg.dur:>7.3f}s  {fmt(sg.out_start):<13} {'punch' if i % 2 else '-'}")
    print("\nREMOVED (silence)")
    for cs, ce in cuts:
        print(f"     {fmt(cs)} -> {fmt(ce)}  ({ce - cs:.3f}s)")
    print(f"\nsource {src_dur:.3f}s -> edit {out_dur:.3f}s  (removed {src_dur - out_dur:.3f}s)\n")


# --------------------------------------------------------------------------- #
# Transcript inputs
# --------------------------------------------------------------------------- #
def transcribe(src, workdir, model_size, language):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        die("faster-whisper is not installed. Run: pip install faster-whisper")
    wav = workdir / "audio16k.wav"
    p = run([FFMPEG, "-hide_banner", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    if p.returncode != 0:
        die(f"audio extraction failed:\n{p.stderr[-1500:]}")
    # CPU + int8: no VRAM use at all, ~1 GB RAM for the 'small' model.
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(wav), word_timestamps=True, vad_filter=True, language=language)
    words, srt = [], []
    for n, seg in enumerate(segments, 1):
        srt.append(f"{n}\n{srt_time(seg.start)} --> {srt_time(seg.end)}\n{seg.text.strip()}\n")
        for w in seg.words or []:
            words.append([round(w.start, 3), round(w.end, 3), w.word.strip()])
    wav.unlink(missing_ok=True)
    (workdir / "transcript_source.srt").write_text("\n".join(srt), encoding="utf-8")
    (workdir / "words.json").write_text(json.dumps(words, ensure_ascii=False, indent=0), encoding="utf-8")
    return words


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(path):
    txt = Path(path).read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", txt.strip()):
        lines = block.strip().split("\n")
        for i, ln in enumerate(lines):
            m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", ln.strip())
            if m:
                g = list(map(int, m.groups()))
                s = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                e = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                text = re.sub(r"<[^>]+>", "", " ".join(lines[i + 1:])).strip()
                if text:
                    cues.append((s, e, text))
                break
    return cues


# --------------------------------------------------------------------------- #
# ASS captions
# --------------------------------------------------------------------------- #
def ass_escape(s):
    return s.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def ass_header(p):
    W, H = p["width"], p["height"]
    return (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,"
        "Alignment,MarginL,MarginR,MarginV,Encoding\n"
        f"Style: Cap,{p['font']},{p['cap_size']},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,{p['cap_outline']},{p['cap_shadow']},{p['cap_align']},"
        f"{int(W * 0.08)},{int(W * 0.08)},{p['cap_margin_v']},1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )


def build_ass_words(words, segs, p):
    """Word-by-word highlight. Words whose midpoint lies in a cut are dropped
    (their audio is gone) and counted so the caller can warn."""
    kept, dropped = [], 0
    for s, e, text in words:
        mid = (s + e) / 2
        if not any(sg.start <= mid <= sg.end for sg in segs):
            dropped += 1
            continue
        os_, oe = to_out(s, segs, "next"), to_out(e, segs, "prev")
        if oe <= os_:
            oe = os_ + 0.05
        kept.append((os_, oe, ass_escape(text.upper() if p["cap_upper"] else text)))
    # chunk into on-screen groups
    chunks, cur = [], []
    for w in kept:
        if cur and (len(cur) >= p["cap_words"] or w[0] - cur[-1][1] > 0.6):
            chunks.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?]$", w[2]):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    hi = p["cap_highlight"]
    lines = []
    for ch in chunks:
        for i, (ws, we, _) in enumerate(ch):
            end = ch[i + 1][0] if i + 1 < len(ch) else we
            if end <= ws:
                continue
            parts = []
            for j, (_, _, t) in enumerate(ch):
                if j == i:
                    parts.append(f"{{\\c{hi}&\\fscx112\\fscy112\\t(0,90,\\fscx100\\fscy100)}}{t}{{\\r}}")
                else:
                    parts.append(t)
            lines.append(f"Dialogue: 0,{ass_time(ws)},{ass_time(end)},Cap,,0,0,0,,{' '.join(parts)}")
    return ass_header(p) + "\n".join(lines) + "\n", dropped


def build_ass_srt(cues, segs, p):
    lines = []
    for s, e, text in cues:
        os_, oe = to_out(s, segs, "next"), to_out(e, segs, "prev")
        if oe - os_ < 0.05:
            continue
        t = ass_escape(text.upper() if p["cap_upper"] else text)
        lines.append(f"Dialogue: 0,{ass_time(os_)},{ass_time(oe)},Cap,,0,0,0,,{t}")
    return ass_header(p) + "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def even(x):
    return int(round(x / 2.0)) * 2


def vfit(p, z):
    W, H = p["width"], p["height"]
    if p["fit"] == "cover":
        return f"scale={even(W * z)}:{even(H * z)}:force_original_aspect_ratio=increase,crop={W}:{H}"
    return f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2"


def build_filter(segs, p, ass_name):
    parts, pads = [], []
    for i, sg in enumerate(segs):
        z = p["zoom"] if (p["zoom"] > 1 and i % 2) else 1.0
        f = round(min(0.012, sg.dur / 4), 3)  # ~12 ms edge fades: no clicks at cuts
        parts.append(f"[0:v:0]trim=start={sg.start:.3f}:end={sg.end:.3f},setpts=PTS-STARTPTS,"
                     f"{vfit(p, z)},setsar=1[v{i}]")
        parts.append(f"[0:a:0]atrim=start={sg.start:.3f}:end={sg.end:.3f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:st=0:d={f},afade=t=out:st={sg.dur - f:.3f}:d={f}[a{i}]")
        pads.append(f"[v{i}][a{i}]")
    parts.append(f"{''.join(pads)}concat=n={len(segs)}:v=1:a=1[cv][ca]")
    parts.append(f"[cv]{'ass=' + ass_name + ',' if ass_name else ''}format=yuv420p[vout]")
    parts.append("[ca]loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[aout]")
    return ";\n".join(parts)


def ffmpeg_major():
    p = run([FFMPEG, "-version"])
    m = re.search(r"ffmpeg version n?(\d+)\.", p.stdout)
    return int(m.group(1)) if m else None


def render(src, out, workdir, filter_text, threads):
    fscript = workdir / "filtergraph.txt"
    fscript.write_text(filter_text, encoding="utf-8")  # file avoids Windows' command-length limit
    major = ffmpeg_major()
    flags = ["-/filter_complex", "-filter_complex_script"]
    if major is not None and major < 7:
        flags.reverse()
    last = None
    for flag in flags:
        cmd = [FFMPEG, "-hide_banner", "-y", "-i", str(src), flag, str(fscript),
               "-map", "[vout]", "-map", "[aout]",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
               "-threads", str(threads), str(out)]
        # cwd = workdir so the ass filter can use a bare relative filename
        # (avoids Windows drive-letter escaping problems inside filtergraphs).
        last = run(cmd, cwd=str(workdir))
        if last.returncode == 0:
            return
        if "nrecognized option" in last.stderr or "Option not found" in last.stderr:
            continue
        break
    die(f"FFmpeg failed (exit {last.returncode}). Filtergraph saved at {fscript}\n"
        f"--- last FFmpeg output ---\n{last.stderr[-1500:]}")


def validate(out, expected, p):
    if not out.exists() or out.stat().st_size == 0:
        die("output file missing or empty")
    info = probe(out)
    problems = []
    if (info["width"], info["height"]) != (p["width"], p["height"]):
        problems.append(f"resolution {info['width']}x{info['height']} != {p['width']}x{p['height']}")
    if not info["has_audio"]:
        problems.append("no audio stream")
    if abs(info["duration"] - expected) > 0.35:
        problems.append(f"duration {info['duration']:.3f}s vs expected {expected:.3f}s")
    print("VALIDATION (ffprobe)")
    print(f"  file      {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  duration  {info['duration']:.3f}s  (expected {expected:.3f}s)")
    print(f"  size      {info['width']}x{info['height']}   audio: {'yes' if info['has_audio'] else 'NO'}")
    print("  status    " + ("OK" if not problems else "PROBLEMS: " + "; ".join(problems)))
    print("  Not checked automatically: lip-sync and caption timing - scrub the file to confirm.")
    return not problems


# --------------------------------------------------------------------------- #
def main(argv=None):
    """Returns 0 = ok, 2 = rendered but validation found problems.
    Raises PipelineError on failure. `argv` lets the GUI call this directly."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--template", default="Reel_FastPaced", choices=PRESETS)
    ap.add_argument("--min-silence", type=float, default=1.0, help="cut silences >= this many seconds (default 1.0)")
    ap.add_argument("--pad", type=float, help="seconds of silence kept at each side of a cut (template default)")
    ap.add_argument("--noise", type=float, default=-35.0, help="silence threshold in dB (default -35)")
    ap.add_argument("--transcribe", action="store_true", help="run local faster-whisper for word-level captions")
    ap.add_argument("--model", default="small", help="whisper model: tiny/base/small/medium (default small)")
    ap.add_argument("--language", default=None, help="e.g. en, hi (default: auto-detect)")
    ap.add_argument("--words", help="words.json from an earlier run: [[start,end,\"word\"], ...] in SOURCE time")
    ap.add_argument("--srt", help="existing SRT in SOURCE time (plain cues, no word highlight)")
    ap.add_argument("--edl", help="edl.json to use instead of silence detection (edit it by hand first)")
    ap.add_argument("--font", default="Arial")
    ap.add_argument("--threads", type=int, default=0, help="x264 threads, 0 = auto (lower it to ease RAM/CPU)")
    ap.add_argument("--dry-run", action="store_true", help="write the EDL and stop")
    ap.add_argument("--force", action="store_true", help="allow overwriting the output file")
    a = ap.parse_args(argv)

    src, out = Path(a.input).resolve(), Path(a.output).resolve()
    if not src.is_file():
        die(f"input not found: {src}")
    if out == src:
        die("output path equals the input path; refusing to touch the source")
    if out.exists() and not a.force and not a.dry_run:
        die(f"{out} exists; use --force to overwrite")

    p = dict(PRESETS[a.template], font=a.font)
    if a.pad is not None:
        p["pad"] = a.pad
    workdir = out.parent / (out.stem + "_work")
    workdir.mkdir(parents=True, exist_ok=True)

    # 1. probe
    info = probe(src)
    if not info["has_video"] or not info["has_audio"]:
        die("input needs both a video and an audio stream")
    print(f"Source: {src.name}  {info['width']}x{info['height']}  {info['duration']:.3f}s")

    # 2-3. analysis + EDL
    if a.edl:
        raw = json.loads(Path(a.edl).read_text())
        segs = [Segment(float(s["start"]), float(s["end"])) for s in raw["segments"]]
        cuts = []
        prev = 0.0
        for sg in segs:
            if sg.start - prev > 0.001:
                cuts.append((prev, sg.start))
            prev = sg.end
        if info["duration"] - prev > 0.001:
            cuts.append((prev, info["duration"]))
    else:
        silences, trailing = detect_silences(src, a.noise, a.min_silence)
        segs, cuts = build_segments(info["duration"], silences, trailing, a.min_silence, p["pad"])
    if not segs:
        die("nothing left after cutting; try a lower --noise (e.g. -45) or a larger --min-silence")
    out_dur = finalize(segs)
    print_edl(segs, cuts, info["duration"], out_dur)
    edl_path = workdir / "edl.json"
    edl_path.write_text(json.dumps({"source": src.name, "segments": [
        {"start": s.start, "end": s.end, "out_start": s.out_start} for s in segs]}, indent=2))
    print(f"EDL saved: {edl_path}  (edit it and re-run with --edl to override)")
    if a.dry_run:
        return 0

    # 4-5. captions
    ass_name = None
    words = None
    if a.words:
        words = json.loads(Path(a.words).read_text(encoding="utf-8"))
    elif a.transcribe:
        print("Transcribing locally (CPU)...")
        words = transcribe(src, workdir, a.model, a.language)
    if words is not None:
        ass, dropped = build_ass_words(words, segs, p)
        if dropped:
            print(f"WARNING: {dropped} transcribed word(s) fall inside removed regions and have no captions. "
                  f"Check the cuts around them (lower --noise or raise --min-silence if speech was clipped).")
        (workdir / "captions.ass").write_text(ass, encoding="utf-8")
        ass_name = "captions.ass"
    elif a.srt:
        (workdir / "captions.ass").write_text(build_ass_srt(parse_srt(a.srt), segs, p), encoding="utf-8")
        ass_name = "captions.ass"
    else:
        print("No --transcribe / --words / --srt given: rendering without captions.")

    # 6. render
    print("Rendering (libx264, CPU)...")
    render(src, out, workdir, build_filter(segs, p, ass_name), a.threads)

    # 7. validate
    ok = validate(out, out_dur, p)
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PipelineError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
