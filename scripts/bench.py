"""
Quick test bench for the transcript fix pass.

Cuts the first few minutes of each processed episode into a separate storage
folder (_BENCH_STORAGE), so changes to the LLM/verification pass can be tried in a
few minutes instead of re-running whole episodes. Real notes are never touched.

    .venv\\Scripts\\python scripts\\bench.py setup [--minutes 10] [--tail 5]   cut clips (start + end), transcribe once
    .venv\\Scripts\\python scripts\\bench.py run                    fix pass only, then score
    .venv\\Scripts\\python scripts\\bench.py score                  score the current bench notes

Every run starts from the same glossary snapshot, so runs are comparable. Each applied
correction is judged with the answer key (scripts/bench_key.json: right / damage), else
against the publisher's transcript at that moment (better / same / worse than what was
heard). Score = right - damage; wrong forms still in the transcripts are counted too.
Results go to runs/<time>.json; corrections that changed since the previous run are printed.
"""

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "_BRAIN_STORAGE"
BENCH = ROOT / "_BENCH_STORAGE"

# Point Zettlecast at the bench storage before it is imported
os.environ["STORAGE_PATH"] = str(BENCH)
os.environ["LANCEDB_PATH"] = str(BENCH / ".lancedb")
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

import zettlecast  # noqa: E402,F401  (imports torch first; see AGENTS.md)
from zettlecast.podcast.reference import normalize, parse_reference  # noqa: E402

CORRECTION = re.compile(r"^- \[(\d+)s\] (.*?) → \*\*(.*?)\*\* \(([\w+]+), (\w+), penalty")


def setup(minutes: int, tail: int) -> None:
    from zettlecast.podcast.queue import TranscriptionQueue

    real = json.loads((REAL / "podcast_queue.json").read_text(encoding="utf-8"))
    if BENCH.exists():
        shutil.rmtree(BENCH)
    queue = TranscriptionQueue(state_file=BENCH / "podcast_queue.json")
    for item in real["items"].values():
        ep = item["episode"]
        src = Path(ep["audio_path"])
        dest = BENCH / "podcasts" / (ep.get("podcast_name") or "default") / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", str(minutes * 60), "-c", "copy", str(dest)],
            check=True,
        )
        # the publisher's transcript keeps its times: the clip starts at 0
        for ref in src.parent.glob(f"{src.stem}.transcript.*"):
            shutil.copy2(ref, dest.with_name(ref.name))
        clips = [(dest, ep.get("episode_title"))]
        if tail:
            # the outro too (thanks, sign-offs, sponsor reads): the last minutes, times shifted to 0
            end = dest.with_name(f"{src.stem} (end){src.suffix}")
            duration = float(subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)],
                capture_output=True, text=True, check=True,
            ).stdout)
            offset = max(0.0, duration - tail * 60)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{offset:.3f}", "-i", str(src),
                            "-c", "copy", str(end)], check=True)
            ref = src.with_name(f"{src.stem}.transcript.json")
            if ref.exists():
                data = json.loads(ref.read_text(encoding="utf-8"))
                data["segments"] = [
                    {**s, "startTime": s["startTime"] - offset, "endTime": s["endTime"] - offset}
                    for s in data.get("segments", []) if s.get("startTime", 0) >= offset
                ]
                end.with_name(f"{end.stem}.transcript.json").write_text(json.dumps(data), encoding="utf-8")
            clips.append((end, f"{ep.get('episode_title') or src.stem} (end)"))
        for clip, title in clips:
            queue.add(clip, podcast_name=ep.get("podcast_name"), episode_title=title,
                      feed_url=ep.get("feed_url"), description=ep.get("description"),
                      published=ep.get("published"), transcript_url=ep.get("transcript_url"))
        print(f"  {minutes} min of {src.stem}" + (f" + last {tail} min" if tail else ""))
    for sub in ("glossaries", "lookups"):
        if (REAL / sub).exists():
            shutil.copytree(REAL / sub, BENCH / sub)
    if (BENCH / "glossaries").exists():
        shutil.copytree(BENCH / "glossaries", BENCH / "glossaries.seed")
    print("Transcribing the clips once (later runs reuse it)...")
    run()


def run() -> None:
    from zettlecast.pipeline import run_podcast_queue
    from zettlecast.podcast.queue import TranscriptionQueue

    seed = BENCH / "glossaries.seed"
    if seed.exists():
        shutil.rmtree(BENCH / "glossaries", ignore_errors=True)
        shutil.copytree(seed, BENCH / "glossaries")
    queue = TranscriptionQueue()
    queue.reprocess()
    queue.retry_failed()
    started = time.time()
    summary = run_podcast_queue(sync=False, on_episode_done=lambda o: print(
        f"  {o.status}: {o.title[:60]} ({o.processing_seconds:.0f}s)"
    ))
    print(f"{summary.completed} done, {summary.failed} failed in {time.time() - started:.0f}s")
    score()


def _key_verdict(original: str, chosen: str, window: list, key: dict) -> str:
    """
    right / damage from the answer key, or "" when the key says nothing. The key says how
    names are spelled, not where they occur, so a newly introduced key spelling only counts as
    right when the publisher heard a similar word there ("Kube" -> "Koop" with "Coop" heard).
    """
    def has(term: str, s: str) -> bool:
        return re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", s) is not None

    words = {w for term in key["correct"] for w in term.split() if w[:1].isupper()}
    if any(has(t, original) and not has(t, chosen) for t in key["correct"]) \
            or any(has(w, original) and not has(w, chosen) for w in words) \
            or any(has(t, chosen) and not has(t, original) for t in key["wrong"]):
        return "damage"
    introduced = [w for w in words if has(w, chosen) and not has(w, original)]
    heard_there = any(
        difflib.SequenceMatcher(None, w.lower(), h).ratio() >= 0.5 and min(len(w), len(h)) / max(len(w), len(h)) >= 0.6
        for w in introduced for h in window
    )
    if heard_there or any(has(t, original) and not has(t, chosen) for t in key["wrong"]):
        return "right"
    return ""


def score() -> None:
    key = json.loads((ROOT / "scripts" / "bench_key.json").read_text(encoding="utf-8"))
    rows, totals = [], {"right": 0, "damage": 0, "better": 0, "same": 0, "worse": 0, "review": 0}
    wrong_left: dict = {}
    for note in sorted((BENCH / "transcripts").glob("*.md")):
        text = note.read_text(encoding="utf-8")
        meta = yaml.safe_load(text.split("---")[1])
        audio = Path(meta["source"])
        refs = list(audio.parent.glob(f"{audio.stem}.transcript.*"))
        ref = parse_reference(refs[0].read_text(encoding="utf-8")) if refs else []
        transcript = text.split("## Transcript", 1)[-1]
        for form in key["wrong"]:
            n = len(re.findall(rf"(?<![\w-]){re.escape(form)}(?![\w-])", transcript))
            if n:
                wrong_left[form] = wrong_left.get(form, 0) + n
        review = text.split("## Needs Review", 1)[1].split("\n## ")[0] if "## Needs Review" in text else ""
        totals["review"] += sum(1 for line in review.splitlines() if line.startswith("- "))
        section = text.split("## Automatic Corrections", 1)[-1].split("\n## ")[0]
        for line in section.splitlines():
            m = CORRECTION.match(line)
            if not m:
                continue
            t, original, chosen, source = float(m[1]), m[2], m[3], m[4]
            window = [k for w in ref if t - 1.5 <= w.start <= t + 6 for k in normalize(w.text)]
            verdict = _key_verdict(original, chosen, window, key)
            if not verdict and ref:
                def hits(s: str) -> float:
                    words = normalize(s)
                    return sum(w in window for w in words) / max(len(words), 1)

                o, c = hits(original), hits(chosen)
                verdict = "better" if c > o else "worse" if c < o else "same"
            verdict = verdict or "no reference"
            totals[verdict] = totals.get(verdict, 0) + 1
            rows.append({"episode": note.stem[:40], "time": t, "original": original, "chosen": chosen,
                         "source": source, "verdict": verdict})

    runs = BENCH / "runs"
    runs.mkdir(exist_ok=True)
    previous = sorted(runs.glob("*.json"))
    out = runs / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    result = {"score": totals["right"] - totals["damage"], "totals": totals, "wrong_left": wrong_left,
              "corrections": rows}
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    for r in rows:
        if r["verdict"] in ("damage", "worse"):
            print(f"  {r['verdict']:7}{r['episode'][:30]:30} {r['time']:6.0f}s  {r['original']!r} -> {r['chosen']!r} ({r['source']})")
    if previous:
        def key_of(r):
            return (r["episode"], r["original"], r["chosen"])

        before = {key_of(r) for r in json.loads(previous[-1].read_text(encoding="utf-8"))["corrections"]}
        now = {key_of(r) for r in rows}
        for k in sorted(now - before):
            print(f"  + new      {k[0][:30]:30} {k[1]!r} -> {k[2]!r}")
        for k in sorted(before - now):
            print(f"  - dropped  {k[0][:30]:30} {k[1]!r} -> {k[2]!r}")
    print(f"Wrong forms left in transcripts: {wrong_left or 'none'}")
    print(f"Score (right - damage): {result['score']}  {totals}  (saved {out.relative_to(ROOT)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--minutes", type=int, default=10)
    p.add_argument("--tail", type=int, default=5, help="also clip the last N minutes (0 = no)")
    sub.add_parser("run")
    sub.add_parser("score")
    args = parser.parse_args()
    if args.cmd == "setup":
        setup(args.minutes, args.tail)
    elif args.cmd == "run":
        run()
    else:
        score()


if __name__ == "__main__":
    main()
