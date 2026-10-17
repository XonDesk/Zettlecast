"""
Quick test bench for the transcript fix pass.

Cuts the first few minutes of each processed episode into a separate storage
folder (_BENCH_STORAGE), so changes to the LLM/verification pass can be tried in a
few minutes instead of re-running whole episodes. Real notes are never touched.

    .venv\\Scripts\\python scripts\\bench.py setup [--minutes 10]   cut clips, transcribe them once
    .venv\\Scripts\\python scripts\\bench.py run                    fix pass only, then score
    .venv\\Scripts\\python scripts\\bench.py score                  score the current bench notes

Every run starts from the same glossary snapshot, so runs are comparable. The score
checks each applied correction against the publisher's transcript at that moment
(better / same / worse than what was heard) and writes runs/<time>.json; the
corrections that changed since the previous run are printed.
"""

import argparse
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


def setup(minutes: int) -> None:
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
        queue.add(dest, podcast_name=ep.get("podcast_name"), episode_title=ep.get("episode_title"),
                  feed_url=ep.get("feed_url"), description=ep.get("description"),
                  published=ep.get("published"), transcript_url=ep.get("transcript_url"))
        print(f"  {minutes} min of {src.stem}")
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


def score() -> None:
    rows, totals = [], {"better": 0, "same": 0, "worse": 0}
    for note in sorted((BENCH / "transcripts").glob("*.md")):
        text = note.read_text(encoding="utf-8")
        meta = yaml.safe_load(text.split("---")[1])
        audio = Path(meta["source"])
        refs = list(audio.parent.glob(f"{audio.stem}.transcript.*"))
        ref = parse_reference(refs[0].read_text(encoding="utf-8")) if refs else []
        section = text.split("## Automatic Corrections", 1)[-1].split("\n## ")[0]
        for line in section.splitlines():
            m = CORRECTION.match(line)
            if not m:
                continue
            t, original, chosen, source = float(m[1]), m[2], m[3], m[4]
            verdict = "no reference"
            if ref:
                window = [k for w in ref if t - 1.5 <= w.start <= t + 6 for k in normalize(w.text)]

                def hits(s: str) -> float:
                    words = normalize(s)
                    return sum(w in window for w in words) / max(len(words), 1)

                o, c = hits(original), hits(chosen)
                verdict = "better" if c > o else "worse" if c < o else "same"
                totals[verdict] += 1
            rows.append({"episode": note.stem[:40], "time": t, "original": original, "chosen": chosen,
                         "source": source, "verdict": verdict})

    runs = BENCH / "runs"
    runs.mkdir(exist_ok=True)
    previous = sorted(runs.glob("*.json"))
    out = runs / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"totals": totals, "corrections": rows}, indent=1, ensure_ascii=False), encoding="utf-8")

    for r in rows:
        if r["verdict"] == "worse":
            print(f"  worse  {r['episode'][:30]:30} {r['time']:6.0f}s  {r['original']!r} -> {r['chosen']!r} ({r['source']})")
    if previous:
        def key(r):
            return (r["episode"], r["original"], r["chosen"])

        before = {key(r) for r in json.loads(previous[-1].read_text(encoding="utf-8"))["corrections"]}
        now = {key(r) for r in rows}
        for k in sorted(now - before):
            print(f"  + new      {k[0][:30]:30} {k[1]!r} -> {k[2]!r}")
        for k in sorted(before - now):
            print(f"  - dropped  {k[0][:30]:30} {k[1]!r} -> {k[2]!r}")
    print(f"Corrections vs publisher: {totals}  (saved {out.relative_to(ROOT)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup")
    p.add_argument("--minutes", type=int, default=10)
    sub.add_parser("run")
    sub.add_parser("score")
    args = parser.parse_args()
    if args.cmd == "setup":
        setup(args.minutes)
    elif args.cmd == "run":
        run()
    else:
        score()


if __name__ == "__main__":
    main()
