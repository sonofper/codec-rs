# Benchmarks the codec and keeps a history of the results for every commit.
#
#     bench.py run    <codec binary> <commit> <history.csv> [test files dir]
#     bench.py render <history.csv> <BENCHMARKS.md>
#
# Use the committed test files (LF), not a Windows checkout (CRLF), or the sizes
# won't match CI: git show HEAD:test_files/x > dir/x
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["commit", "subject", "method", "file", "original", "compressed", "compress_ms",
          "decompress_ms", "roundtrip"]
# The competition is ranked on this file.
HEADLINE = "180kb.lyx"


def run(binary, commit, history, files):
    binary = str(Path(binary).resolve())
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sha = git("rev-parse", "--short=7", commit) or commit
    subject = git("show", "-s", "--format=%s", commit)
    rows = list(csv.DictReader(open(history))) if Path(history).exists() else []
    rows = [r for r in rows if r["commit"] != sha]

    methods = [line.split() for line in open(ROOT / "bench/methods.txt")
               if line.strip() and not line.startswith("#")]
    tmp = Path(tempfile.mkdtemp())
    for name, comp, decomp in methods:
        for f in sorted(Path(files).iterdir(), key=lambda p: p.stat().st_size):
            row = {"commit": sha, "subject": subject, "method": name, "file": f.name,
                   "original": f.stat().st_size, "compressed": "", "compress_ms": "",
                   "decompress_ms": "", "roundtrip": "CRASH"}
            t0 = time.perf_counter()
            if subprocess.run([binary, comp, f, tmp / "c"], capture_output=True).returncode == 0:
                t1 = time.perf_counter()
                row["compressed"] = (tmp / "c").stat().st_size
                row["compress_ms"] = round((t1 - t0) * 1000)
                if subprocess.run([binary, decomp, tmp / "c", tmp / "d"], capture_output=True).returncode == 0:
                    row["decompress_ms"] = round((time.perf_counter() - t1) * 1000)
                    same = (tmp / "d").read_bytes() == f.read_bytes()
                    row["roundtrip"] = "ok" if same else "MISMATCH"
            print(row["method"], row["file"], row["original"], "->", row["compressed"], row["roundtrip"])
            rows.append(row)

    with open(history, "w", newline="") as out:
        writer = csv.DictWriter(out, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render(history, out):
    rows = list(csv.DictReader(open(history)))
    commits = list(dict.fromkeys(r["commit"] for r in rows))
    files = list(dict.fromkeys(r["file"] for r in rows))
    subjects = {r["commit"]: r["subject"] for r in rows}

    # Smallest result that gave back the same file.
    def best(commit, file):
        ok = [r for r in rows if r["commit"] == commit and r["file"] == file and r["roundtrip"] == "ok"]
        return min(ok, key=lambda r: int(r["compressed"]), default=None)

    lines = ["# Benchmarks", "", "Made by `bench/bench.py` on every push to main. Sizes in bytes.", "",
             f"## {HEADLINE}", "", "| Commit | Change | Method | Size | Change in size |", "|---|---|---|---:|---:|"]
    previous = None
    for c in commits:
        b = best(c, HEADLINE)
        if b is None:
            lines.append(f"| `{c}` | {subjects[c]} | – | – | – |")
            continue
        size = int(b["compressed"])
        diff = "" if previous is None else f"{size - previous:+,}"
        lines.append(f"| `{c}` | {subjects[c]} | {b['method']} | {size:,} | {diff} |")
        previous = size

    lines += ["", "## All files (best method)", "", "| Commit | " + " | ".join(files) + " |",
              "|---|" + "---:|" * len(files)]
    for c in commits:
        sizes = [best(c, f) for f in files]
        lines.append(f"| `{c}` | " + " | ".join("–" if b is None else f"{int(b['compressed']):,}" for b in sizes) + " |")
    Path(out).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else ROOT / "test_files")
    else:
        render(sys.argv[2], sys.argv[3])
