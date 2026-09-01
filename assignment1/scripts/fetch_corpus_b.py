"""Regenerates data/python_stdlib_source.txt (Group B comparison corpus:
non-literary text). Deterministic given the same Python installation.

Run from the assignment1/ directory:
    python3 scripts/fetch_corpus_b.py
"""
import os
import random
import sysconfig

TARGET_CHARS = 260_000  # a bit above the assignment's 200 kB minimum
OUT_PATH = "data/python_stdlib_source.txt"


def main():
    stdlib = sysconfig.get_paths()["stdlib"]
    files = []
    for root, _dirs, fnames in os.walk(stdlib):
        if "test" in root or "site-packages" in root or "__pycache__" in root:
            continue
        for fname in fnames:
            if fname.endswith(".py"):
                files.append(os.path.join(root, fname))
    files.sort()  # deterministic starting order
    random.seed(42)
    random.shuffle(files)

    chunks = []
    total = 0
    n_files = 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        rel = os.path.relpath(path, stdlib)
        chunks.append(f"# --- {rel} ---\n")
        chunks.append(content)
        total += len(content)
        n_files += 1
        if total >= TARGET_CHARS:
            break

    text = "".join(chunks)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {OUT_PATH}: {len(text)} chars from {n_files} files")


if __name__ == "__main__":
    main()
