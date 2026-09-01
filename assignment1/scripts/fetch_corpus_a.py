"""Regenerates data/pride_and_prejudice.txt (Group A comparison corpus:
modern English prose), downloaded from Project Gutenberg with the
boilerplate header/footer stripped.

Run from the assignment1/ directory:
    python3 scripts/fetch_corpus_a.py
"""
import os
import urllib.request

URL = "https://www.gutenberg.org/files/1342/1342-0.txt"
OUT_PATH = "data/pride_and_prejudice.txt"
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK"


def main():
    raw_path = "data/_pride_and_prejudice_raw.txt"
    os.makedirs("data", exist_ok=True)
    urllib.request.urlretrieve(URL, raw_path)

    with open(raw_path, encoding="utf-8") as fh:
        text = fh.read()

    start = text.find(START_MARKER)
    start = text.find("\n", start) + 1
    end = text.find(END_MARKER)
    body = text[start:end].strip()

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(body)

    os.remove(raw_path)
    print(f"wrote {OUT_PATH}: {len(body)} chars")


if __name__ == "__main__":
    main()
