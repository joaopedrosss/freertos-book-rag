"""Baixa o livro "Mastering the FreeRTOS Real Time Kernel" e o divide em chunks.

Estratégia de chunking (orientada à estrutura do documento):
  - Cada seção numerada (##, ###, ####) vira uma unidade; títulos dentro de blocos
    de código (ex.: `#define`) são ignorados.
  - Seções longas são quebradas em janelas de ~MAX_WORDS palavras com sobreposição,
    sem cortar no meio de um bloco de código quando possível.
  - Cada chunk recebe um cabeçalho com o caminho "Capítulo › Seção › Subseção".
    Isso ajuda tanto a busca lexical quanto a semântica (o texto "Listing 4.2"
    sozinho não diz do que trata a seção).
  - Cada chunk guarda o número da seção e um link direto para o GitHub, usado nas citações.

Uso:
  python src/ingest.py                 # clona o livro (commit fixo) e gera data/chunks.jsonl
  python src/ingest.py --book_dir DIR  # usa um clone local
"""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

REPO = "https://github.com/FreeRTOS/FreeRTOS-Kernel-Book"
COMMIT = "ba20b2f820926587d5ee20ce312b3789d63f795c"  # fixado para resultados reprodutíveis
MAX_WORDS = 350
OVERLAP_WORDS = 60
MIN_WORDS = 25  # seções menores que isso são unidas à seguinte

HEADING = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")
SECTION_NUM = re.compile(r"^(\d+(?:\.\d+)*)\s+(.*)$")


def github_anchor(title: str) -> str:
    """Reproduz o slug que o GitHub gera para títulos Markdown."""
    s = title.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def clean(text: str) -> str:
    text = re.sub(r"<a name=[^>]*></a>", "", text)
    text = re.sub(r"<[^>]+>", "", text)  # tags HTML (figuras)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # imagens markdown
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)  # links → texto
    text = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", text)  # ênfase
    text = re.sub(r"\\([_*#<>\[\]])", r"\1", text)  # escapes do markdown: Heap\_5 → Heap_5
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_sections(md: str, fname: str):
    """Gera (caminho_de_títulos, título, nível, texto) para cada seção do capítulo."""
    sections, path, buf, in_code = [], [], [], False
    current = None

    def flush():
        if current is not None:
            sections.append({**current, "text": clean("\n".join(buf))})

    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), re.sub(r"\\(.)", r"\1", m.group(2)).replace("*", "").strip()
            path = path[: level - 1] + [title]
            num = SECTION_NUM.match(title)
            current = {
                "file": fname,
                "level": level,
                "title": title,
                "section": num.group(1) if num else title,
                "breadcrumb": " › ".join(path),
                "url": f"{REPO}/blob/{COMMIT}/{fname}#{github_anchor(title)}",
            }
            buf = []
        else:
            buf.append(line)
    flush()
    return sections


def windows(text: str):
    """Quebra um texto longo em janelas com sobreposição, preferindo fronteiras de parágrafo."""
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], []
    for p in paras:
        words = p.split()
        if cur and len(cur) + len(words) > MAX_WORDS:
            chunks.append(cur)
            cur = cur[-OVERLAP_WORDS:]
        cur = cur + words
        while len(cur) > MAX_WORDS * 1.5:  # parágrafo/listagem gigante
            chunks.append(cur[:MAX_WORDS])
            cur = cur[MAX_WORDS - OVERLAP_WORDS:]
    if cur:
        chunks.append(cur)
    return [" ".join(c) for c in chunks]


def build_chunks(book_dir: Path):
    chunks = []
    for f in sorted(book_dir.glob("ch[0-9][0-9].md")):
        sections = parse_sections(f.read_text(encoding="utf-8"), f.name)
        carry = None
        for s in sections:
            if carry:  # seção curta anterior (ex.: "4.1 Introduction" vazia) junta-se a esta
                s = {**s, "text": (carry["text"] + "\n\n" + s["text"]).strip()}
                carry = None
            if len(s["text"].split()) < MIN_WORDS and s is not sections[-1]:
                carry = s
                continue
            parts = windows(s["text"])
            for i, part in enumerate(parts):
                chunks.append({
                    "id": f"{s['section']}#{i}",
                    "section": s["section"],
                    "title": s["title"],
                    "breadcrumb": s["breadcrumb"],
                    "url": s["url"],
                    "text": f"{s['breadcrumb']}\n\n{part}",
                })
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book_dir", default=None)
    ap.add_argument("--out", default="data/chunks.jsonl")
    args = ap.parse_args()

    if args.book_dir:
        book = Path(args.book_dir)
    else:
        tmp = Path(tempfile.mkdtemp())
        subprocess.run(["git", "clone", "-q", REPO, str(tmp / "book")], check=True)
        subprocess.run(["git", "-C", str(tmp / "book"), "checkout", "-q", COMMIT], check=True)
        book = tmp / "book"

    chunks = build_chunks(book)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    sizes = sorted(len(c["text"].split()) for c in chunks)
    print(f"{len(chunks)} chunks de {len({c['section'] for c in chunks})} seções → {out}")
    print(f"palavras por chunk: mediana {sizes[len(sizes) // 2]}, máx {sizes[-1]}")


if __name__ == "__main__":
    main()
