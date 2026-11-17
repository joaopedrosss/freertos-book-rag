"""Pipeline RAG: recupera trechos do livro e gera uma resposta com citações.

Uso:
  python src/rag.py "Qual a diferença entre mutex e semáforo binário?"
  python src/rag.py --retriever bm25 --k 5 "What is priority inversion?"
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import build_retriever  # noqa: E402

NOT_FOUND_PT = "Não encontrei essa informação no livro."
NOT_FOUND_EN = "I couldn't find this information in the book."

SYSTEM_PROMPT = f"""Você é um assistente técnico que responde perguntas sobre o kernel FreeRTOS
usando SOMENTE os trechos do livro "Mastering the FreeRTOS Real Time Kernel" fornecidos.

Regras:
1. Use apenas informações presentes nos trechos. Não use conhecimento externo.
2. Cite as fontes com o número do trecho entre colchetes, ex.: [1] ou [2][3], logo após a frase que ele sustenta.
3. Responda no mesmo idioma da pergunta. Mantenha nomes de APIs e constantes como estão (ex.: xQueueSend()).
4. Seja direto: 2 a 6 frases. Use uma lista curta só se a pergunta pedir passos ou comparação.
5. Se os trechos não contêm a resposta, responda exatamente "{NOT_FOUND_PT}" (pergunta em português)
   ou "{NOT_FOUND_EN}" (pergunta em inglês), sem citações."""


def format_context(hits):
    blocks = []
    for n, (chunk, _score) in enumerate(hits, 1):
        blocks.append(f"[{n}] (Seção {chunk['section']})\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks)


def is_refusal(answer: str) -> bool:
    a = answer.strip().lower()
    return a.startswith(NOT_FOUND_PT.lower()[:25]) or a.startswith(NOT_FOUND_EN.lower()[:25])


def cited_indices(answer: str, n_sources: int):
    found = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    return sorted(i for i in found if 1 <= i <= n_sources)


class RAG:
    def __init__(self, retriever, llm, k=5):
        self.retriever, self.llm, self.k = retriever, llm, k

    def answer(self, question: str) -> dict:
        hits = self.retriever.search(question, k=self.k)
        user = f"Trechos do livro:\n\n{format_context(hits)}\n\n---\n\nPergunta: {question}"
        text = self.llm.chat(SYSTEM_PROMPT, user)
        cited = cited_indices(text, len(hits))
        return {
            "question": question,
            "answer": text,
            "refused": is_refusal(text),
            "cited": cited,
            "sources": [
                {"n": n, "section": c["section"], "title": c["title"], "url": c["url"], "score": round(s, 4)}
                for n, (c, s) in enumerate(hits, 1)
            ],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--retriever", default="hybrid", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    from llm import LLM

    rag = RAG(build_retriever(args.retriever), LLM(), k=args.k)
    out = rag.answer(args.question)
    print("\n" + out["answer"] + "\n")
    for s in out["sources"]:
        mark = "★" if s["n"] in out["cited"] else " "
        print(f"{mark} [{s['n']}] {s['title']}  {s['url']}")


if __name__ == "__main__":
    main()
