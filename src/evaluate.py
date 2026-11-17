"""Avaliação do sistema RAG em duas camadas.

1) Recuperação (não precisa de LLM, roda em segundos):
   Para cada pergunta, em português e em inglês, verifica se algum dos k trechos
   recuperados pertence a uma das seções "ouro". Métricas: Hit@1, Hit@3, Hit@5, MRR@10.

2) Geração (precisa de LLM; use --generate):
   - Correção: um LLM-juiz compara a resposta com a resposta de referência (0, 1 ou 2).
   - Citação correta: a resposta cita ao menos um trecho de uma seção ouro.
   - Recusa indevida: o sistema disse "não encontrei" para uma pergunta que o livro responde.
   - Recusa correta: o sistema disse "não encontrei" para perguntas fora do escopo do livro.

Uso:
  python src/evaluate.py                                # recuperação: bm25, dense e hybrid
  python src/evaluate.py --methods bm25                 # só BM25 (sem baixar modelos)
  python src/evaluate.py --generate --retriever hybrid  # + geração (usa LLM_PROVIDER etc.)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import BM25, Dense, Hybrid, load_chunks  # noqa: E402

JUDGE_PROMPT = """Você avalia respostas de um assistente sobre o FreeRTOS.
Compare a RESPOSTA com a REFERÊNCIA (que é correta) e dê uma nota:
2 = correta e cobre o ponto principal da referência;
1 = parcialmente correta ou incompleta, sem erros graves;
0 = errada, contradiz a referência, ou não responde.
Ignore estilo, idioma e citações. Responda APENAS com JSON: {"score": <0|1|2>, "reason": "<uma frase>"}"""


def load_questions(path="eval/questions.jsonl"):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def retrieval_metrics(retriever, questions, lang):
    hits = {1: 0, 3: 0, 5: 0}
    rr = 0.0
    per_q = []
    for q in questions:
        results = retriever.search(q[f"q_{lang}"], k=10)
        sections = [c["section"] for c, _ in results]
        rank = next((i + 1 for i, s in enumerate(sections) if s in q["gold_sections"]), None)
        for k in hits:
            hits[k] += int(rank is not None and rank <= k)
        rr += 1 / rank if rank else 0
        per_q.append({"id": q["id"], "lang": lang, "rank": rank, "top3": sections[:3]})
    n = len(questions)
    return {**{f"hit@{k}": round(v / n, 3) for k, v in hits.items()}, "mrr@10": round(rr / n, 3)}, per_q


def judge(llm, question, reference, answer):
    raw = llm.chat(JUDGE_PROMPT, f"PERGUNTA: {question}\n\nREFERÊNCIA: {reference}\n\nRESPOSTA: {answer}", max_tokens=150)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
        return int(data.get("score", 0)), data.get("reason", raw)
    except (ValueError, TypeError):
        return 0, f"juiz retornou formato inválido: {raw[:120]}"


def run_generation(retriever, questions, lang, k, sleep):
    from llm import LLM
    from rag import RAG

    llm = LLM()
    rag = RAG(retriever, llm, k=k)
    rows = []
    for i, q in enumerate(questions, 1):
        out = rag.answer(q[f"q_{lang}"])
        cited_sections = {out["sources"][n - 1]["section"] for n in out["cited"]}
        row = {"id": q["id"], "in_scope": q["in_scope"], "question": out["question"], "answer": out["answer"],
               "refused": out["refused"], "cited_sections": sorted(cited_sections),
               "gold_sections": q["gold_sections"]}
        if q["in_scope"]:
            row["cited_gold"] = bool(cited_sections & set(q["gold_sections"]))
            if out["refused"]:
                row["score"], row["judge_reason"] = 0, "recusou uma pergunta respondível"
            else:
                time.sleep(sleep)
                row["score"], row["judge_reason"] = judge(llm, q[f"q_{lang}"], q["reference"], out["answer"])
        print(f"  [{i}/{len(questions)}] {q['id']}: "
              + (f"nota={row.get('score')} cita_ouro={row.get('cited_gold')}" if q["in_scope"]
                 else f"recusou={row['refused']}"))
        rows.append(row)
        time.sleep(sleep)

    ins = [r for r in rows if r["in_scope"]]
    oos = [r for r in rows if not r["in_scope"]]
    summary = {
        "llm": repr(llm),
        "answer_score": round(sum(r["score"] for r in ins) / (2 * len(ins)), 3),
        "fully_correct": round(sum(r["score"] == 2 for r in ins) / len(ins), 3),
        "cited_gold": round(sum(r["cited_gold"] for r in ins) / len(ins), 3),
        "false_refusal": round(sum(r["refused"] for r in ins) / len(ins), 3),
        "oos_refusal": round(sum(r["refused"] for r in oos) / len(oos), 3) if oos else None,
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="bm25,dense,hybrid")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--retriever", default="hybrid", help="recuperador usado na geração")
    ap.add_argument("--lang", default="pt", choices=["pt", "en"], help="idioma das perguntas na geração")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=1.0, help="pausa entre chamadas (limites de taxa)")
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()

    chunks = load_chunks()
    questions = load_questions()
    in_scope = [q for q in questions if q["in_scope"]]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    built = {}
    methods = args.methods.split(",")
    if "bm25" in methods or "hybrid" in methods:
        built["bm25"] = BM25(chunks)
    if "dense" in methods or "hybrid" in methods:
        built["dense"] = Dense(chunks)
    if "hybrid" in methods:
        built["hybrid"] = Hybrid([built["bm25"], built["dense"]])

    table, details = {}, []
    for name in methods:
        for lang in ["en", "pt"]:
            m, per_q = retrieval_metrics(built[name], in_scope, lang)
            table[f"{name}/{lang}"] = m
            details += [{"method": name, **d} for d in per_q]
            print(f"{name:7s} {lang}: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    (out / "retrieval_metrics.json").write_text(json.dumps(table, indent=2))
    with (out / "retrieval_details.jsonl").open("w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    if args.generate:
        if args.retriever not in built:
            raise SystemExit(f"Inclua '{args.retriever}' em --methods para usá-lo na geração.")
        retr = built[args.retriever]
        print(f"\nGeração com {args.retriever}, k={args.k}, perguntas em {args.lang}:")
        summary, rows = run_generation(retr, questions, args.lang, args.k, args.sleep)
        summary.update({"retriever": args.retriever, "k": args.k, "lang": args.lang})
        tag = f"{args.retriever}_{args.lang}"
        (out / f"generation_summary_{tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        with (out / f"generation_results_{tag}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
