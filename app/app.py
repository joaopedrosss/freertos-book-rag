"""Demo Gradio: pergunte sobre o FreeRTOS em português ou inglês.

Local (na raiz do repositório):
  export GROQ_API_KEY=...            # ou LLM_PROVIDER=ollama para rodar 100% local
  python app/app.py

Hugging Face Spaces: veja app/README.md (as chaves vão em Settings → Secrets).
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "src").exists():  # no Space, src/ e data/ ficam ao lado de app.py
    ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import gradio as gr  # noqa: E402
from llm import LLM  # noqa: E402
from rag import RAG  # noqa: E402
from retrieval import build_retriever  # noqa: E402

RAG_PIPELINE = RAG(build_retriever(os.environ.get("RETRIEVER", "hybrid")), LLM(), k=5)


def linkify(answer: str, sources) -> str:
    urls = {s["n"]: s["url"] for s in sources}
    return re.sub(r"\[(\d+)\]", lambda m: f"[[{m.group(1)}]]({urls[int(m.group(1))]})"
                  if int(m.group(1)) in urls else m.group(0), answer)


def ask(question: str):
    if not question or not question.strip():
        return "", ""
    out = RAG_PIPELINE.answer(question.strip())
    answer_md = linkify(out["answer"], out["sources"])
    lines = []
    for s in out["sources"]:
        star = "★ " if s["n"] in out["cited"] else ""
        lines.append(f"{star}**[{s['n']}]** [{s['title']}]({s['url']})")
    return answer_md, "\n\n".join(lines)


with gr.Blocks(title="FreeRTOS Book RAG") as demo:
    gr.Markdown(
        "# Pergunte ao livro do FreeRTOS\n"
        "Respostas geradas **somente** a partir do livro *Mastering the FreeRTOS Real Time Kernel* "
        "(CC BY-SA 4.0), com citações para cada seção usada. Funciona em português e inglês."
    )
    q = gr.Textbox(label="Pergunta", placeholder="Ex.: Qual a diferença entre mutex e semáforo binário?", lines=2)
    btn = gr.Button("Perguntar", variant="primary")
    answer = gr.Markdown(label="Resposta")
    gr.Markdown("### Trechos recuperados (★ = citado na resposta)")
    sources = gr.Markdown()
    gr.Examples(
        [["Qual a diferença entre vTaskDelay() e vTaskDelayUntil()?"],
         ["O que é inversão de prioridade e como o mutex ajuda?"],
         ["When should I use task notifications instead of a queue?"],
         ["Como detectar estouro de pilha de uma tarefa?"]],
        inputs=q,
    )
    btn.click(ask, q, [answer, sources])
    q.submit(ask, q, [answer, sources])

if __name__ == "__main__":
    demo.launch()
