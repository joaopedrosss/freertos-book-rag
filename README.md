# FreeRTOS Book RAG — perguntas em português sobre documentação técnica em inglês

Sistema de **Retrieval-Augmented Generation** que responde perguntas sobre o kernel FreeRTOS usando
**somente** o livro oficial *[Mastering the FreeRTOS Real Time Kernel](https://github.com/FreeRTOS/FreeRTOS-Kernel-Book)*,
com **citação da seção** que sustenta cada afirmação e **recusa** quando o livro não tem a resposta.

O foco do projeto é a **avaliação**: um conjunto de 35 perguntas com respostas de referência
mede separadamente a recuperação e a geração, e compara três estratégias de busca.

- 🕹️ **Demo:** [Hugging Face Space](https://huggingface.co/spaces/joaopedrosss/freertos-book-rag)
- 💸 **Custo:** zero — embeddings rodam em CPU; o LLM pode ser Groq/Gemini (plano gratuito) ou Ollama local.

```
pergunta ──► BM25 ─────────┐
   (PT/EN)                 ├─► RRF ─► top-5 trechos ─► LLM + prompt com regras ─► resposta com [n]
          └► e5 multilíngue┘         (seção + link)      (só o contexto, cite, recuse)   + links das seções
```

## Resultados

### Recuperação — a seção correta aparece entre os trechos recuperados?

30 perguntas, cada uma escrita em inglês **e** em português, com as seções "ouro" que a respondem.

| Recuperador | Idioma | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---|---|---|---|---|
| BM25 | EN | 0.933 | 0.967 | 0.967 | 0.944 |
| BM25 | **PT** | 0.300 | 0.600 | 0.600 | 0.433 |
| Dense (multilingual-e5-small) | EN | TODO | TODO | TODO | TODO |
| Dense (multilingual-e5-small) | PT | TODO | TODO | TODO | TODO |
| Híbrido (RRF) | EN | TODO | TODO | TODO | TODO |
| Híbrido (RRF) | PT | TODO | TODO | TODO | TODO |

<!-- TODO: preencha com reports/retrieval_metrics.json depois de rodar `python src/evaluate.py`. -->

**Leitura:** a busca lexical é excelente em inglês (as perguntas usam o vocabulário do livro e nomes de
API como `xQueueOverwrite()`), mas cai pela metade em português: "inversão de prioridade" não
compartilha palavras com "priority inversion". Os embeddings multilíngues resolvem isso; o híbrido
busca manter a precisão do BM25 em nomes de API sem perder o português.

### Geração — a resposta final está correta e bem citada?

| Métrica | Valor |
|---|---|
| Nota média (LLM-juiz vs. referência, 0–1) | TODO |
| Respostas totalmente corretas | TODO |
| Cita uma seção ouro | TODO |
| Recusa indevida (pergunta respondível) | TODO |
| Recusa correta (5 perguntas fora do escopo) | TODO |

<!-- TODO: preencha com reports/generation_summary_hybrid_pt.json. Informe o LLM usado. -->

## Decisões de projeto

- **Chunking pela estrutura do documento.** Cada seção numerada vira uma unidade (seções longas viram
  janelas de ~350 palavras com sobreposição). Cada chunk leva o caminho
  `Capítulo › Seção › Subseção` no início: o texto "Listing 5.14" sozinho não diz do que a seção trata.
  Resultado: 409 chunks de 227 seções.
- **BM25 próprio** em vez de biblioteca, para tokenizar identificadores do FreeRTOS:
  `xQueueSendToBack` vira `xqueuesendtoback, queue, send, back`.
- **Busca híbrida com Reciprocal Rank Fusion**, que combina rankings sem calibrar escalas de score diferentes.
- **Sem banco vetorial:** com 409 vetores, a busca é um produto de matrizes em NumPy (< 1 ms).
  FAISS/pgvector passam a valer a pena com dezenas de milhares de chunks.
- **Recusa explícita:** o prompt exige uma frase fixa quando o contexto não responde. Isso torna a
  recusa mensurável e evita respostas inventadas.
- **Livro fixado num commit**, para os resultados serem reprodutíveis.

## Como rodar

```bash
pip install -r requirements.txt
python src/ingest.py                       # baixa o livro e gera data/chunks.jsonl (já incluso)
python src/evaluate.py                     # métricas de recuperação (CPU, ~2 min na 1ª vez)

export GROQ_API_KEY=...                    # chave gratuita em console.groq.com
python src/rag.py "Qual a diferença entre mutex e semáforo binário?"
python src/evaluate.py --methods hybrid --generate --sleep 4   # avaliação da geração (~65 chamadas)
python app/app.py                          # interface web local
```

**Outros provedores:** `LLM_PROVIDER=gemini` + `GEMINI_API_KEY`, ou `LLM_PROVIDER=ollama` para rodar
100% local (`ollama pull qwen2.5:3b`). Qualquer API compatível com OpenAI funciona com
`LLM_PROVIDER=custom LLM_BASE_URL=... LLM_MODEL=...`.

## Estrutura

```
src/ingest.py      download do livro (commit fixo) e chunking por seção
src/retrieval.py   BM25, Dense (e5 multilíngue) e Híbrido (RRF)
src/llm.py         cliente OpenAI-compatível (Groq, Gemini, Ollama)
src/rag.py         prompt, geração, extração de citações e detecção de recusa
src/evaluate.py    métricas de recuperação e de geração (LLM-juiz)
eval/questions.jsonl  30 perguntas PT/EN com seções ouro e referências + 5 fora do escopo
app/               demo Gradio para Hugging Face Spaces
reports/           métricas e respostas detalhadas
```

## Limitações

- **Conjunto pequeno (35 perguntas)**: diferenças de 1–2 perguntas mudam as métricas em 3–7 pontos.
- **Viés a favor do BM25 em inglês:** as perguntas foram escritas por quem leu o livro, com o mesmo
  vocabulário. Usuários reais fariam perguntas mais vagas.
- **LLM-juiz** pode ser leniente ou rígido; as notas brutas e justificativas ficam em
  `reports/generation_results_*.jsonl` para conferência manual.
- Cobre só o livro do kernel, não a referência completa da API nem ports específicos (ex.: ESP-IDF).

## Próximos passos

- Reranker (cross-encoder) sobre os top-20 do híbrido.
- Ampliar as perguntas com dúvidas reais de fóruns (ex.: FreeRTOS forums, Stack Overflow).
- Indexar a documentação do ESP-IDF para perguntas específicas do ESP32.

## Licença

Código: MIT. Livro e chunks derivados (`data/chunks.jsonl`): CC BY-SA 4.0, © Amazon.com, Inc. ou afiliadas.
