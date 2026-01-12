import json
import http.client
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from constants import DATA_DOC_NAME

CHUNKS_PATH = Path(f"data/{DATA_DOC_NAME}.chunks.jsonl")
EMB_PATH = Path(f"data/{DATA_DOC_NAME}.embeddings.npy")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

HISTORY_TURNS = 6  # how many Q/A pairs to keep

def call_llm(prompt: str) -> str:
    conn = http.client.HTTPConnection("localhost", 11434)
    body = {
        "model": "qwen3:8b",
        "prompt": prompt,
        "stream": False,
    }
    conn.request("POST", "/api/generate", json.dumps(body))
    resp = json.loads(conn.getresponse().read())
    return resp["response"]

def load_chunks(path: Path):
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            obj = json.loads(line)
            text = obj.get("text") or obj.get("content") or ""
            meta = obj.get("metadata") or {}
            chunk_id = obj.get("id") or obj.get("chunk_id") or str(i)
            chunks.append({"id": chunk_id, "text": text, "metadata": meta})
    return chunks

def format_history(history: list[dict], max_turns: int) -> str:
    # history items: {"q": "...", "a": "..."}
    recent = history[-max_turns:]
    if not recent:
        return "(none)"
    lines = []
    for i, t in enumerate(recent, start=1):
        lines.append(f"Turn {i} - User: {t['q']}")
        lines.append(f"Turn {i} - Assistant: {t['a']}")
    return "\n".join(lines)

def build_prompt(question: str, contexts: list[dict], history: list[dict]) -> str:
    ctx_blocks = []
    for c in contexts:
        m = c["metadata"] or {}
        header = (
            f"[id={c['id']} | chapter={m.get('chapter')} | section={m.get('section')} | "
            f"pages={m.get('page_start')}-{m.get('page_end')}]"
        )
        ctx_blocks.append(header + "\n" + c["text"].strip())

    context_text = "\n\n---\n\n".join(ctx_blocks)
    history_text = format_history(history, HISTORY_TURNS)

    return f"""
You are a PostgreSQL tutor.
Answer using ONLY the provided CONTEXT for factual claims.
Use CHAT HISTORY only to understand what the user refers to (pronouns, "that", follow-ups).
Answer naturally and concisely.

========================
INTERNAL CONTEXT FORMAT (DO NOT OUTPUT)
Each context block has an internal identifier like:
[id=FILE|CHAPTER|PAGES|PART]

Meaning:
- CHAPTER is the chapter or section title (example: "1.3. Creating a Database")
- PAGES is the page range in the PDF (example: "2-4")

This format is INTERNAL ONLY.
NEVER output strings in square brackets.
NEVER output "id=", file paths, or "part".

========================
OUTPUT RULES
- You MAY mention chapter titles and page numbers in plain human language.
- Allowed example:
  This is explained in the section '1.3. Creating a Database' (pages 2–4).
- Forbidden:
  [id=...|1.3. Creating a Database|p2-4|part1]

========================
If the context is insufficient:
Say you don't know and ask ONE precise clarifying question.

CHAT HISTORY:
{history_text}

QUESTION:
{question}

CONTEXT:
{context_text}

ANSWER:
""".strip()

def retrieve(question: str, chunks: list[dict], chunk_emb: np.ndarray, embedder, top_k: int = 5):
    q_emb = embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
    scores = chunk_emb @ q_emb
    top_idx = np.argsort(-scores)[:top_k]
    results = []
    for idx in top_idx:
        c = chunks[int(idx)]
        results.append({**c, "score": float(scores[int(idx)])})
    return results

def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing {CHUNKS_PATH}")
    if not EMB_PATH.exists():
        raise FileNotFoundError(f"Missing {EMB_PATH}")

    chunks = load_chunks(CHUNKS_PATH)
    emb = np.load(EMB_PATH)

    if emb.shape[0] != len(chunks):
        raise ValueError(f"Embeddings rows ({emb.shape[0]}) != chunks ({len(chunks)})")

    embedder = SentenceTransformer(MODEL_NAME)

    history: list[dict] = []

    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if not q or q.lower() == "exit":
            break

        # Retrieval can optionally include history; start simple with current q only
        ctx = retrieve(q, chunks, emb, embedder, top_k=5)

        print("\nTop contexts:")
        for c in ctx:
            m = c["metadata"] or {}
            print(f"- score={c['score']:.3f} id={c['id']} section={m.get('section')} pages={m.get('page_start')}-{m.get('page_end')}")

        prompt = build_prompt(q, ctx, history)
        answer = call_llm(prompt)
        print("\nAnswer:\n", answer)

        history.append({"q": q, "a": answer})

if __name__ == "__main__":
    main()
