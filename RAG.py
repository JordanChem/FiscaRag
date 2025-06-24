import os
import faiss
import pickle
import numpy as np
import openai
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

# —— CONFIG ——
load_dotenv()
# Use the new OpenAI client
client = OpenAI()
EMBED_MODEL           = "text-embedding-3-small"
CHAT_MODEL            = "o4-mini-2025-04-16"
FAISS_INDEX_FILE      = "tindle_index.faiss"
IDS_PKL               = "tindle_ids.pkl"
CHUNKS_PKL            = "tindle_chunks.pkl"
TOP_K                 = 10
MAX_TOKENS_CONTEXT    = 4000
SYSTEM_PROMPT = (
    "Tu es un assistant expert en droit fiscal. "
    "Fais d’abord appel aux passages fournis pour répondre. "
    "Si ces passages sont insuffisants, utilise tes connaissances générales en le précisant clairement."
)

# —— CHARGEMENT DE L'INDEX ——
index = faiss.read_index(FAISS_INDEX_FILE)
with open(IDS_PKL, "rb") as f:
    ids = pickle.load(f)
with open(CHUNKS_PKL, "rb") as f:
    chunks_dict = pickle.load(f)

# —— TOKEN COUNTER ——
enc = tiktoken.get_encoding("cl100k_base")
def num_tokens(s: str) -> int:
    return len(enc.encode(s))


# —— FONCTIONS RAG ——

def embed_question(question: str) -> list[float]:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[question]
    )
    # on récupère l'attribut .data, puis .embedding
    return resp.data[0].embedding


def retrieve_chunks(q_emb: list[float], k: int = TOP_K):
    xq = np.array([q_emb], dtype="float32")
    distances, indices = index.search(xq, k)
    out = []
    for dist, idx in zip(distances[0], indices[0]):
        cid  = ids[idx]
        meta = chunks_dict[cid]
        out.append({
            "score":  float(dist),
            "id":     cid,
            "offset": meta["offset"],
            "text":   meta["text"]
        })
    return out


def build_context(chunks, max_tokens=MAX_TOKENS_CONTEXT):
    parts, tokens = [], 0
    for c in sorted(chunks, key=lambda x: x["score"]):
        piece = f"(Source: {c['id']}@{c['offset']}) {c['text']}"
        nt = num_tokens(piece)
        if tokens + nt > max_tokens:
            break
        parts.append(piece)
        tokens += nt
    return "\n\n".join(parts)


def make_prompt(question: str, context: str):
    return [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user",   "content": f"Question: {question}\n\nContexte:\n{context}"}
    ]


def answer_question(question: str, k: int = TOP_K) -> str:
    # 1) Embed
    q_emb = embed_question(question)

    # 2) Retrieve
    top_chunks = retrieve_chunks(q_emb, k)

    # 3) Assemble
    context = build_context(top_chunks)

    # 4) Prompt
    messages = make_prompt(question, context)
    # 5) Call LLM
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages    )
    return resp.choices[0].message.content

# —— EXEMPLE ——
if __name__ == "__main__":
    question = "Quels sont les délais pour la réhabilitation d'hôtels en outre-mer ?"
    print(answer_question(question, k=10))