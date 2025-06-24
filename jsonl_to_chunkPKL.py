import json, pickle

JSONL_FILE = "./json_chunk/all_data.jsonl"   # ton JSONL d’origine
CHUNKS_PKL = "tindle_chunks.pkl"

chunks = {}
with open(JSONL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        cid  = obj["id"]
        text = obj["text"]
        meta = obj.get("metadata", {})
        offset = meta.get("offset", 0)
        chunks[cid] = {
            "text":   text,
            "offset": offset,
            **{k:v for k,v in meta.items() if k!="offset"}
        }

with open(CHUNKS_PKL, "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ {len(chunks)} chunks saved to {CHUNKS_PKL}")