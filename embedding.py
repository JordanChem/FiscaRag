import os
import json
import asyncio
import time
import openai
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

# Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Paramètres
BATCH_SIZE = 1000  # Nombre de lignes par batch
CONCURRENCY = 2    # Réduire la concurrence pour éviter les rate limits
MAX_TOKENS_PER_TEXT = 8000  # Limite par texte

def text_cleaner(text):
    """Nettoie le texte pour l'API OpenAI"""
    if not isinstance(text, str):
        text = str(text)
    
    # Supprimer les caractères problématiques
    import re
    text = re.sub(r'[§°²³¹⁴⁵⁶⁷⁸⁹]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Limiter la longueur
    if len(text) > 32000:  # ~8000 tokens
        text = text[:32000]
    
    return text

async def embed_batch(texts, model="text-embedding-3-small", max_retries=3):
    """Embed un batch de textes avec gestion des rate limits"""
    # Nettoyer et valider les textes
    clean_texts = []
    valid_indices = []
    
    for i, text in enumerate(texts):
        if text and isinstance(text, str):
            clean_text = text_cleaner(text)
            if clean_text.strip():
                # Vérifier la longueur en tokens
                encoding = tiktoken.encoding_for_model(model)
                tokens = len(encoding.encode(clean_text))
                if tokens <= MAX_TOKENS_PER_TEXT:
                    clean_texts.append(clean_text)
                    valid_indices.append(i)
    
    if not clean_texts:
        return [], []
    
    # Retry avec backoff exponentiel
    for attempt in range(max_retries):
        try:
            response = await openai_client.embeddings.create(
                model=model,
                input=clean_texts
            )
            embeddings = [item.embedding for item in response.data]
            return valid_indices, embeddings
            
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                # Extraire le temps d'attente du message d'erreur
                import re
                wait_match = re.search(r'try again in (\d+\.?\d*)s', error_msg)
                if wait_match:
                    wait_time = float(wait_match.group(1)) + 2  # Ajouter 2s de marge
                else:
                    wait_time = (2 ** attempt) * 10  # Backoff exponentiel
                
                print(f"⏳ Rate limit atteint, attente de {wait_time:.1f}s (tentative {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ Erreur API: {e}")
                if attempt == max_retries - 1:
                    return [], []
                await asyncio.sleep(2 ** attempt)  # Backoff pour autres erreurs
    
    return [], []

async def process_all_data(input_file, output_file):
    """Traite tout le fichier JSONL"""
    print(f"🔄 Lecture du fichier {input_file}")
    
    # Lire toutes les lignes
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(json.loads(line.strip()))
    
    total_lines = len(lines)
    print(f"📊 {total_lines} lignes à traiter")
    
    # Créer le fichier Parquet
    if os.path.exists(output_file):
        os.remove(output_file)
    
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("embedding", pa.list_(pa.float32()))
    ])
    writer = pq.ParquetWriter(output_file, schema)
    
    # Traitement par batch
    total_batches = (total_lines + BATCH_SIZE - 1) // BATCH_SIZE
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    async def process_batch(batch_idx):
        async with semaphore:
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, total_lines)
            batch_lines = lines[start_idx:end_idx]
            
            # Extraire les IDs et textes
            batch_ids = [line['id'] for line in batch_lines]
            batch_texts = [line['text'] for line in batch_lines]
            
            # Embedder
            valid_indices, embeddings = await embed_batch(batch_texts)
            
            if valid_indices and embeddings:
                # Garder seulement les IDs valides
                valid_ids = [batch_ids[i] for i in valid_indices]
                
                # Écrire dans Parquet
                table = pa.Table.from_pydict({
                    "id": valid_ids,
                    "embedding": embeddings
                }, schema=schema)
                writer.write_table(table)
                
                print(f"✅ Batch {batch_idx + 1}/{total_batches}: {len(embeddings)} embeddings")
            else:
                print(f"⚠️ Batch {batch_idx + 1}/{total_batches}: aucun embedding valide")
    
    # Lancer tous les batchs
    start_time = time.time()
    tasks = [process_batch(i) for i in range(total_batches)]
    await asyncio.gather(*tasks)
    
    writer.close()
    
    elapsed = time.time() - start_time
    print(f"🎉 Terminé en {elapsed:.1f} secondes")
    print(f"💾 Fichier sauvegardé: {output_file}")

def main():
    print("🚀 Début du script d'embedding")
    
    input_file = "./json_chunk/all_data.jsonl"
    output_file = "tindle_embeddings.parquet"
    
    try:
        asyncio.run(process_all_data(input_file, output_file))
    except RuntimeError:
        # Fallback pour les notebooks
        loop = asyncio.get_event_loop()
        loop.run_until_complete(process_all_data(input_file, output_file))

if __name__ == "__main__":
    main()