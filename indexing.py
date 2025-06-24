import pyarrow.parquet as pq
import numpy as np
import pickle
import faiss
import os

# 1. Paramètres
PARQUET_FILE = "tindle_embeddings.parquet"
FAISS_INDEX_FILE = "tindle_index.faiss"
EMBED_DIM = None          # sera détecté automatiquement plus bas
FAISS_INDEX_FACTORY = "IVF4096,Flat"   # ou "Flat" pour un index exact

# 2. Charger les données depuis Parquet
print("➡️ Lecture du Parquet...")
table = pq.read_table(PARQUET_FILE, columns=["id", "embedding"])
ids = table.column("id").to_pylist()
embeddings = table.column("embedding")

# If embeddings are stored as Arrow FixedSizeList or List of float32, this will work:
xb = np.vstack([np.array(e.as_py(), dtype="float32") for e in embeddings])
n, d = xb.shape
EMBED_DIM = d
print(f"✔️ {n} vecteurs de dimension {d}")

print(embeddings[0])
print(type(embeddings[0]))

for i in range(5):
    print(embeddings[i], type(embeddings[i]))

# 3. Créer l'index FAISS
#    - IVF (Inverted File) pour gros volume ou Flat pour exact
print("➡️ Création de l'index FAISS...")
index = faiss.index_factory(d, FAISS_INDEX_FACTORY, faiss.METRIC_L2)

# Si on utilise un IVF on doit entraîner l'index sur les données
if not index.is_trained:
    print("📚 Entraînement de l'index…")
    index.train(xb)

# 4. Alimenter l'index
print("➡️ Ajout des vecteurs à l'index…")
index.add(xb)  # on pourrait aussi faire add_with_ids pour custom ids

# 5. Sauvegarder l'index et la liste d'IDs
print(f"➡️ Sauvegarde de l'index dans `{FAISS_INDEX_FILE}`…")
faiss.write_index(index, FAISS_INDEX_FILE)

# Pour réutiliser plus tard, tu charge dans un dict ou un fichier à part
with open("tindle_ids.pkl", "wb") as f:
    pickle.dump(ids, f)

print("✅ Index FAISS prêt et sauvegardé !")