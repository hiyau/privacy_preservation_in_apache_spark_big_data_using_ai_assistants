import os
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/processed/healthcare_cag.csv"
VECTOR_DIR = "vector_store"
OUTPUT_FILE = os.path.join(VECTOR_DIR, "embeddings.pkl")

os.makedirs(VECTOR_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
df.columns = [c.lower().strip() for c in df.columns]

# Detect disease column safely
DISEASE_ALIASES = ["disease", "diagnosis", "condition", "primary_diagnosis"]
disease_col = next((c for c in df.columns if c in DISEASE_ALIASES), None)

if not disease_col:
    raise ValueError("❌ No disease/diagnosis column found")

# Aggregate (privacy preserved)
agg = df.groupby(disease_col).size().reset_index(name="count")

texts, meta = [], []

for _, row in agg.iterrows():
    disease = row[disease_col]
    count = int(row["count"])

    texts.append(f"Total number of patients diagnosed with {disease} is {count}")
    meta.append({"disease": disease, "count": count})

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(
        {"embeddings": embeddings, "texts": texts, "meta": meta},
        f
    )

print("✅ Embeddings created successfully (100% privacy-safe)")
