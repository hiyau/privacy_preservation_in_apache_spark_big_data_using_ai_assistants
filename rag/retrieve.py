import re
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rag.llm_intent import classify_intent


# -----------------------------
# RULE-BASED SAFETY CHECK
# -----------------------------
def is_safe_aggregate_query(query: str) -> bool:
    query = query.lower()

    aggregate_patterns = [
        r"how many",
        r"count",
        r"total",
        r"number of",
        r"statistics",
        r"sum",
        r"cases"
    ]

    identifier_patterns = [
        r"name",
        r"patient id",
        r"insurance",
        r"address",
        r"phone",
        r"email",
        r"list",
        r"show patients",
        r"who are",
        r"details"
    ]

    if any(re.search(p, query) for p in identifier_patterns):
        return False

    return any(re.search(p, query) for p in aggregate_patterns)


# -----------------------------
# MAIN RETRIEVAL FUNCTION
# -----------------------------
def retrieve(query: str):
    # 1️⃣ Rule-based override (strongest)
    if is_safe_aggregate_query(query):
        intent = "AGGREGATE"
    else:
        intent = classify_intent(query)

    # 2️⃣ Privacy enforcement
    if intent != "AGGREGATE":
        return [
            "❌ Access denied: This request violates healthcare privacy regulations. "
            "Only aggregated, non-identifiable information is permitted."
        ]

    # 3️⃣ Load vector store
    with open("vector_store/embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    embeddings = np.array(data["embeddings"])
    texts = data["texts"]
    meta = data["meta"]

    # 4️⃣ Embed user query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])

    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_idx = scores.argmax()

    disease = meta[top_idx]["disease"]
    count = meta[top_idx]["count"]

    # 5️⃣ SAFE RESPONSE (NO PII)
    return [
        f"🩺 Disease: {disease}\n"
        f"📊 Total Patients: {count}\n"
        f"🔒 Personal patient details are protected under healthcare privacy regulations."
    ]
