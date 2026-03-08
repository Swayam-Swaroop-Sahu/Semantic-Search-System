import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import fetch_20newsgroups
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

app = FastAPI(title="AI/ML Semantic Search")

# Choice: all-MiniLM-L6-v2 is industry standard for lightweight semantic tasks
model = SentenceTransformer('all-MiniLM-L6-v2')

class SystemState:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.gmm = None
        self.pca = None
        self.cache = [] # Internal data structure for first-principles cache
        self.stats = {"total_entries": 0, "hit_count": 0, "miss_count": 0}
        self.threshold = 0.82 # Tunable Decision: Balances "smart" matching vs accuracy

state = SystemState()

def clean_text(text):
    """Deliberate choice: Remove short posts and excess whitespace to reduce noise."""
    text = text.replace('\n', ' ').strip()
    return text if len(text) > 100 else None

@app.on_event("startup")
async def startup_event():
    print("Fetching 20 Newsgroups dataset (Source: UCI Archive via Sklearn)...")
    
    # We use 'remove' to satisfy the 'Deliberate choice' requirement of the task.
    # This strips headers/footers which are just 'noise' (emails, dates) 
    # and not 'semantic meaning'.
    raw_data = fetch_20newsgroups(
        subset='all', 
        remove=('headers', 'footers', 'quotes')
    )
    
    # Cleaning phase
    cleaned = [clean_text(doc) for doc in raw_data.data]
    state.documents = [doc for doc in cleaned if doc is not None][:2500] # Limit for speed
    
    print(f"Vectorizing {len(state.documents)} documents...")
    state.embeddings = model.encode(state.documents, convert_to_numpy=True)
    
    # Part 2: Fuzzy Clustering
    # PCA used to reduce dimensionality before GMM for better cluster stability
    state.pca = PCA(n_components=20, random_state=42)
    reduced_embs = state.pca.fit_transform(state.embeddings)
    
    # GMM allows for 'Soft Assignments' (probabilities) instead of Hard Labels
    state.gmm = GaussianMixture(n_components=10, random_state=42)
    state.gmm.fit(reduced_embs)
    print("System Ready.")

# --- Part 3: First Principles Semantic Cache ---
def check_cache(query_text, query_emb):
    for entry in state.cache:
        # Manual Cosine Similarity implementation
        norm_q = np.linalg.norm(query_emb)
        norm_e = np.linalg.norm(entry["emb"])
        similarity = np.dot(query_emb, entry["emb"]) / (norm_q * norm_e)
        
        if similarity >= state.threshold:
            return entry, float(similarity)
    return None, 0.0

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def perform_query(request: QueryRequest):
    query_text = request.query
    query_emb = model.encode(query_text)
    
    # 1. Check Semantic Cache
    match, sim_score = check_cache(query_text, query_emb)
    
    if match:
        state.stats["hit_count"] += 1
        return {
            "query": query_text,
            "cache_hit": True,
            "matched_query": match["query"],
            "similarity_score": round(sim_score, 4),
            "result": match["result"],
            "dominant_cluster": match["cluster"]
        }
    
    # 2. Cache Miss: Perform Search
    state.stats["miss_count"] += 1
    
    # Determine dominant cluster
    query_pca = state.pca.transform(query_emb.reshape(1, -1))
    cluster_idx = int(np.argmax(state.gmm.predict_proba(query_pca)[0]))
    
    # Find most similar document
    similarities = np.dot(state.embeddings, query_emb) / (np.linalg.norm(state.embeddings, axis=1) * np.linalg.norm(query_emb))
    best_idx = np.argmax(similarities)
    result = state.documents[best_idx][:400] # Return snippet
    
    # Store in Cache
    state.cache.append({
        "query": query_text,
        "emb": query_emb,
        "result": result,
        "cluster": cluster_idx
    })
    state.stats["total_entries"] = len(state.cache)
    
    return {
        "query": query_text,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": round(float(np.max(similarities)), 4),
        "result": result,
        "dominant_cluster": cluster_idx
    }

@app.get("/cache/stats")
async def get_stats():
    total = state.stats["hit_count"] + state.stats["miss_count"]
    hit_rate = state.stats["hit_count"] / total if total > 0 else 0
    return {
        **state.stats,
        "hit_rate": round(hit_rate, 4)
    }

@app.delete("/cache")
async def flush_cache():
    state.cache = []
    state.stats = {"total_entries": 0, "hit_count": 0, "miss_count": 0}
    return {"status": "Cache Cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)