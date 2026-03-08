# Semantic Search System - AI/ML Engineer Task

A lightweight semantic search system built with FastAPI, utilizing fuzzy clustering and a custom semantic cache implementation.

## Key Components
- **Dataset:** 20 Newsgroups (Fetched via Sklearn).
- **Preprocessing:** Deliberate removal of headers, footers, and quotes to focus on pure semantic content.
- **Embeddings:** `all-MiniLM-L6-v2` for high-performance, lightweight vectorization.
- **Fuzzy Clustering:** Implemented using **Gaussian Mixture Models (GMM)** with PCA dimensionality reduction to provide probabilistic (soft) cluster assignments.
- **Semantic Cache:** Built from first principles using **Cosine Similarity**. The system recognizes semantically similar queries to avoid redundant computations.

## Setup & Running
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `.\venv\Scripts\Activate.ps1` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the service: `uvicorn app:app --reload`
5. Access API Docs: `http://127.0.0.1:8000/docs`

## Design Decisions
- **Threshold (0.82):** Chosen to balance "smart" matching without risking false positives in the cache.
- **PCA (16 components):** Used to reduce noise and improve the stability of the Gaussian clusters.
- **GMM:** Preferred over K-Means because real-world topics (like "Politics" and "Guns") overlap; GMM captures this "fuzziness" perfectly.