# Semantic Search System — AI/ML Engineer Task

An end-to-end lightweight semantic search system built for the Trademarkia AI/ML Task. This system features fuzzy clustering, a custom-built semantic cache, and a containerized FastAPI service.

## Key Features
- **Dataset:** Dataset: UCI 20 Newsgroups corpus accessed via sklearn fetch_20newsgroups helper.
- **Semantic Analysis:** Leveraging `all-MiniLM-L6-v2` Sentence Transformers for 384-dimensional vector embeddings.
- **Fuzzy Clustering:** Implemented via **Gaussian Mixture Models (GMM)** to allow documents to belong to multiple categories with varying probabilities.
- **First-Principles Cache:** A custom-built semantic cache layer that avoids redundant semantic retrieval and embedding computations by identifying semantically similar queries via Cosine Similarity.
- **Production Ready:** Includes Docker and Docker Compose support for containerized deployment.

## Design Decisions & Justifications
- **Noisy Data Handling:** Deliberately stripped `headers`, `footers`, and `quotes` from the raw newsgroups. This ensures the model learns the actual *content* of the message rather than overfitting on email signatures or metadata.
- **PCA (20 components):** High-dimensional embeddings (384) can lead to the "curse of dimensionality" in clustering. PCA was used to reduce dimensional noise before clustering while preserving major semantic structure while ensuring the GMM clusters are semantically dense.
- **Tunable Threshold (0.82):** The semantic cache uses a similarity floor of 0.82. This value was chosen through iterative testing to ensure the system is "smart" enough to catch rephrased questions (e.g., "Space launch" vs "Rocket mission") while maintaining high precision.
- **Custom Cache:** Avoided Redis/Memcached to demonstrate a deep understanding of state management and similarity mathematics from first principles.

## Setup & Running
### Option 1: Local Run
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `.\venv\Scripts\Activate.ps1` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the service: `uvicorn app:app --reload`
5. Access API Docs: `http://127.0.0.1:8000/docs`

### Option 2: Using Docker 
docker-compose up --build