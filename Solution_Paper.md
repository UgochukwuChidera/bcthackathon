# BCT Hackathon: AI-Driven Review Generation & Cross-Domain Recommendation

**Author:** OpenCode Agent  
**Date:** May 24, 2026

---

## 1. Executive Summary
This paper details the development of two AI-driven solutions: **Task A (User Modeling Agent)** and **Task B (Cross-Domain Recommendation Agent)**. Both systems utilize a "Two-Pass" architecture powered by Large Language Models (LLMs) and Vector Embeddings to bridge the gap between free-text user intent and structured machine output. 

## 2. Methodology & Architecture

### 2.1 Task A: User Modeling Agent (Review Generator)
The goal was to generate "human-sounding" reviews that reflect a specific user's persona, habits, and cultural context.

*   **Pass 1: Normaliser:** A raw persona description (e.g., "Student in Lagos") is processed by `GPT-4o-mini` to extract a canonical JSON profile including country, average rating, writing style, and sample reviews.
*   **Nigerian Slang Integration:** A specialized slang dictionary was implemented. If the persona is identified as Nigerian, the generator injects subtle Pidgin/slang (e.g., "abeg", "wahala") to enhance authenticity.
*   **Pass 2: Generation:** The LLM acts as the extracted persona to review a specific product, ensuring the rating matches the persona's historical bias (e.g., a "harsh" reviewer rarely gives 5 stars).

### 2.2 Task B: Cross-Domain Recommendation Agent
A hybrid retrieval system was designed to handle a unified index of Amazon products, Yelp restaurants, and Goodreads books.

*   **Query Normalisation:** Unlike Task A, Task B processes a "Query" containing both persona data and intent (e.g., "I am a dev, I need a laptop under $1500").
*   **Vector Retrieval:** We use `sentence-transformers` (`all-MiniLM-L6-v2`) to compute cosine similarities between the query embedding and a 100,000-item index.
*   **Hybrid Scoring:** A custom scoring function modifies the cosine similarity by:
    *   **Price Penalty:** Heavy penalization for items exceeding the user's budget.
    *   **Rating Bonus:** Slight boost for highly-rated items.
    *   **Category Boost:** Enhancing items that match explicit keywords in the intent.
*   **LLM Re-Ranking:** The top 20 candidates are sent to an LLM to perform final semantic re-ordering.
*   **Diversity (MMR):** Maximal Marginal Relevance is applied to ensure the final Top 10 are not redundant (e.g., not just 10 different listings of the same laptop model).

## 3. Implementation Details
*   **Frontend:** Built with Gradio, featuring a tabbed interface for both agents.
*   **Storage:** Large files (embeddings `.npy` and metadata `.csv`) are managed via **Git LFS** (Large File Storage).
*   **Hosting:** Deployed on **Hugging Face Spaces** using a custom `app.py` entry point.

## 4. Research Considerations & Difficulties

### 4.1 Device & Resource Constraints
Running a 100,000-item vector search in a web-based environment presented memory challenges. This was solved by:
1.  **Pre-computing Embeddings:** Embeddings are generated once and stored as NumPy arrays.
2.  **Efficient Loading:** Using `pandas` with `low_memory=False` to handle large metadata files.

### 4.2 Learning Curve & Deployment
*   **Hugging Face LFS:** Navigating the "LFS Pointer" system was a critical learning step. We ensured GitHub only stores pointers to keep the repo lightweight, while HF pulls the full binary objects.
*   **Secrets Management:** Ensuring API keys (OpenRouter) are never committed to code but managed via HF Environment Secrets.

## 5. Research Considerations & Results
The "Two-Pass" approach significantly outperformed single-prompt generation. By forcing the model to first "think" about the persona (canonicalization), the subsequent output was more consistent and culturally nuanced. In recommendation tests, the hybrid scoring (Cosine + Rules + LLM) produced more practical results than raw vector search, which often ignores hard constraints like budget.

## 6. Future Work
*   **Model Context Protocol (MCP):** Integrating MCPs to allow the agent to fetch live data (e.g., current prices) instead of relying on static CSVs.
*   **Robust Embeddings:** Moving from `MiniLM` to `BGE-M3` or OpenAI's `text-embedding-3-large` for better semantic understanding.
*   **Device-Local Inference:** Implementing `WebGPU` or `ONNX` to run embeddings entirely client-side, reducing server load.

---
**BCT Hackathon 2026**
