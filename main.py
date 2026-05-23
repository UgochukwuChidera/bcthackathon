import gradio as gr
import numpy as np
import pandas as pd
import re
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================
# 0. LLM client setup (OpenRouter)
# ============================================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = "openai/gpt-4o-mini"

def call_llm(prompt, json_mode=True):
    """Single LLM call with optional JSON mode."""
    try:
        kwargs = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip()
        return json.loads(content) if json_mode else content
    except Exception as e:
        return {"error": str(e)} if json_mode else None

# ============================================
# 1. Load pre‑computed embeddings and metadata
# ============================================
print("Loading unified embeddings and metadata...")
embeddings = np.load('unified_embeddings.npy')
metadata = pd.read_csv('unified_metadata.csv')

# Ensure required columns exist
required_cols = ['type', 'name', 'category', 'rating_num', 'price_bucket', 'price_numeric']
for col in required_cols:
    if col not in metadata.columns:
        if col == 'price_numeric':
            metadata[col] = np.nan
        else:
            metadata[col] = 'unknown'

print(f"Loaded {len(metadata)} items, embeddings shape: {embeddings.shape}")

# ============================================
# 2. Load the embedding model
# ============================================
model = SentenceTransformer('all-MiniLM-L6-v2')

# ============================================
# 3. Domain detection via LLM
# ============================================
def detect_domain(query):
    prompt = f"""
Classify the following user request into exactly one of these categories: 'product', 'restaurant', 'book'.
If the request does not fit any of these three, return 'other'.

Return ONLY valid JSON: {{"domain": "product"}} (or "restaurant", "book", "other").

User request: {query}
"""
    result = call_llm(prompt, json_mode=True)
    if "error" in result:
        return "other"
    return result.get("domain", "other")

# ============================================
# 4. Helper functions for re‑scoring and fallback generation
# ============================================
def extract_budget(query):
    match = re.search(r'under\s*\$?(\d+(?:\.\d+)?)', query, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r'\$(\d+(?:\.\d+)?)', query)
    if match:
        return float(match.group(1))
    return None

CATEGORY_MAP = {
    "laptop": ["Computers & Tablets", "Laptops", "Computer"],
    "computer": ["Computers & Tablets", "Desktop", "Computer"],
    "headphones": ["Headphones & Earbuds", "Headphones"],
    "earbuds": ["Headphones & Earbuds"],
    "phone": ["Cell Phones & Accessories", "Smartphones"],
    "shaving": ["Shaving & Hair Removal Products"],
    "clipper": ["Shaving & Hair Removal Products", "Hair Care Products"],
    "trimmer": ["Shaving & Hair Removal Products", "Hair Care Products"],
    "restaurant": ["Restaurants", "Italian", "Chinese", "Sushi", "Pizza"],
    "book": ["books", "Fantasy", "Science Fiction", "Mystery"],
}

def get_target_categories(query):
    query_lower = query.lower()
    for keyword, categories in CATEGORY_MAP.items():
        if keyword in query_lower:
            return categories
    return None

def recommend_from_index(query, domain, top_k=10, retrieval_k=50):
    """Retrieve items from the unified index filtered by domain."""
    budget = extract_budget(query)
    target_cats = get_target_categories(query)

    # Filter metadata and embeddings to the detected domain
    domain_mask = metadata['type'] == domain
    if not any(domain_mask):
        return pd.DataFrame()  # no items of this domain
    domain_metadata = metadata[domain_mask].reset_index(drop=True)
    domain_embeddings = embeddings[domain_mask]

    # Encode query and compute similarity only within the domain
    query_vec = model.encode([query])
    sims = cosine_similarity(query_vec, domain_embeddings).flatten()
    top_indices = np.argsort(sims)[-retrieval_k:][::-1]

    candidates = []
    for idx in top_indices:
        row = domain_metadata.iloc[idx]
        cos_sim = sims[idx]

        # Star bonus
        stars = row.get('rating_num', 0)
        if pd.isna(stars):
            star_bonus = 1.0
        else:
            star_bonus = 1 + (stars - 3) / 5
            star_bonus = max(0.5, min(1.5, star_bonus))

        # Price penalty
        price = row.get('price_numeric', np.nan)
        price_penalty = 1.0
        if budget is not None and not pd.isna(price) and price > 0:
            if price > budget:
                over_ratio = price / budget
                price_penalty = max(0.1, 0.5 ** (over_ratio - 1))

        # Category penalty
        cat = row.get('category', '')
        cat_penalty = 1.0
        if target_cats is not None:
            match = any(tc.lower() in str(cat).lower() for tc in target_cats)
            if not match:
                cat_penalty = 0.1
            else:
                cat_penalty = 1.2

        final_score = cos_sim * star_bonus * price_penalty * cat_penalty
        candidates.append((idx, final_score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    final_indices = [c[0] for c in candidates[:top_k]]
    results = domain_metadata.iloc[final_indices].copy()
    results['score'] = [c[1] for c in candidates[:top_k]]

    # Format actual price
    if 'price_numeric' in results.columns:
        results['actual_price'] = results['price_numeric'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
        )
    else:
        results['actual_price'] = "N/A"

    display_cols = ['type', 'name', 'category', 'rating_num', 'actual_price', 'score']
    display_map = {
        'type': 'Type',
        'name': 'Name',
        'category': 'Category',
        'rating_num': 'Rating',
        'actual_price': 'Price',
        'score': 'Score'
    }
    available = [col for col in display_cols if col in results.columns]
    results = results[available].rename(columns=display_map)
    return results.head(top_k)

def fallback_llm_recommend(query, domain, top_k=10):
    """Generate recommendations via LLM when domain is not in the catalogue."""
    prompt = f"""
The user requested: "{query}"
They are looking for a {domain}. We do not have a catalogue of {domain}s.
Based on your knowledge, recommend {top_k} realistic {domain}s that match the user's preferences.
Return a ranked list as a bullet list, each item with a one‑line justification.
"""
    response = call_llm(prompt, json_mode=False)
    if isinstance(response, dict) and "error" in response:
        return f"Error generating recommendations: {response['error']}"
    return response

# ============================================
# 5. Main recommendation function (multi‑stage)
# ============================================
def recommend(query, top_k=10):
    # Stage 1: Detect domain
    domain = detect_domain(query)
    if domain == "other":
        return fallback_llm_recommend(query, domain, top_k)

    # Stage 2: Retrieve from index
    results = recommend_from_index(query, domain, top_k)
    if results.empty:
        return f"No items of type '{domain}' found in our catalogue."
    return results

# ============================================
# 6. Gradio Interface
# ============================================
def format_recommendations(query):
    output = recommend(query)
    if isinstance(output, pd.DataFrame):
        if output.empty:
            return "No recommendations found."
        return output.to_html(index=False, escape=False, float_format="%.3f")
    else:
        # Fallback text from LLM or error message
        return f"<pre>{output}</pre>"

with gr.Blocks(title="Task B: Recommendation Agent") as demo:
    gr.Markdown("# Task B: Cross‑Domain Recommendation Agent")
    gr.Markdown("Enter a persona description. The system will detect the domain (product, restaurant, book) and return relevant recommendations.")
    
    with gr.Row():
        query_input = gr.Textbox(label="Persona Description", lines=3,
                                 placeholder="e.g., I need a lightweight laptop for students under $500")
        submit_btn = gr.Button("Get Recommendations")
    
    output_html = gr.HTML(label="Recommendations")
    
    submit_btn.click(fn=format_recommendations, inputs=query_input, outputs=output_html)

if __name__ == "__main__":
    demo.launch(share=True, strict_cors=False)