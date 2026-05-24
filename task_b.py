import gradio as gr
import numpy as np
import pandas as pd
import re
import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4o-mini"
CATALOGUE_PATH = os.environ.get('CATALOGUE_PATH', '/data/catalogue.csv')
PRE_COMPUTED_EMBEDDINGS = 'unified_embeddings.npy'
PRE_COMPUTED_METADATA = 'unified_metadata.csv'

# ------------------------------------------------------------
# 1. Column mapping & Data Loading
# ------------------------------------------------------------
def match_column(df, candidates, default=None):
    df_cols_lower = {col.lower(): col for col in df.columns}
    for cand in candidates:
        cand_lower = cand.lower()
        if cand_lower in df_cols_lower:
            return df_cols_lower[cand_lower]
    return default

def load_catalogue(path):
    df = pd.read_csv(path)
    name_col = match_column(df, ['name', 'title', 'item_name', 'product_name'])
    category_col = match_column(df, ['category', 'categories', 'cat', 'type'])
    rating_col = match_column(df, ['rating', 'stars', 'average_rating', 'avg_rating', 'avg rating', 'average rating'])
    price_col = match_column(df, ['price', 'amount', 'listprice', 'price_numeric'])
    type_col = match_column(df, ['type', 'domain', 'item_type'])

    out = pd.DataFrame()
    out['name'] = df[name_col].fillna('unknown') if name_col else 'unknown'
    out['category'] = df[category_col].fillna('general') if category_col else 'general'
    out['rating_num'] = pd.to_numeric(df[rating_col], errors='coerce').fillna(0) if rating_col else 0
    out['price_numeric'] = pd.to_numeric(df[price_col], errors='coerce') if price_col else np.nan
    out['type'] = df[type_col].fillna('product') if type_col else 'product'
    return out

def create_rich_text(row):
    parts = [f"TYPE: {row['type']}", f"Name: {row['name']}"]
    if row['category']:
        parts.append(f"Category: {row['category']}")
    rating = row['rating_num']
    if pd.notna(rating) and rating > 0:
        bucket = 'high' if rating >= 4.0 else 'medium' if rating >= 2.5 else 'low'
        parts.append(f"Rating: {rating:.1f} ({bucket})")
    else:
        parts.append("Rating: unknown")
    price = row['price_numeric']
    if pd.notna(price) and price > 0:
        parts.append(f"Price: ${price:.2f}")
    else:
        parts.append("Price: unknown")
    return " . ".join(parts)

# ------------------------------------------------------------
# 2. LLM client and Query normaliser
# ------------------------------------------------------------
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

def call_llm(prompt, json_mode=True):
    try:
        kwargs = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip()
        return json.loads(content) if json_mode else content
    except Exception as e:
        print(f"LLM error: {e}")
        return {"error": str(e)} if json_mode else None

def normalise_query(free_text):
    prompt = f"""
Convert this user query (which may contain a persona and specific product/intent data) into a structured JSON profile for recommendation.
Fields:
- name, age, gender, occupation, location, country (extract if present)
- intent (what is the user looking for specifically?)
- description (1-2 sentences capturing the persona and request context)
- preferences (JSON object: {{key_desires: "...", constraints: "...", writing_style: "..."}})
- history (string: past likes/dislikes/returns mentioned)
- budget (extracted numeric value if present, else null)
- domain (one of: product, restaurant, book, other)

Query: {free_text}
Return ONLY valid JSON.
"""
    return call_llm(prompt, json_mode=True)

# ------------------------------------------------------------
# 3. Helper functions for pricing, ranking, etc.
# ------------------------------------------------------------
def get_target_categories(query_text, intent):
    text = (str(query_text) + " " + str(intent)).lower()
    CATEGORY_MAP = {
        "laptop": ["Computers & Tablets", "Laptops", "Computer"],
        "computer": ["Computers & Tablets", "Desktop", "Computer"],
        "television": ["TV", "Television", "Electronics", "TVs"],
        "tv": ["TV", "Television", "Electronics", "TVs"],
        "monitor": ["Monitors", "Computer Monitors", "Displays"],
        "headphones": ["Headphones & Earbuds", "Headphones"],
        "earbuds": ["Headphones & Earbuds"],
        "phone": ["Cell Phones & Accessories", "Smartphones"],
        "shaving": ["Shaving & Hair Removal Products"],
        "clipper": ["Shaving & Hair Removal Products", "Hair Care Products"],
        "trimmer": ["Shaving & Hair Removal Products", "Hair Care Products"],
        "restaurant": ["Restaurants", "Italian", "Chinese", "Sushi", "Pizza", "Cafe"],
        "book": ["books", "Fantasy", "Science Fiction", "Mystery", "Novel"],
    }
    for kw, cats in CATEGORY_MAP.items():
        if kw in text:
            return cats
    return None

def initial_rank(metadata, embeddings, model, query_text, normalized_data, price_percentiles, retrieval_k=50):
    domain = normalized_data.get('domain', 'product')
    intent = normalized_data.get('intent', '')
    budget = normalized_data.get('budget')
    
    target_cats = get_target_categories(query_text, intent)
    
    # Filter by domain if possible
    mask = metadata['type'] == domain
    if not mask.any():
        mask = pd.Series([True] * len(metadata))
        domain_used = "unified"
    else:
        domain_used = domain

    dm = metadata[mask].reset_index(drop=True)
    demb = embeddings[mask]
    
    qvec = model.encode([query_text + " " + intent])
    sims = cosine_similarity(qvec, demb).flatten()
    top_idx = np.argsort(sims)[-retrieval_k:][::-1]
    
    candidates = []
    for idx in top_idx:
        row = dm.iloc[idx]
        cos_sim = sims[idx]
        
        # Star bonus
        stars = row['rating_num']
        star_bonus = 1 + (stars - 3) / 5 if not pd.isna(stars) else 1.0
        star_bonus = max(0.5, min(1.5, star_bonus))
        
        # Price penalty/bonus
        price = row['price_numeric']
        price_penalty = 1.0
        if budget is not None and not np.isnan(price) and price > 0:
            if price > budget:
                price_penalty = 0.6 ** (price / budget - 1)
                price_penalty = max(0.2, price_penalty)
        
        # Category match
        cat = row['category']
        cat_bonus = 1.2 if target_cats and any(tc.lower() in cat.lower() for tc in target_cats) else 1.0
        
        final_score = cos_sim * star_bonus * price_penalty * cat_bonus
        
        # Store detailed scores for the table
        res_row = row.to_dict()
        res_row['_cos_sim'] = cos_sim
        res_row['_star_bonus'] = star_bonus
        res_row['_price_penalty'] = price_penalty
        res_row['_cat_bonus'] = cat_bonus
        
        candidates.append((idx, final_score, res_row))
        
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:retrieval_k], domain_used

def llm_rerank(query_text, normalized_data, candidates, top_k=20):
    if not candidates:
        return []
    items = []
    for i, (_, _, row) in enumerate(candidates[:top_k]):
        p = row.get('price_numeric', 'N/A')
        price_str = f"${p:.2f}" if not np.isnan(p) else "N/A"
        items.append(f"{i+1}. {row.get('name', 'Unknown')} (Category: {row.get('category', 'N/A')}, Price: {price_str})")
    
    prompt = f"""
Query: "{query_text}"
Intent: {normalized_data.get('intent')}
Preferences: {json.dumps(normalized_data.get('preferences'))}
Candidates:
{chr(10).join(items)}

Reorder these 20 items from most to least relevant based on the user's specific request.
Return ONLY a JSON array of the original indices (1-based).
"""
    result = call_llm(prompt, json_mode=True)
    if isinstance(result, list):
        try:
            ordered = [candidates[int(i)-1] for i in result if 1 <= int(i) <= len(candidates)]
            return ordered
        except: pass
    return candidates[:top_k]

def mmr_diversity(candidates, lambda_param=0.6, top_k=10):
    if len(candidates) <= top_k:
        return candidates
    selected = [candidates[0]]
    remaining = candidates[1:]
    while len(selected) < top_k and remaining:
        max_mmr = -1e9
        best_idx = -1
        for i, (idx, score, row) in enumerate(remaining):
            max_sim = 0
            for (_, _, sel_row) in selected:
                cat_sim = 1.0 if row.get('category') == sel_row.get('category') else 0.0
                name_overlap = len(set(row.get('name','').split()) & set(sel_row.get('name','').split())) / max(1, len(row.get('name','').split()))
                sim = 0.7 * cat_sim + 0.3 * name_overlap
                max_sim = max(max_sim, sim)
            mmr = lambda_param * score - (1 - lambda_param) * max_sim
            if mmr > max_mmr:
                max_mmr, best_idx = mmr, i
        selected.append(remaining.pop(best_idx))
    return selected

def recommend_flow(metadata, embeddings, model, query_text, enable_diversity=True):
    # 1. Normalisation
    norm = normalise_query(query_text)
    if "error" in norm:
        return f"Error: {norm['error']}", {}, []
    
    # 2. Retrieval & Scoring
    price_vals = metadata['price_numeric'].dropna().values
    pcts = np.percentile(price_vals, [50, 80, 90]) if len(price_vals) > 0 else [0,0,0]
    
    candidates, domain_used = initial_rank(metadata, embeddings, model, query_text, norm, pcts)
    norm['retrieval_domain'] = domain_used
    
    if not candidates:
        return "No matches found.", norm, []
    
    # 3. LLM Rerank
    top20 = llm_rerank(query_text, norm, candidates)
    
    # 4. Diversity
    final = mmr_diversity(top20, top_k=10) if enable_diversity else top20[:10]
    
    return final, norm, top20

# ------------------------------------------------------------
# 4. UI Rendering & Table logic
# ------------------------------------------------------------
_ACCENT      = "#4a6fa5"
_ACCENT_LITE = "#dce6f5"
_ROW_ALT     = "#f7f5f2"
_ROW_BASE    = "#ffffff"
_TEXT_MAIN   = "#1a1a1a"
_TEXT_MID    = "#444444"
_BORDER      = "#e0dbd4"

def build_results_table(items, is_top_20=False):
    if not items: return ""
    
    max_score = max(it[1] for it in items) if items else 1.0
    
    # Column Headers
    cols = ["#", "Name", "Category", "Rating", "Price", "Original Sim", "Computation (Final Score)"]
    header = "".join([f"<th style='padding:12px; border-bottom:2px solid {_BORDER}; text-align:left; font-size:11px; text-transform:uppercase;'>{c}</th>" for c in cols])
    
    rows_html = ""
    for i, (idx, score, r) in enumerate(items):
        bg = _ACCENT_LITE if i == 0 and not is_top_20 else (_ROW_ALT if i % 2 == 0 else _ROW_BASE)
        
        # Computation string
        comp = f"Sim: {r['_cos_sim']:.3f}"
        if r['_star_bonus'] != 1.0: comp += f" × <span style='color:green;'>Star: {r['_star_bonus']:.2f}</span>"
        if r['_price_penalty'] < 1.0: comp += f" × <span style='color:red;'><b>Penalty: {r['_price_penalty']:.2f}</b></span>"
        if r['_cat_bonus'] > 1.0: comp += f" × <span style='color:blue;'>Cat: {r['_cat_bonus']:.1f}</span>"
        comp += f" = <b>{score:.3f}</b>"
        
        rows_html += f"""
        <tr style="background:{bg}; font-size:13px; color:{_TEXT_MAIN};">
            <td style="padding:10px; border-bottom:1px solid {_BORDER};">{i+1}</td>
            <td style="padding:10px; border-bottom:1px solid {_BORDER}; font-weight:500;">{r['name']}</td>
            <td style="padding:10px; border-bottom:1px solid {_BORDER}; color:{_TEXT_MID};">{r['category']}</td>
            <td style="padding:10px; border-bottom:1px solid {_BORDER};">★ {r['rating_num']:.1f}</td>
            <td style="padding:10px; border-bottom:1px solid {_BORDER};">${r['price_numeric']:.2f}</td>
            <td style="padding:10px; border-bottom:1px solid {_BORDER};">{r['_cos_sim']:.4f}</td>
            <td style="padding:10px; border-bottom:1px solid {_BORDER};">{comp}</td>
        </tr>
        """
        
    return f"""
    <table style="width:100%; border-collapse:collapse; margin-top:10px;">
        <thead><tr>{header}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """

def format_output(metadata, embeddings, model, query, enable_diversity):
    final_items, norm, top20_items = recommend_flow(metadata, embeddings, model, query, enable_diversity)
    
    if isinstance(final_items, str):
        return f"<div style='color:red; padding:20px;'>{final_items}</div>"
        
    # Build sections
    persona_json = json.dumps(norm, indent=2)
    
    final_table = build_results_table(final_items)
    top20_table = build_results_table(top20_items, is_top_20=True)
    
    tooltip_html = """
    <div style="background:#fff9e6; border:1px solid #ffe58f; padding:10px; border-radius:6px; font-size:12px; color:#856404; margin-bottom:10px;">
        💡 <b>How Top 10 are selected:</b> We retrieve 50 candidates using Cosine Similarity, score them based on budget and ratings, 
        use an LLM to re-rank the top 20, and finally apply MMR (Maximal Marginal Relevance) to pick the 10 most diverse and relevant matches.
    </div>
    """
    
    html = f"""
    <div style="font-family:sans-serif; padding:10px;">
        <details open style="border:1px solid {_BORDER}; border-radius:8px; padding:15px; background:#fff;">
            <summary style="font-weight:bold; cursor:pointer; font-size:16px;">🔍 Full Recommendation Analysis (Click to collapse)</summary>
            
            <div style="margin-top:20px;">
                <h4 style="margin:0 0 10px 0; color:{_ACCENT};">1. Normalised Query & Persona</h4>
                <pre style="background:#13131f; color:#cdd6f4; padding:15px; border-radius:8px; font-size:13px; font-family:'Fira Code', monospace; overflow:auto; border:1px solid #2e2e3e;">{persona_json}</pre>
            </div>

            <div style="margin-top:20px;">
                <h4 style="margin:0 0 10px 0; color:{_ACCENT};">2. Top 10 Selected Recommendations</h4>
                {tooltip_html}
                <div style="overflow-x:auto;">{final_table}</div>
            </div>

            <div style="margin-top:30px;">
                <details style="border-top:1px solid {_BORDER}; padding-top:15px;">
                    <summary style="cursor:pointer; color:{_TEXT_MID}; font-weight:500;">📋 View Top 20 Candidates (Pre-Diversity Filtering)</summary>
                    <div style="overflow-x:auto; margin-top:10px;">{top20_table}</div>
                </details>
            </div>
        </details>
    </div>
    """
    return html

# ------------------------------------------------------------
# 5. App Setup & Launch
# ------------------------------------------------------------
print("Initializing Recommendation Agent...")
model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists(PRE_COMPUTED_EMBEDDINGS) and os.path.exists(PRE_COMPUTED_METADATA):
    embeddings = np.load(PRE_COMPUTED_EMBEDDINGS)
    metadata = pd.read_csv(PRE_COMPUTED_METADATA, low_memory=False)
else:
    # Fallback to loading CSV and generating (first time)
    metadata = load_catalogue('catalogue.csv') # Assuming local file
    metadata['rich_text'] = metadata.apply(create_rich_text, axis=1)
    embeddings = model.encode(metadata['rich_text'].tolist(), show_progress_bar=True)
    np.save(PRE_COMPUTED_EMBEDDINGS, embeddings)
    metadata.to_csv(PRE_COMPUTED_METADATA, index=False)

CSS = """
.page-wrap { max-width: 1000px; margin: 0 auto; padding: 20px; }
.input-card { background: #f9f7f4; border: 1px solid #e0dbd4; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
"""

with gr.Blocks(css=CSS, title="Recommendation Agent v2") as demo:
    with gr.Column(elem_classes="page-wrap"):
        gr.HTML("<div style='text-align:center;'><h2>Task B: Advanced Recommendation Agent</h2><p>Persona extraction + Hybrid Scoring + LLM Reranking</p></div>")
        
        with gr.Column(elem_classes="input-card"):
            inp = gr.Textbox(label="Query / Persona", lines=4, placeholder="Describe who you are and what you need...")
            with gr.Row():
                div_cb = gr.Checkbox(label="Enable Diversity (MMR)", value=True)
                btn = gr.Button("Get Recommendations", variant="primary")
        
        out = gr.HTML()

    btn.click(fn=lambda q, d: format_output(metadata, embeddings, model, q, d), inputs=[inp, div_cb], outputs=out)

if __name__ == "__main__":
    demo.launch()
