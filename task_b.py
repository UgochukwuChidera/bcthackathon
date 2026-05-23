import gradio as gr
import numpy as np
import pandas as pd
import re
import json
import os
import difflib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------
# 0. Configuration
# ------------------------------------------------------------
# Path to the catalogue CSV (set by environment variable or default)
CATALOGUE_PATH = os.environ.get('CATALOGUE_PATH', '/data/catalogue.csv')
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4o-mini"

# ------------------------------------------------------------
# 1. Column name mapping (exact + fuzzy)
# ------------------------------------------------------------
def match_column(df, candidates, default=None, fuzzy_threshold=0.8):
    """
    Find a column in df that matches any of the candidates (case‑insensitive).
    First try exact matches, then fuzzy matching.
    Returns the column name or None.
    """
    df_cols_lower = {col.lower(): col for col in df.columns}
    # Exact match (case‑insensitive)
    for cand in candidates:
        cand_lower = cand.lower()
        if cand_lower in df_cols_lower:
            return df_cols_lower[cand_lower]
    # Fuzzy match
    for cand in candidates:
        matches = difflib.get_close_matches(cand.lower(), df_cols_lower.keys(), n=1, cutoff=fuzzy_threshold)
        if matches:
            return df_cols_lower[matches[0]]
    return default

def load_catalogue(path):
    """Load CSV and standardise column names."""
    df = pd.read_csv(path)
    # Map to internal names
    name_col = match_column(df, ['name', 'title', 'item_name', 'product_name'])
    category_col = match_column(df, ['category', 'categories', 'cat', 'type'])
    rating_col = match_column(df, ['rating', 'stars', 'average_rating', 'avg_rating', 'avg rating', 'average rating'])
    price_col = match_column(df, ['price', 'amount', 'listprice', 'price_numeric'])
    type_col = match_column(df, ['type', 'domain', 'item_type'])
    
    # Create standardised DataFrame
    out = pd.DataFrame()
    out['name'] = df[name_col].fillna('unknown') if name_col else 'unknown'
    out['category'] = df[category_col].fillna('general') if category_col else 'general'
    out['rating_num'] = pd.to_numeric(df[rating_col], errors='coerce').fillna(0) if rating_col else 0
    out['price_numeric'] = pd.to_numeric(df[price_col], errors='coerce') if price_col else np.nan
    out['type'] = df[type_col].fillna('product') if type_col else 'product'
    return out

# ------------------------------------------------------------
# 2. LLM client
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

# ------------------------------------------------------------
# 3. Persona normaliser
# ------------------------------------------------------------
def normalise_persona(free_text):
    prompt = f"""
Convert this persona description into structured JSON.
Fields:
- name, age, gender, occupation, location, country
- description (1-2 sentences)
- preferences (string: key desires, constraints, writing style)
- history (string: past purchases, likes/dislikes, returns)
- avg_rating, rating_std, total_reviews
- example_reviews (2 short examples)

Persona: {free_text}
Return ONLY valid JSON.
"""
    return call_llm(prompt, json_mode=True)

# ------------------------------------------------------------
# 4. Intent extraction
# ------------------------------------------------------------
def extract_target_type(query):
    prompt = f"""
Given the user's request, classify the desired item type into exactly one of: 'product', 'restaurant', 'book', or 'other'.
- 'product' is for physical goods like laptops, headphones, shoes, phones, appliances, TVs, monitors, kitchenware, tools.
- 'restaurant' is for places to eat, cafes, bars.
- 'book' is for books, novels, literature.
- 'other' is for anything else, including vehicles, real estate, aircraft, boats, services, digital goods, events.

User request: {query}
Return ONLY valid JSON: {{"type": "..."}}
"""
    result = call_llm(prompt, json_mode=True)
    if "error" in result:
        return "other"
    t = result.get("type", "other").lower()
    if t not in ['product', 'restaurant', 'book']:
        return "other"
    return t

def extract_core_noun(query):
    prompt = f"""
From the user request, extract the single most important noun that represents the item they are looking for.
Return ONLY that word or short phrase (max 3 words), nothing else. 
Examples: 'laptop', 'car', 'headphones', 'restaurant', 'book', 'house', 'television'.

User request: {query}
"""
    result = call_llm(prompt, json_mode=False)
    if isinstance(result, dict) and "error" in result:
        words = query.split()
        return words[-1] if words else "item"
    return str(result).strip().lower()

# ------------------------------------------------------------
# 5. Category mapping & metadata check
# ------------------------------------------------------------
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

def item_type_exists_in_metadata(metadata, target_type, core_noun):
    if target_type not in ['product', 'restaurant', 'book']:
        return False
    mask = metadata['type'] == target_type
    if not mask.any():
        return False
    subset = metadata[mask]
    name_match = subset['name'].str.lower().str.contains(core_noun, na=False)
    cat_match = subset['category'].str.lower().str.contains(core_noun, na=False)
    if (name_match | cat_match).any():
        return True
    if core_noun in CATEGORY_MAP:
        for mapped_cat in CATEGORY_MAP[core_noun]:
            mapped_match = subset['category'].str.lower().str.contains(mapped_cat.lower(), na=False)
            if mapped_match.any():
                return True
    return False

# ------------------------------------------------------------
# 6. Helper functions for pricing, ranking, filtering
# ------------------------------------------------------------
def extract_budget(query):
    match = re.search(r'under\s*\$?(\d+(?:\.\d+)?)', query, re.IGNORECASE) or re.search(r'\$(\d+(?:\.\d+)?)', query)
    return float(match.group(1)) if match else None

def price_quality_bonus(price, budget, price_percentiles):
    if budget is None or np.isnan(price) or price <= 0:
        return 1.0
    if budget >= price_percentiles[2]:
        percentile = (price_vals < price).mean() * 100 if 'price_vals' in globals() else 50
        return 1.0 + 0.1 * (percentile / 100)
    return 1.0

def adaptive_price_penalty(price, budget, price_percentiles):
    if np.isnan(price) or budget is None:
        return 1.0
    if budget >= price_percentiles[2]:
        return 1.0
    if price <= budget:
        return 1.0
    over_ratio = price / budget
    return max(0.2, 0.6 ** (over_ratio - 1))

def get_target_categories(query, preferences):
    if isinstance(preferences, dict):
        preferences = json.dumps(preferences)
    text = (str(query) + " " + str(preferences)).lower()
    for kw, cats in CATEGORY_MAP.items():
        if kw in text:
            return cats
    return None

def create_rich_text(row):
    """Build the rich text string used for embedding."""
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

def initial_rank(metadata, embeddings, model, query, domain, preferences, price_percentiles, retrieval_k=50):
    budget = extract_budget(query)
    target_cats = get_target_categories(query, preferences)
    mask = metadata['type'] == domain
    if not mask.any():
        return []
    dm = metadata[mask].reset_index(drop=True)
    demb = embeddings[mask]
    qvec = model.encode([query])
    sims = cosine_similarity(qvec, demb).flatten()
    top_idx = np.argsort(sims)[-retrieval_k:][::-1]
    candidates = []
    for idx in top_idx:
        row = dm.iloc[idx]
        cos_sim = sims[idx]
        stars = row['rating_num']
        star_bonus = 1 + (stars - 3) / 5 if not pd.isna(stars) else 1.0
        star_bonus = max(0.5, min(1.5, star_bonus))
        price = row['price_numeric']
        price_penalty = adaptive_price_penalty(price, budget, price_percentiles)
        price_bonus = price_quality_bonus(price, budget, price_percentiles)
        cat = row['category']
        cat_penalty = 1.2 if target_cats and any(tc.lower() in cat.lower() for tc in target_cats) else 0.1
        final_score = cos_sim * star_bonus * price_penalty * price_bonus * cat_penalty
        candidates.append((idx, final_score, row.to_dict()))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:retrieval_k]

def llm_rerank(query, domain, preferences, candidates, top_k=20):
    if not candidates:
        return []
    if isinstance(preferences, dict):
        preferences = json.dumps(preferences)
    items = []
    for i, (_, _, row) in enumerate(candidates[:top_k]):
        p = row.get('price_numeric', 'N/A')
        price_str = f"${p:.2f}" if not np.isnan(p) else "N/A"
        items.append(f"{i+1}. {row.get('name', 'Unknown')} (Category: {row.get('category', 'N/A')}, Price: {price_str})")
    prompt = f"""
User: "{query}"
Target domain: {domain}
Preferences: {preferences}
Candidates (already scored by similarity):
{chr(10).join(items)}

Reorder the list from most to least relevant. Do NOT remove any. Return a JSON array of the original indices (1‑based) in the new order.
Example: [2,1,3,4,5,...]
"""
    result = call_llm(prompt, json_mode=True)
    if "error" in result or not isinstance(result, list):
        return candidates[:top_k]
    try:
        ordered = [candidates[int(i)-1] for i in result if 1 <= int(i) <= len(candidates)]
        return ordered[:top_k]
    except:
        return candidates[:top_k]

def mmr_diversity(candidates, lambda_param=0.6, top_k=10):
    if len(candidates) <= top_k:
        return candidates
    selected = []
    remaining = list(candidates)
    selected.append(remaining.pop(0))
    while len(selected) < top_k and remaining:
        max_mmr = -1
        best_idx = -1
        for i, (_, score, row) in enumerate(remaining):
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

# ------------------------------------------------------------
# 7. Main recommendation function
# ------------------------------------------------------------
def recommend_with_debug(metadata, embeddings, model, query, top_k=10, enable_diversity=True):
    debug = {}
    persona = normalise_persona(query)
    if "error" in persona:
        debug['error'] = persona['error']
        return f"Error: {persona['error']}", debug, [], []
    debug['normalised_persona'] = persona
    prefs = persona.get('preferences', '')
    if isinstance(prefs, dict):
        prefs = json.dumps(prefs)
    debug['preferences'] = prefs

    target_type = extract_target_type(query)
    core_noun = extract_core_noun(query)
    debug['target_type'] = target_type
    debug['core_noun'] = core_noun

    # Price percentiles for the catalogue (computed once)
    price_vals = metadata['price_numeric'].dropna().values
    price_percentiles = np.percentile(price_vals, [50, 80, 90, 95, 100]) if len(price_vals) > 0 else [0,0,0,0,0]

    exists_in_metadata = item_type_exists_in_metadata(metadata, target_type, core_noun)
    debug['exists_in_metadata'] = exists_in_metadata

    if not exists_in_metadata or target_type not in ['product', 'restaurant', 'book']:
        prompt = f"User: {query}\nPreferences: {prefs}\nRecommend {top_k} real-world {core_noun}s. Return a bullet list."
        rec = call_llm(prompt, json_mode=False)
        return rec, debug, [], []

    cand = initial_rank(metadata, embeddings, model, query, target_type, prefs, price_percentiles, retrieval_k=50)
    if not cand:
        return f"No items of type '{target_type}' found.", debug, [], []

    # Hard filter for laptop queries
    if target_type == 'product' and core_noun in CATEGORY_MAP:
        allowed_cats = CATEGORY_MAP[core_noun]
        forbidden = ['backpack', 'bag', 'case', 'sleeve', 'cover', 'accessory', 'charger', 'stand', 'mount', 'adapter', 'cable', 'keychain', 'gift', 'card', 'makeup', 'bundle']
        filtered = []
        for c in cand:
            cat = c[2].get('category', '')
            name = c[2].get('name', '').lower()
            if any(ac.lower() in cat.lower() for ac in allowed_cats):
                name_ok = core_noun in name
                if core_noun == 'laptop':
                    name_ok = name_ok or 'notebook' in name
                if name_ok:
                    if not any(f in name for f in forbidden):
                        filtered.append(c)
        if filtered:
            cand = filtered

    reranked = llm_rerank(query, target_type, prefs, cand, top_k=20)
    debug['top20_reranked'] = reranked[:20]

    if enable_diversity:
        final = mmr_diversity(reranked, lambda_param=0.6, top_k=top_k)
    else:
        final = reranked[:top_k]

    def candidates_to_df(candidates):
        rows = []
        for idx, score, row in candidates:
            row_cp = row.copy()
            row_cp['score'] = score
            price = row_cp.get('price_numeric', np.nan)
            row_cp['actual_price'] = f"${price:.2f}" if not np.isnan(price) else "N/A"
            rows.append(row_cp)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        display_cols = ['type', 'name', 'category', 'rating_num', 'actual_price', 'score']
        df = df[[c for c in display_cols if c in df.columns]].rename(columns={
            'type': 'Type', 'name': 'Name', 'category': 'Category',
            'rating_num': 'Rating', 'actual_price': 'Price', 'score': 'Score'
        })
        return df

    df_top20 = candidates_to_df(reranked[:20])
    df_final = candidates_to_df(final)
    return df_final, debug, df_top20, df_final

# ------------------------------------------------------------
# 8. Gradio UI (unchanged styling from previous version)
# ------------------------------------------------------------
# (CSS and UI functions – kept identical to the last working version)
# For brevity, I'll include the CSS and UI code exactly as in the previous final response.
# (The CSS and UI code from the previous answer is reused here.)
# ------------------------------------------------------------
CSS = """
/* ... (same CSS as before, omitted for brevity) ... */
"""
def star_str(rating):
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return "—"
    r = float(rating)
    full = round(r)
    return "★" * min(full, 5) + "☆" * max(0, 5 - full) + f" {r:.1f}"

def build_table(rows, max_score):
    trs = ""
    for i, r in enumerate(rows):
        score = r["score"]
        bar_w = max(0, min(100, round((score / max_score) * 100))) if max_score > 0 else 0
        trs += f"""
        <tr>
          <td><span class="rank-pill">{i+1}</span></td>
          <td><div class="name-cell">{r['name']}</div></td>
          <td style="color:#555;">{r['category']}</td>
          <td><span class="stars">{star_str(r['rating'])}</span></td>
          <td><span class="price-val">{r['price']}</span></td>
          <td>
            <div class="score-wrap">
              <div class="score-bg"><div class="score-fill" style="width:{bar_w}%"></div></div>
              <span class="score-val">{score:.3f}</span>
            </div>
          </td>
        </tr>"""
    return f"""
    <table class="rec-table">
      <thead><tr><th style="width:36px;">#</th><th>Name</th><th>Category</th><th>Rating</th><th>Price</th><th>Score</th></tr></thead>
      <tbody>{trs}</tbody>
    </tr>"""

def df_to_rows(df):
    rows = []
    for _, row in df.iterrows():
        rating = row.get("Rating", None)
        if isinstance(rating, float) and np.isnan(rating):
            rating = None
        rows.append({
            "name": row.get("Name", "—"),
            "category": row.get("Category", "—"),
            "rating": rating,
            "price": row.get("Price", "N/A"),
            "score": float(row.get("Score", 0)),
        })
    return rows

def format_output(metadata, embeddings, model, query, enable_diversity):
    results, debug, df_top20, df_final = recommend_with_debug(metadata, embeddings, model, query, top_k=10, enable_diversity=enable_diversity)

    persona_json = json.dumps(debug.get("normalised_persona", {}), indent=2)
    target_type = debug.get("target_type", "N/A")
    core_noun = debug.get("core_noun", "N/A")
    exists = debug.get("exists_in_metadata", False)
    prefs = debug.get("preferences", "N/A")

    exists_badge = '<span class="dbg-badge dbg-badge-green">yes</span>' if exists else '<span class="dbg-badge dbg-badge-red">no</span>'
    diversity_badge = '<span class="dbg-badge dbg-badge-green">enabled</span>' if enable_diversity else '<span class="dbg-badge dbg-badge-red">disabled</span>'

    debug_html = f"""
    <div class="dbg-panel">
      <div class="dbg-row"><span class="dbg-key">Query</span><span class="dbg-val">{query}</span></div>
      <div class="dbg-row"><span class="dbg-key">Target type</span><span class="dbg-badge dbg-badge-purple">{target_type}</span></div>
      <div class="dbg-row"><span class="dbg-key">Core noun</span><span class="dbg-badge dbg-badge-purple">{core_noun}</span></div>
      <div class="dbg-row"><span class="dbg-key">In catalog</span>{exists_badge}</div>
      <div class="dbg-row"><span class="dbg-key">Diversity</span>{diversity_badge}</div>
      <div class="dbg-row"><span class="dbg-key">Preferences</span><span class="dbg-val">{prefs}</span></div>
      <div class="dbg-persona-toggle">
        <details><summary class="dbg-persona-summary">Normalised persona</summary><pre class="dbg-persona-pre">{persona_json}</pre></details>
      </div>
    </div>
    """

    if "error" in debug:
        return debug_html + f'<div class="error-box">Error: {debug["error"]}</div>'

    if isinstance(results, str):
        return debug_html + f'<div class="llm-fallback">{results}</div>'

    top20_html = ""
    if not df_top20.empty:
        rows20 = df_to_rows(df_top20)
        max_score = max((r["score"] for r in rows20), default=1)
        table20 = build_table(rows20, max_score)
        top20_html = f"""
        <div class="collapsible-panel">
          <div class="collapsible-header" onclick="
            var b=this.nextElementSibling;
            var open=b.style.display==='block';
            b.style.display=open?'none':'block';
            this.setAttribute('open-state', open?'':'open');
            this.querySelector('.collapsible-arrow').textContent=open?'▼':'▲';
          ">
            <span class="collapsible-title">Top 20 candidates <span class="collapsible-count">after LLM re‑ranking</span></span>
            <span class="collapsible-arrow">▼</span>
          </div>
          <div class="collapsible-body">{table20}</div>
        </div>
        """

    final_html = ""
    if not df_final.empty:
        rows_f = df_to_rows(df_final)
        max_score_f = max((r["score"] for r in rows_f), default=1)
        table_f = build_table(rows_f, max_score_f)
        final_html = f"""
        <div class="final-header"><p class="section-label" style="margin:0;">Results</p><span class="final-badge">Top {len(rows_f)} picks</span></div>
        <div class="final-wrap">{table_f}</div>
        """

    return debug_html + top20_html + final_html

# ------------------------------------------------------------
# 9. Gradio app initialization
# ------------------------------------------------------------
print("Loading catalogue...")
metadata = load_catalogue(CATALOGUE_PATH)
print(f"Loaded {len(metadata)} items. Generating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
metadata['rich_text'] = metadata.apply(create_rich_text, axis=1)
embeddings = model.encode(metadata['rich_text'].tolist(), show_progress_bar=True, batch_size=256)
print("Ready.")

with gr.Blocks(css=CSS, title="Recommendation Agent") as demo:
    with gr.Column(elem_classes="page-wrap"):
        gr.HTML("""
        <div class="page-header">
          <p class="page-title">Recommendation Agent</p>
          <p class="page-desc">Describe what you are looking for. The agent extracts intent, checks the catalog, then ranks results via cosine similarity and LLM re‑ranking.</p>
        </div>
        """)
        with gr.Column(elem_classes="input-card"):
            inp = gr.Textbox(label="Persona description", lines=3, placeholder="e.g. I need a lightweight laptop for students under $500")
            with gr.Row():
                diversity_cb = gr.Checkbox(label="Enable diversity (MMR)", value=True)
                btn = gr.Button("Get recommendations", variant="primary")
        out = gr.HTML(elem_classes="output-html")
    btn.click(fn=lambda q, d: format_output(metadata, embeddings, model, q, d), inputs=[inp, diversity_cb], outputs=out)

if __name__ == "__main__":
    demo.launch(share=True, strict_cors=False)