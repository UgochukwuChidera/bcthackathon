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
# 1. Column mapping (exact match, case‑insensitive)
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
# 2. LLM client and persona normaliser
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
# 3. Helper functions for pricing, ranking, etc.
# ------------------------------------------------------------
def extract_budget(query):
    match = re.search(r'under\s*\$?(\d+(?:\.\d+)?)', query, re.IGNORECASE) or re.search(r'\$(\d+(?:\.\d+)?)', query)
    return float(match.group(1)) if match else None

def get_target_categories(query, preferences):
    if isinstance(preferences, dict):
        preferences = json.dumps(preferences)
    text = (str(query) + " " + str(preferences)).lower()
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

# Domain keywords for explicit filtering
DOMAIN_KEYWORDS = {
    "product": ["laptop", "computer", "phone", "headphones", "tv", "monitor", "shaving", "clipper", "trimmer"],
    "restaurant": ["restaurant", "cafe", "bar", "eat", "dinner", "lunch", "food"],
    "book": ["book", "novel", "read", "fiction", "literature", "story"]
}

def detect_explicit_domain(query):
    q_lower = query.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                return domain
    return None

def initial_rank_unified(metadata, embeddings, model, query, preferences, price_percentiles, retrieval_k=50):
    """Retrieve from all types (no domain filter)."""
    budget = extract_budget(query)
    target_cats = get_target_categories(query, preferences)
    qvec = model.encode([query])
    sims = cosine_similarity(qvec, embeddings).flatten()
    top_idx = np.argsort(sims)[-retrieval_k:][::-1]
    candidates = []
    for idx in top_idx:
        row = metadata.iloc[idx]
        cos_sim = sims[idx]
        stars = row['rating_num']
        star_bonus = 1 + (stars - 3) / 5 if not pd.isna(stars) else 1.0
        star_bonus = max(0.5, min(1.5, star_bonus))
        price = row['price_numeric']
        if budget is not None and not np.isnan(price) and price > 0 and budget < price_percentiles[2]:
            price_penalty = 0.6 ** (price / budget - 1) if price > budget else 1.0
            price_penalty = max(0.2, price_penalty)
        else:
            price_penalty = 1.0
        price_bonus = 1.0
        if budget is not None and not np.isnan(price) and budget >= price_percentiles[2]:
            percentile = (metadata['price_numeric'].dropna().values < price).mean() * 100
            price_bonus = 1.0 + 0.1 * (percentile / 100)
        cat = row['category']
        cat_penalty = 1.2 if target_cats and any(tc.lower() in cat.lower() for tc in target_cats) else 0.1
        final_score = cos_sim * star_bonus * price_penalty * price_bonus * cat_penalty
        candidates.append((idx, final_score, row.to_dict()))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:retrieval_k]

def initial_rank_filtered(metadata, embeddings, model, query, domain, preferences, price_percentiles, retrieval_k=50):
    """Retrieve only from a specific domain."""
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
        if budget is not None and not np.isnan(price) and price > 0 and budget < price_percentiles[2]:
            price_penalty = 0.6 ** (price / budget - 1) if price > budget else 1.0
            price_penalty = max(0.2, price_penalty)
        else:
            price_penalty = 1.0
        price_bonus = 1.0
        if budget is not None and not np.isnan(price) and budget >= price_percentiles[2]:
            percentile = (dm['price_numeric'].dropna().values < price).mean() * 100
            price_bonus = 1.0 + 0.1 * (percentile / 100)
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

def recommend_with_debug(metadata, embeddings, model, query, top_k=10, enable_diversity=True):
    debug = {}

    # 1. Persona normalisation
    persona = normalise_persona(query)
    if "error" in persona:
        debug['error'] = persona['error']
        return f"Error: {persona['error']}", debug, [], []
    debug['normalised_persona'] = persona
    prefs = persona.get('preferences', '')
    if isinstance(prefs, dict):
        prefs = json.dumps(prefs)
    debug['preferences'] = prefs

    # 2. Detect explicit domain from query
    explicit_domain = detect_explicit_domain(query)
    debug['explicit_domain'] = explicit_domain if explicit_domain else "None"

    # 3. Price percentiles
    price_vals = metadata['price_numeric'].dropna().values
    price_percentiles = np.percentile(price_vals, [50, 80, 90, 95, 100]) if len(price_vals) > 0 else [0,0,0,0,0]

    # 4. Choose retrieval method
    if explicit_domain:
        cand = initial_rank_filtered(metadata, embeddings, model, query, explicit_domain, prefs, price_percentiles, retrieval_k=50)
        domain_used = explicit_domain
    else:
        cand = initial_rank_unified(metadata, embeddings, model, query, prefs, price_percentiles, retrieval_k=50)
        domain_used = "unified (all types)"

    debug['domain_used'] = domain_used

    if not cand:
        return f"No items found.", debug, [], []

    # 5. Hard filter for laptop queries
    core_noun = None
    if explicit_domain == "product":
        words = query.split()
        if words:
            core_noun = words[-1].lower()
        laptop_keywords = ["laptop", "notebook", "computer"]
        for kw in laptop_keywords:
            if kw in query.lower():
                core_noun = kw
                break
        if core_noun in ["laptop", "notebook", "computer"]:
            allowed_cats = ["Computers & Tablets", "Laptops", "Computer"]
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

    # 6. LLM re‑ranking
    reranked = llm_rerank(query, domain_used, prefs, cand, top_k=20)
    debug['top20_reranked'] = reranked[:20]

    # 7. Diversity
    if enable_diversity:
        final = mmr_diversity(reranked, lambda_param=0.6, top_k=top_k)
    else:
        final = reranked[:top_k]

    # 8. Convert to DataFrame
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
# 4. Table rendering — fixed contrast + bold #1 item
# ------------------------------------------------------------

# Palette
_ACCENT      = "#4a6fa5"   # muted steel-blue: rank pill, score bar, #1 accent stripe
_ACCENT_LITE = "#dce6f5"   # light blue tint: #1 row background
_ROW_ALT     = "#f7f5f2"   # warm off-white: even rows
_ROW_BASE    = "#ffffff"   # white: odd rows
_TEXT_MAIN   = "#1a1a1a"   # near-black: high contrast body text
_TEXT_MID    = "#444444"   # mid-grey: secondary text
_BORDER      = "#e0dbd4"   # warm light border

CSS = """
body, .gradio-container { font-family: system-ui, -apple-system, sans-serif; }
.page-wrap { max-width: 960px; margin: 0 auto; padding: 24px 16px; }
.page-header { margin-bottom: 20px; }
.page-title { font-size: 24px; font-weight: 700; color: #1a1a1a; margin: 0 0 6px; }
.page-desc  { font-size: 14px; color: #555; margin: 0; }
.input-card { background: #f9f7f4; border: 1px solid #e0dbd4; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
.output-html { margin-top: 8px; }

/* Debug panel */
.dbg-panel { background: #1e1e2e; color: #cdd6f4; border-radius: 8px; padding: 14px 16px; margin-bottom: 14px; font-size: 12px; line-height: 1.6; }
.dbg-row   { display: flex; gap: 10px; margin-bottom: 4px; flex-wrap: wrap; }
.dbg-key   { color: #89b4fa; font-weight: 600; min-width: 140px; }
.dbg-val   { color: #cdd6f4; }
.dbg-badge { display: inline-block; padding: 1px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.dbg-badge-purple { background: #6c5fc7; color: #fff; }
.dbg-badge-green  { background: #2d9e5f; color: #fff; }
.dbg-persona-toggle { margin-top: 8px; }
.dbg-persona-summary { cursor: pointer; color: #89b4fa; font-weight: 600; }
.dbg-persona-pre { background: #13131f; color: #a6e3a1; border-radius: 6px; padding: 10px; margin-top: 6px; font-size: 11px; overflow-x: auto; white-space: pre-wrap; }

.error-box    { background: #fff0f0; border: 1px solid #e57373; border-radius: 8px; padding: 12px; color: #c62828; }
.llm-fallback { background: #fffde7; border: 1px solid #f9a825; border-radius: 8px; padding: 12px; color: #5d4037; }
"""

def star_str(rating):
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return "—"
    r = float(rating)
    full = round(r)
    return "★" * min(full, 5) + "☆" * max(0, 5 - full) + f" {r:.1f}"


def build_table(rows, max_score):
    header_style = (
        f"background:#f0ede8; color:#333; font-size:11px; font-weight:700; "
        f"text-transform:uppercase; letter-spacing:.06em; padding:10px 12px; "
        f"border-bottom:2px solid {_BORDER}; white-space:nowrap;"
    )

    trs = ""
    for i, r in enumerate(rows):
        score  = r["score"]
        bar_w  = max(0, min(100, round((score / max_score) * 100))) if max_score > 0 else 0
        is_top = (i == 0)

        row_bg   = _ACCENT_LITE if is_top else (_ROW_ALT if i % 2 == 0 else _ROW_BASE)
        fw       = "700" if is_top else "400"
        border_l = f"border-left:3px solid {_ACCENT};" if is_top else "border-left:3px solid transparent;"

        pill_style = (
            f"background:{_ACCENT}; color:#fff; border-radius:50%; "
            "display:inline-block; width:22px; height:22px; line-height:22px; "
            "text-align:center; font-size:12px; font-weight:700;"
        )
        name_style = (
            f"font-weight:{fw}; color:{_TEXT_MAIN}; font-size:13px; "
            "max-width:420px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;"
        )
        cat_style   = f"color:{_TEXT_MID}; font-size:12px; font-weight:{fw};"
        star_style  = f"color:#c8860a; font-size:12px; font-weight:{fw};"
        price_style = f"color:{_TEXT_MAIN}; font-weight:{fw}; font-size:13px; white-space:nowrap;"
        td_pad      = "padding:9px 12px; vertical-align:middle;"

        trs += f"""
        <tr style="background:{row_bg}; {border_l}">
          <td style="{td_pad} width:36px; text-align:center;">
            <span style="{pill_style}">{i+1}</span>
          </td>
          <td style="{td_pad}" title="{r['name']}">
            <div style="{name_style}">{r['name']}</div>
          </td>
          <td style="{td_pad}"><span style="{cat_style}">{r['category']}</span></td>
          <td style="{td_pad}"><span style="{star_style}">{star_str(r['rating'])}</span></td>
          <td style="{td_pad}"><span style="{price_style}">{r['price']}</span></td>
          <td style="{td_pad} min-width:120px;">
            <div style="display:flex; align-items:center; gap:6px;">
              <div style="flex:1; height:6px; background:#ddd; border-radius:3px; overflow:hidden;">
                <div style="width:{bar_w}%; height:100%; background:{_ACCENT}; border-radius:3px;"></div>
              </div>
              <span style="font-size:11px; color:{_TEXT_MID}; font-weight:{fw}; min-width:38px; text-align:right;">
                {score:.3f}
              </span>
            </div>
          </td>
        </tr>"""

    th = f"<th style='{header_style}'>"
    return f"""
    <table style="width:100%; border-collapse:collapse; font-family:system-ui,sans-serif;">
      <thead>
        <tr>
          {th}#</th>
          {th}Name</th>
          {th}Category</th>
          {th}Rating</th>
          {th}Price</th>
          {th}Score</th>
        </tr>
      </thead>
      <tbody>{trs}</tbody>
    </table>"""


def df_to_rows(df):
    rows = []
    for _, row in df.iterrows():
        rating = row.get("Rating", None)
        if isinstance(rating, float) and np.isnan(rating):
            rating = None
        rows.append({
            "name":     row.get("Name", "—"),
            "category": row.get("Category", "—"),
            "rating":   rating,
            "price":    row.get("Price", "N/A"),
            "score":    float(row.get("Score", 0)),
        })
    return rows

# ------------------------------------------------------------
# 5. Main format_output
# ------------------------------------------------------------
def format_output(metadata, embeddings, model, query, enable_diversity):
    results, debug, df_top20, df_final = recommend_with_debug(metadata, embeddings, model, query, top_k=10, enable_diversity=enable_diversity)

    persona_json   = json.dumps(debug.get("normalised_persona", {}), indent=2)
    explicit_domain = debug.get("explicit_domain", "None")
    domain_used    = debug.get("domain_used", "None")
    prefs          = debug.get("preferences", "N/A")

    debug_html = f"""
    <div class="dbg-panel">
      <div class="dbg-row"><span class="dbg-key">Query</span><span class="dbg-val">{query}</span></div>
      <div class="dbg-row"><span class="dbg-key">Explicit domain</span><span class="dbg-badge dbg-badge-purple">{explicit_domain}</span></div>
      <div class="dbg-row"><span class="dbg-key">Retrieval domain</span><span class="dbg-badge dbg-badge-purple">{domain_used}</span></div>
      <div class="dbg-row"><span class="dbg-key">Diversity</span><span class="dbg-badge dbg-badge-green">{"enabled" if enable_diversity else "disabled"}</span></div>
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
        rows20    = df_to_rows(df_top20)
        max_score = max((r["score"] for r in rows20), default=1)
        table20   = build_table(rows20, max_score)
        top20_html = f"""
        <details style="margin-bottom:16px;">
            <summary style="font-weight:bold; cursor:pointer; color:#1a1a1a;">
              📋 Top 20 Candidates (after re‑ranking)
            </summary>
            <div style="overflow-x:auto; border:1px solid {_BORDER}; background:white; margin-top:8px; border-radius:6px;">
              {table20}
            </div>
        </details>
        """

    final_html = ""
    if not df_final.empty:
        rows_f      = df_to_rows(df_final)
        max_score_f = max((r["score"] for r in rows_f), default=1)
        table_f     = build_table(rows_f, max_score_f)
        final_html  = f"""
        <div style="overflow-x:auto; border-radius:8px; border:2px solid {_ACCENT}; background:white;">
            {table_f}
        </div>
        """

    return debug_html + top20_html + final_html

# ------------------------------------------------------------
# 6. App initialization: load or generate embeddings
# ------------------------------------------------------------
print("Starting Recommendation Agent...")
model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists(PRE_COMPUTED_EMBEDDINGS) and os.path.exists(PRE_COMPUTED_METADATA):
    print("Loading pre‑computed embeddings and metadata...")
    embeddings = np.load(PRE_COMPUTED_EMBEDDINGS)
    metadata   = pd.read_csv(PRE_COMPUTED_METADATA)
    print(f"Loaded {len(metadata)} items from pre‑computed files.")
else:
    print("Pre‑computed files not found. Looking for catalogue CSV...")
    catalogue_candidates = ['catalogue.csv', CATALOGUE_PATH, '/data/catalogue.csv']
    catalogue_path = None
    for cand in catalogue_candidates:
        if os.path.exists(cand):
            catalogue_path = cand
            break
    if catalogue_path is None:
        raise FileNotFoundError(
            "No pre‑computed embeddings and no catalogue CSV found. "
            "Please provide a catalogue CSV (catalogue.csv, or set CATALOGUE_PATH)."
        )
    print(f"Loading catalogue from {catalogue_path}...")
    metadata = load_catalogue(catalogue_path)
    print(f"Loaded {len(metadata)} items. Generating embeddings...")
    metadata['rich_text'] = metadata.apply(create_rich_text, axis=1)
    embeddings = model.encode(metadata['rich_text'].tolist(), show_progress_bar=True, batch_size=256)
    print("Embeddings generated.")
    np.save(PRE_COMPUTED_EMBEDDINGS, embeddings)
    metadata.to_csv(PRE_COMPUTED_METADATA, index=False)
    print(f"Saved embeddings to {PRE_COMPUTED_EMBEDDINGS} and metadata to {PRE_COMPUTED_METADATA}")

# ------------------------------------------------------------
# 7. Gradio UI
# ------------------------------------------------------------
with gr.Blocks(css=CSS, title="Recommendation Agent") as demo:
    with gr.Column(elem_classes="page-wrap"):
        gr.HTML("""
        <div class="page-header">
          <p class="page-title">Recommendation Agent</p>
          <p class="page-desc">Describe what you are looking for. The agent extracts intent, checks the catalog, then ranks results via cosine similarity and LLM re‑ranking.</p>
        </div>
        """)
        with gr.Column(elem_classes="input-card"):
            inp = gr.Textbox(label="Persona description", lines=3, placeholder="e.g., I need a lightweight laptop for students under $500")
            with gr.Row():
                diversity_cb = gr.Checkbox(label="Enable diversity (MMR)", value=True)
                btn = gr.Button("Get recommendations", variant="primary")
        out = gr.HTML(elem_classes="output-html")
    btn.click(fn=lambda q, d: format_output(metadata, embeddings, model, q, d), inputs=[inp, diversity_cb], outputs=out)

if __name__ == "__main__":
    demo.launch()