import json
import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------- OpenRouter Client ----------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = "openai/gpt-4o-mini"

# ---------- Nigerian Slang Dictionary ----------
NIGERIAN_SLANGS = {
    "abeg": "please",
    "sef": "anyway / just",
    "mtchew": "expression of disgust (like tch!)",
    "wahala": "trouble / problem",
    "naija": "Nigeria",
    "chop": "to eat",
    "sabi": "to know / understand",
    "comot": "to remove / get out",
    "dey": "is / are / be (continuous)",
    "ole": "thief",
    "oya": "hurry up / let's go",
    "sha": "anyway / after all",
    "no wahala": "no problem / okay",
    "you get me?": "you understand?",
    "notin": "nothing",
    "wetin": "what",
    "bamboo": "fake / counterfeit",
    "fall my hand": "disappoint me",
    "god abeg": "please God (expression of exasperation)",
    "sharp": "quick / smart",
    "japa": "to flee / emigrate",
    "mumu": "fool / foolish",
    "craze": "crazy / mad",
    "sweet": "good / tasty / enjoyable",
    "suffer head": "foolish / stupid",
    "gbosa": "applause / something excellent",
    "shayo": "alcohol / drink",
    "pepper": "spice / trouble / good looks",
    "wahala dey": "there is trouble",
    "shey mean?": "you know? / right?",
}


def get_slang_instruction(country, preferences):
    if country != "Nigeria":
        return ""

    pref_lower = preferences.lower()
    if any(phrase in pref_lower for phrase in ["no slang", "formal", "proper english", "no pidgin"]):
        return ""

    slang_list = "\n".join([f"- {word}: {meaning}" for word, meaning in NIGERIAN_SLANGS.items()])
    if any(phrase in pref_lower for phrase in ["slang", "pidgin", "allow slang"]):
        return f"""
Nigerian Slang & Pidgin Guide (the persona is comfortable with slang, use it naturally but not excessively):
{slang_list}
Rules: Use slang occasionally (1-2 max per review) to match the persona's voice.
"""
    else:
        return f"""
Nigerian Slang & Pidgin Guide (use ONLY if it fits the persona's voice – keep it very subtle):
{slang_list}
Rules: Avoid forcing slang. If the persona seems formal, skip slang entirely.
"""


def call_llm(prompt, json_mode=True):
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


def normalise_persona(free_text):
    prompt = f"""
Convert this persona description into a structured JSON profile.
Fields:
- name (extract or invent a short first name that fits the persona)
- country (if mentioned, e.g., Nigeria, Ghana, UK, else "Unknown")
- description (a 1-2 sentence bio that captures the persona's background, personality, and writing style)
- preferences (any explicit desires about writing style, language use, rating behaviour, likes/dislikes, tone, etc.)
- avg_rating (1-5, estimate from description, default 3.0)
- rating_std (0-2, estimate consistency, default 1.0)
- total_reviews (estimate, default 5)
- example_reviews (generate 2 short example reviews this person would write, reflecting their style and preferences)

Persona: {free_text}

Return ONLY valid JSON:
{{"name": "...", "country": "...", "description": "...", "preferences": "...", "avg_rating": X.X, "rating_std": X.X, "total_reviews": N, "example_reviews": ["...", "..."]}}
"""
    return call_llm(prompt, json_mode=True)


def generate_review(persona_text, product_title, product_price):
    if not persona_text.strip():
        return {}, "Please enter a persona description.", ""

    profile = normalise_persona(persona_text)
    if "error" in profile:
        return {}, f"Error: {profile['error']}", ""

    country = profile.get("country", "Unknown")
    preferences = profile.get("preferences", "")
    slang_instruction = get_slang_instruction(country, preferences)
    examples = "\n".join(
        f"- {ex[:200]}" for ex in profile.get("example_reviews", [])[:2]
    )

    generation_prompt = f"""
You are {profile.get('name', 'a reviewer')} from {country}.
Persona description: {profile.get('description', '')}
Preferences: {preferences}

Your typical rating: {profile['avg_rating']} stars (std={profile['rating_std']}).
Examples of your writing:
{examples}

{slang_instruction}

Now review this product:
Title: {product_title or "a product"}
Price: {product_price or "unspecified"}

Write a short, authentic review (2-3 sentences) that matches your persona and preferences exactly.
Return JSON: {{"rating": (integer 1-5), "review": "text"}}
"""
    result = call_llm(generation_prompt, json_mode=True)
    review = result.get("review", "No review generated.")
    rating = result.get("rating", 3)
    try:
        rating = max(1, min(5, int(rating)))
    except Exception:
        rating = 3

    rating_display = f"{rating} / 5"

    return profile, f'"{review}"', rating_display


# ---------- Persona library ----------
DEFAULT_PERSONAS = {
    "Student in Lagos": "I am Ada, a 22-year-old university student in Lagos. I hate expensive items and write short, angry reviews. I use small Pidgin sometimes.",
    "Mechanic in Enugu": "I am Chidi, a mechanic from Enugu. I don't like wasting money. I write short reviews and might use 'mtchew' when annoyed.",
    "Tech bro in Abuja": "I am Femi, a software developer in Abuja. I am analytical, rarely give 5 stars, focus on functionality. No slang.",
    "Retired teacher": "I am Mrs. Grace, a retired teacher from Ibadan. I am generous with ratings but write long, detailed reviews. Very proper English.",
}

persona_library: dict = dict(DEFAULT_PERSONAS)


def get_filtered_choices(search_term=""):
    term = search_term.strip().lower()
    if not term:
        return list(persona_library.keys())
    return [
        k for k in persona_library
        if term in k.lower() or term in persona_library[k].lower()
    ]


def search_personas(search_term):
    choices = get_filtered_choices(search_term)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def fill_from_library(name):
    return persona_library.get(name, "") if name else ""


def add_persona(name, description):
    name = name.strip()
    description = description.strip()
    if not name:
        return gr.update(), gr.update(), "Please enter a name."
    if not description:
        return gr.update(), gr.update(), "Please enter a description."
    persona_library[name] = description
    choices = get_filtered_choices()
    return (
        gr.update(choices=choices, value=name),
        gr.update(value=""),
        f"'{name}' added.",
    )


def delete_persona(name):
    if not name or name not in persona_library:
        return gr.update(), "Select a persona first."
    if name in DEFAULT_PERSONAS:
        return gr.update(), f"Can't delete built-in persona '{name}'."
    del persona_library[name]
    choices = get_filtered_choices()
    return (
        gr.update(choices=choices, value=choices[0] if choices else None),
        f"'{name}' removed.",
    )


# ---------- CSS ----------
css = """
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body { width: 100%; height: 100%; margin: 0; padding: 0; }

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: #f7f4ef !important;
    color: #1e1e1e !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 1.5rem 2rem !important;
}

.gradio-container > .main > .wrap { padding: 0 !important; }
footer { display: none !important; }

.uma-header {
    text-align: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #e0d9cf;
}
.uma-header h1 {
    font-family: 'Lora', Georgia, serif !important;
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: #1e1e1e !important;
    letter-spacing: -0.5px;
    margin-bottom: 0.25rem;
}
.uma-header h1 em { color: #c0522a; font-style: italic; }
.uma-header p {
    font-size: 0.82rem;
    color: #7a7068;
    font-weight: 300;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

.panel {
    background: #ffffff !important;
    border: 1px solid #e0d9cf !important;
    border-radius: 14px !important;
    padding: 1.25rem !important;
}

.section-label {
    font-size: 0.65rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.3px !important;
    color: #9a8f82 !important;
    margin: 0 0 0.75rem 0 !important;
    padding-bottom: 0.4rem !important;
    border-bottom: 1px solid #f0ebe3 !important;
}
.section-label-mt {
    margin-top: 1rem !important;
}

textarea, input[type="text"],
.gr-textbox textarea, .gr-textbox input {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    background: #faf8f5 !important;
    border: 1px solid #e0d9cf !important;
    border-radius: 9px !important;
    color: #1e1e1e !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #c0522a !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(192,82,42,0.08) !important;
}

.gr-dropdown select, select {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    background: #faf8f5 !important;
    border: 1px solid #e0d9cf !important;
    border-radius: 9px !important;
    color: #1e1e1e !important;
}

label span, .gr-form label {
    font-size: 0.76rem !important;
    font-weight: 500 !important;
    color: #5a504a !important;
}

.btn-ghost button {
    background: #f0ebe3 !important;
    border: 1px solid #e0d9cf !important;
    border-radius: 9px !important;
    color: #5a504a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    height: 38px !important;
    transition: all 0.15s !important;
}
.btn-ghost button:hover { background: #e5ded5 !important; color: #1e1e1e !important; }

.btn-danger button {
    background: #fdf0ee !important;
    border: 1px solid #f5ccc5 !important;
    border-radius: 9px !important;
    color: #b03020 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    height: 38px !important;
    transition: all 0.15s !important;
}
.btn-danger button:hover { background: #f5ccc5 !important; color: #801c10 !important; }

.btn-primary button {
    background: #c0522a !important;
    border: none !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
    transition: all 0.15s !important;
}
.btn-primary button:hover { background: #a04020 !important; transform: translateY(-1px) !important; }

.gr-tab-nav { border-bottom: 1px solid #e0d9cf !important; margin-bottom: 1rem !important; }
.gr-tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #9a8f82 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 0.5rem 1rem !important;
    margin-bottom: -1px !important;
    transition: all 0.15s !important;
}
.gr-tab-nav button.selected {
    color: #c0522a !important;
    border-bottom-color: #c0522a !important;
}

.gr-json {
    background: #faf8f5 !important;
    border: 1px solid #e0d9cf !important;
    border-radius: 9px !important;
    font-size: 0.78rem !important;
}

.review-out textarea {
    font-family: 'Lora', Georgia, serif !important;
    font-style: italic !important;
    font-size: 0.98rem !important;
    line-height: 1.75 !important;
    color: #2a2018 !important;
    background: #fdf9f5 !important;
    border-left: 3px solid #c0522a !important;
    border-top: 1px solid #e0d9cf !important;
    border-right: 1px solid #e0d9cf !important;
    border-bottom: 1px solid #e0d9cf !important;
    border-radius: 0 9px 9px 0 !important;
    padding: 1rem 1.25rem !important;
}

.rating-out textarea {
    font-family: 'Lora', Georgia, serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #c0522a !important;
    text-align: center !important;
    background: #fdf5f1 !important;
    border: 1px solid #f0d5c8 !important;
    border-radius: 9px !important;
    resize: none !important;
    height: 52px !important;
    padding-top: 12px !important;
    line-height: 1.4 !important;
}

.status-msg textarea {
    font-size: 0.78rem !important;
    color: #5a504a !important;
    background: transparent !important;
    border: none !important;
    padding: 0.1rem 0 !important;
    resize: none !important;
    box-shadow: none !important;
    height: auto !important;
    min-height: 0 !important;
}
.status-msg > label > span { display: none !important; }

.tips-box {
    background: #fdf5f1;
    border-left: 3px solid #c0522a;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin-top: 0.75rem;
    font-size: 0.8rem;
    color: #5a504a;
    line-height: 1.75;
}

.uma-footer {
    text-align: center;
    margin-top: 1.5rem;
    font-size: 0.7rem;
    color: #b0a898;
    letter-spacing: 0.3px;
}

.gr-markdown p { margin: 0 !important; }
.gr-form { gap: 0.6rem !important; }
"""

# ---------- UI ----------
with gr.Blocks(title="User Modeling Agent", css=css) as demo:

    gr.HTML("""
    <div class="uma-header">
        <h1>User Modeling Agent</h1>
        <p>Two-pass normaliser &middot; Persona-driven review generation</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, elem_classes="panel"):

            gr.HTML('<p class="section-label">Persona &amp; product</p>')

            persona_input = gr.Textbox(
                label="Persona description",
                lines=4,
                placeholder="Describe the reviewer — background, mood, rating habits…",
            )
            product_title = gr.Textbox(
                label="Product title",
                placeholder="e.g. Luxury Leather Slippers",
            )
            product_price = gr.Textbox(
                label="Price",
                placeholder="e.g. $200 / ₦150,000",
            )

            gr.HTML('<p class="section-label section-label-mt">Persona library</p>')

            lib_search = gr.Textbox(
                label="Search",
                placeholder="Filter by name or keywords…",
            )
            lib_dropdown = gr.Dropdown(
                choices=list(persona_library.keys()),
                value=list(persona_library.keys())[0],
                label="Select persona",
                interactive=True,
            )

            with gr.Row():
                lib_fill_btn = gr.Button("Load into editor", elem_classes="btn-ghost", scale=3)
                lib_delete_btn = gr.Button("Delete", elem_classes="btn-danger", scale=1)

            lib_status = gr.Textbox(
                show_label=False,
                interactive=False,
                elem_classes="status-msg",
                container=False,
                lines=1,
            )

            gr.HTML('<p class="section-label section-label-mt">Add to library</p>')

            new_name = gr.Textbox(
                label="Name",
                placeholder="e.g. Market trader in Kano",
            )
            new_desc = gr.Textbox(
                label="Description",
                placeholder="Describe this persona…",
                lines=2,
            )
            add_btn = gr.Button("Add to library", elem_classes="btn-ghost")

            gr.HTML('<div style="margin-top:0.75rem"></div>')

            submit_btn = gr.Button(
                "Generate review",
                variant="primary",
                elem_classes="btn-primary",
            )

            gr.HTML("""
            <div class="tips-box">
                <strong>Tips</strong><br>
                Mention <strong>country</strong> (e.g. Nigeria) to activate Pidgin slang.<br>
                Describe <strong>rating style</strong> — harsh, generous, or mixed.<br>
                Add <strong>writing style</strong> — terse, emotional, or detailed.
            </div>
            """)

        with gr.Column(scale=2, elem_classes="panel"):

            gr.HTML('<p class="section-label">Output</p>')

            with gr.Tabs():

                with gr.TabItem("Pass 1 — normalised profile"):
                    profile_output = gr.JSON(label="Canonical persona profile")

                with gr.TabItem("Pass 2 — generated review"):
                    review_output = gr.Textbox(
                        label="Review",
                        lines=6,
                        interactive=False,
                        elem_classes="review-out",
                    )
                    rating_output = gr.Textbox(
                        label="Rating",
                        interactive=False,
                        elem_classes="rating-out",
                        lines=1,
                    )

    gr.HTML("""
    <div class="uma-footer">
        Powered by OpenRouter (GPT-4o-mini) &middot; Nigerian slang dictionary &middot; Two-pass normalisation
    </div>
    """)

    # ---- Events ----
    lib_search.change(fn=search_personas, inputs=lib_search, outputs=lib_dropdown)

    lib_fill_btn.click(fn=fill_from_library, inputs=lib_dropdown, outputs=persona_input)

    lib_delete_btn.click(
        fn=delete_persona,
        inputs=lib_dropdown,
        outputs=[lib_dropdown, lib_status],
    )

    add_btn.click(
        fn=add_persona,
        inputs=[new_name, new_desc],
        outputs=[lib_dropdown, new_name, lib_status],
    )

    submit_btn.click(
        fn=generate_review,
        inputs=[persona_input, product_title, product_price],
        outputs=[profile_output, review_output, rating_output],
    )

if __name__ == "__main__":
    demo.launch()
