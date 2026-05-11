import gradio as gr
import json
import google.generativeai as genai

# Load profiles
with open("user_profiles_improved.json") as f:
    profiles = json.load(f)

genai.configure(api_key="YOUR_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_review(persona_name, product_title, product_price):
    profile = profiles.get(persona_name)
    if not profile:
        # Fallback: use two-pass normaliser (we'll write later)
        return "Persona not found – fallback not implemented yet", 3
    
    prompt = f"""
You are a reviewer from {profile['country']}. You have written {profile['total_reviews']} reviews.
Your average rating is {profile['avg_rating']} stars (std={profile['rating_std']}).
Examples of your writing:
{chr(10).join(f'- {ex}' for ex in profile['example_reviews'])}

Now review this product:
Title: {product_title}
Price: {product_price}

Write a short review (2-3 sentences) that matches your style. Return JSON: {{"rating": (1-5 integer), "review": "text"}}
"""
    response = model.generate_content(prompt)
    # Simple parsing (add error handling)
    import json as jsonlib
    result = jsonlib.loads(response.text)
    return result["review"], result["rating"]

iface = gr.Interface(
    fn=generate_review,
    inputs=[
        gr.Dropdown(choices=list(profiles.keys()), label="Persona Name"),
        gr.Textbox(label="Product Title"),
        gr.Textbox(label="Product Price")
    ],
    outputs=[
        gr.Textbox(label="Generated Review"),
        gr.Number(label="Predicted Rating")
    ],
    title="Task A: User Modeling Agent",
    description="Generate a realistic review and rating for a given user persona."
)

iface.launch()
