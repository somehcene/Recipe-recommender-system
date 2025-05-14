import json
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import base64

st.set_page_config(page_title="Semantic Recipe Recommender", layout="wide")
st.title("🍳 Semantic Recipe Recommender")

# Background image
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64("4901583.jpg")

# CSS Styling
st.markdown(
    f"""
    <style>
    html, body {{
        height: 100% !important;
        margin: 0;
        padding: 0;
        overflow: auto !important;
    }}

    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        overflow: auto !important;
    }}

    .stApp::before {{
        pointer-events: none;
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(255,255,255,0.7);
        backdrop-filter: blur(3px);
        z-index: 0;
    }}

    .stApp > * {{
        position: relative;
        z-index: 1;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}

    .recipe-card {{
        background-color: rgba(255, 255, 255, 0.95);
        padding: 1.5em;
        border-radius: 10px;
        box-shadow: 0px 0px 12px rgba(0, 0, 0, 0.25);
        margin-top: 1em;
        margin-bottom: 2em;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load recipe data
@st.cache_data
def load_recipes_df():
    df = pd.read_csv("recipes_df.csv", index_col=0)
    df.drop_duplicates(subset=['title'], inplace=True)
    df['combined_text'] = df['title'] + ' ' + df['ingredients']
    return df

recipes_df = load_recipes_df()

# Compute embeddings
@st.cache_data
def compute_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

embeddings = compute_embeddings(recipes_df['combined_text'].tolist())

# Load full recipe JSON
@st.cache_data
def load_full_recipes():
    with open("db_recipes.json", "r", encoding="utf-8") as f:
        return json.load(f)

db_recipes = load_full_recipes()

# Sidebar
st.sidebar.header("🔍 Navigate recipes")
user_query = st.sidebar.text_input("Describe your craving:", "creamy chicken with mushrooms")
max_time = st.sidebar.slider("⏱ Max Total Time (mins)", min_value=5, max_value=240, value=90)
cuisine_filter = st.sidebar.multiselect(
    "🌍 Cuisine",
    options=sorted({r.get('cuisine') for r in db_recipes.values() if isinstance(r.get('cuisine'), str)})
)

# Search and display results
if st.sidebar.button("Find Recipes"):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(scores, k=10)

    found = False
    for score, idx in zip(top_results[0], top_results[1]):
        recipe_row = recipes_df.iloc[idx.item()]
        recipe_title = recipe_row['title']
        recipe = db_recipes.get(recipe_title)

        if not recipe:
            continue

        if recipe.get('total_time', 9999) > max_time:
            continue

        if cuisine_filter and recipe.get('cuisine') not in cuisine_filter:
            continue

        found = True
        with st.expander(f"{recipe_title} (Score: {score.item():.2f})"):
            st.markdown(
                f"**Cuisine:** {recipe.get('cuisine')} | **Total Time:** {recipe.get('total_time')} mins"
            )
            st.markdown(
                f"**Calories:** {recipe.get('Calories')} | **Protein:** {recipe.get('Protein')}g"
            )

            st.markdown("**Ingredients:**")
            for ing in eval(recipe.get('ingredients', '[]')):
                st.markdown(f"- {ing}")

            st.markdown("**Steps:**")
            for i, step in enumerate(eval(recipe.get('steps', '[]')), 1):
                st.markdown(f"{i}. {step}")

    if not found:
        st.warning("No matching recipes found with your filters.")
