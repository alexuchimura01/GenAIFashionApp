import os
import json
import base64
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import boto3
from typing import Optional
from PIL import Image

# --- Load environment variables ---
load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv('aws_access_key_id')
AWS_SECRET_ACCESS_KEY = os.getenv('aws_secret_access_key')

# --- Bedrock client setup ---
def get_bedrock_client(runtime=True, aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None):
    service_name = 'bedrock-runtime' if runtime else 'bedrock'
    return boto3.client(
        service_name=service_name,
        region_name="us-west-2",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )

bedrock_runtime = get_bedrock_client()

# --- Claude and Cohere embedding ---
MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
ACCEPT = "application/json"
CONTENT_TYPE = "application/json"

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def query_claude_with_images(image_paths: list, prompt: str) -> str:
    images = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": encode_image_base64(path)
            }
        } for path in image_paths
    ]

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": images + [{"type": "text", "text": prompt}]
            }
        ],
        "max_tokens": 1000
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType=CONTENT_TYPE,
            accept=ACCEPT,
            body=json.dumps(body)
        )
        result = json.loads(response['body'].read())
        if isinstance(result["content"], list):
            return "\n".join([chunk.get("text", "") for chunk in result["content"]])
        return result["content"]
    except Exception as e:
        return f"ERROR: {e}"

def embed_query_with_cohere(text):
    if not text.strip():
        return np.zeros(1024).tolist()
    body = {"texts": [text], "input_type": "search_document"}
    try:
        response = bedrock_runtime.invoke_model(
            modelId="cohere.embed-english-v3",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        result = json.loads(response['body'].read())
        return result["embeddings"][0]
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return np.zeros(1024).tolist()

@st.cache_data
def load_data():
    df = pd.read_csv("cohere_vector_embeddings_combined.csv")
    for col in ["vector1", "vector2", "vector3", "vector4"]:
        df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else np.zeros(1024).tolist())
    return df

def load_celeb_embeddings():
    df_celeb = pd.read_excel("C:/Users/petew/OneDrive/Desktop/Cal Poly/MSBA/GSB-570-(Gen-AI)/Code/Final Project/Celeb_Comp_Embeddings.xlsx")
    df_celeb["CohereEmbedding"] = df_celeb["CohereEmbedding"].apply(lambda x: json.loads(x) if isinstance(x, str) else np.zeros(1024).tolist())
    return df_celeb

def compute_similarity(query_vector, row):
    return max([
        cosine_similarity([query_vector], [np.array(row[f"vector{i}"])])[0][0]
        for i in range(1, 5)
    ])

st.title("🧥 AI Fashion Recommender")

if st.button("🔄 Restart Search"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if "paired_item" not in st.session_state:
    user_query = st.text_input("What kind of clothing are you looking for?", key="user_query")
else:
    user_query = st.session_state.get("initial_prompt", "")

if "initial_prompt" not in st.session_state and user_query:
    st.session_state.initial_prompt = user_query

if "selection_pending" in st.session_state:
    df = load_data()
    idx = st.session_state.selection_pending
    query_vec = embed_query_with_cohere(st.session_state.initial_prompt)
    df["similarity_score"] = df.apply(lambda row: compute_similarity(query_vec, row), axis=1)
    top_matches = df.sort_values(by="similarity_score", ascending=False).head(5)

    match_options = [
        (i, row, os.path.join("ClothingItemImages", row["ImageFilename"]))
        for i, (_, row) in enumerate(top_matches.iterrows())
        if os.path.exists(os.path.join("ClothingItemImages", row["ImageFilename"]))
    ]
    if idx < len(match_options):
        _, row, _ = match_options[idx]
        st.session_state.selected_item = row.to_dict()
        st.session_state.selection_mode = "outfit"
        del st.session_state.selection_pending
        st.rerun()

if st.session_state.get("selection_mode") == "outfit" and "selected_item" in st.session_state:
    df = load_data()
    selected = st.session_state.selected_item

    if "paired_item" not in st.session_state:
        st.markdown("### 👕 Selected Item")
        image_path = os.path.join("ClothingItemImages", selected["ImageFilename"])
        if os.path.exists(image_path):
            st.image(Image.open(image_path), width=300)

        st.markdown("### 🧹 Complete the outfit")
        outfit_query = st.text_input("Describe what you'd like to pair with this item", key="outfit_query")

        if outfit_query:
            vec = embed_query_with_cohere(outfit_query)
            df["similarity_score"] = df.progress_apply(lambda row: compute_similarity(vec, row), axis=1)

            if selected.get("Category", "").lower() == "top":
                df_filtered = df[df["Category"].str.lower() == "bottom"]
            elif selected.get("Category", "").lower() == "bottom":
                df_filtered = df[df["Category"].str.lower() == "top"]
            else:
                df_filtered = df.copy()

            top_matches = df_filtered.sort_values(by="similarity_score", ascending=False).head(5)

            st.markdown("### 👖 Top Matches to Complete Outfit")
            for idx, (_, row) in enumerate(top_matches.iterrows()):
                image_path = os.path.join("ClothingItemImages", row["ImageFilename"])
                if os.path.exists(image_path):
                    if st.button(f"🧹 Select to Pair", key=f"pair_{idx}"):
                        st.session_state.paired_item = row.to_dict()
                        st.rerun()
                    st.image(Image.open(image_path), width=250, caption=f"Score: {row['similarity_score']:.2f}")

    if "paired_item" in st.session_state:
        paired = st.session_state.paired_item
        selected = st.session_state.selected_item

        selected_cat = selected.get("Category", "").lower()
        paired_cat = paired.get("Category", "").lower()

        if selected_cat == "top" or paired_cat == "bottom":
            top, bottom = selected, paired
        elif paired_cat == "top" or selected_cat == "bottom":
            top, bottom = paired, selected
        else:
            top, bottom = selected, paired  # fallback

        st.markdown("### 👔 Final Outfit")
        top_path = os.path.join("ClothingItemImages", top["ImageFilename"])
        bottom_path = os.path.join("ClothingItemImages", bottom["ImageFilename"])
        if os.path.exists(top_path):
            st.image(Image.open(top_path), width=300)
        if os.path.exists(bottom_path):
            st.image(Image.open(bottom_path), width=300)

        prompt = ("Using the pictures of this shirt and pant/short (top/bottom) combination, describe the fashion style using "
                  "concise fashion terminology. Include outfit elements, materials, colors, fit, vibe, and noticeable aesthetic choices.")

        with st.spinner("🧠 Describing outfit with Claude..."):
            description = query_claude_with_images([top_path, bottom_path], prompt)
        st.markdown("### 📝 Claude's Outfit Description")
        st.write(description)

        with st.spinner("📈 Embedding outfit description..."):
            embedding = embed_query_with_cohere(description)

        celeb_df = load_celeb_embeddings()
        celeb_df["similarity"] = celeb_df["CohereEmbedding"].apply(
            lambda x: cosine_similarity([embedding], [x])[0][0]
        )
        top_celebs = celeb_df.sort_values(by="similarity", ascending=False).head(5)

        st.markdown("### 🌟 Most Similar Celebrity Outfits")

        top_match = top_celebs.iloc[0]
        celeb_img_path = os.path.join(
            "C:/Users/petew/OneDrive/Desktop/Cal Poly/MSBA/GSB-570-(Gen-AI)/Code/Final Project/CelebPics",
            top_match["Image_FileName"]
        )
        if os.path.exists(celeb_img_path):
            st.image(Image.open(celeb_img_path), width=300)

        celeb_name = top_match.get("Name", "Unknown")
        archetype = top_match.get("Archetype", "N/A")
        st.markdown(f"**Your celebrity match is {celeb_name}, characterized as _{archetype}_**.")

        st.stop()

elif user_query:
    df = load_data()
    query_vec = embed_query_with_cohere(user_query)
    with st.spinner("Embedding and comparing with catalog..."):
        tqdm.pandas()
        df["similarity_score"] = df.progress_apply(lambda row: compute_similarity(query_vec, row), axis=1)
        top_matches = df.sort_values(by="similarity_score", ascending=False).head(5)

    st.markdown("### 👕 Top Matches")
    match_options = [
        (idx, row, os.path.join("ClothingItemImages", row["ImageFilename"]))
        for idx, (_, row) in enumerate(top_matches.iterrows())
        if os.path.exists(os.path.join("ClothingItemImages", row["ImageFilename"]))
    ]

    for idx, row, image_path in match_options:
        if st.button(f"👕 Select this item", key=f"select_{idx}"):
            st.session_state.selection_pending = idx
            st.rerun()
        st.image(Image.open(image_path), width=250, caption=f"Score: {row['similarity_score']:.2f}")
