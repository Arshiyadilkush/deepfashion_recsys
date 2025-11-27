# app/ui.py
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import urllib.parse

API = "http://127.0.0.1:8000"

# PAGE SETUP 
st.set_page_config(
    page_title="DeepFashion AI Stylist",
    page_icon="🖤",
    layout="wide"
)

# CUSTOM STYLE 
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="st-"] {
            background-color: #0e1117;
            color: #f5f5f5;
            font-family: 'Poppins', sans-serif;
        }
        .stApp {
            background-color: #0e1117;
        }
        h1, h2, h3 {
            color: #ffffff;
        }
        .button-style {
            background-color: #E07A5F;
            color: white;
            border-radius: 10px;
            padding: 0.7rem 1.5rem;
            text-align: center;
            font-weight: 600;
            text-decoration: none;
            display: inline-block;
        }
        .button-style:hover {
            background-color: #FF9F7E;
        }
        .hero {
            text-align: center;
            margin-top: 4rem;
            margin-bottom: 3rem;
        }
        .hero h1 {
            font-size: 3rem;
        }
        .hero p {
            color: #cccccc;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# HERO SECTION 
st.markdown("""
<div class="hero">
    <h1>DeepFashion AI Outfit Stylist 🖤</h1>
    <p>Elevate your wardrobe with AI-powered recommendations.<br>
    The perfect blend of fashion and technology — effortlessly stylish.</p>
    <a href="#upload-section" class="button-style">✨ Get Started Now</a>
</div>
""", unsafe_allow_html=True)

# UPLOAD SECTION 
st.markdown("<div id='upload-section'></div>", unsafe_allow_html=True)
st.subheader("📸 Upload your fashion image")

uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # show uploaded image (convert to PIL to guarantee format)
    img = Image.open(uploaded_file).convert("RGB")
    st.write("Analyzing your style...")

# PREPARE request properly: (filename, content, content_type)
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        with st.spinner("Contacting API..."):
            response = requests.post(f"{API}/search", files=files, params={"topk": 5}, timeout=30)

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if not results:
                st.info("No similar items found.")
            else:
                st.subheader("💫 AI Outfit Recommendations")
                cols = st.columns(5)
                for i, item in enumerate(results):
                    with cols[i % 5]:
                        # request image bytes from API and show them
                        idx = item.get("index")
                        if idx is None:
                            st.text("no index")
                        else:
                            try:
                                r = requests.get(f"http://127.0.0.1:8000/thumb/{idx}", timeout=10)
                                if r.status_code == 200:
                                    img = Image.open(BytesIO(r.content)).convert("RGB")
                                    st.image(img, width="stretch")
                                else:
                                    st.text("thumbnail unavailable")
                            except Exception:
                                st.text("error loading thumbnail")
                        st.caption(f"{item.get('item_id','unknown')}\nscore: {item.get('score',0):.3f}")
        else:
            st.error(f"Server returned {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as re:
        st.error(f"Network / timeout error calling API: {re}")
    except Exception as e:
        st.error(f"Error: {e}")

