import os
import json
import librosa
import numpy as np
import tensorflow as tf
import streamlit as st
import base64
from PIL import Image
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_option_menu import option_menu
from warnings import filterwarnings

filterwarnings('ignore')

# -------------------- User-Defined Image Paths --------------------
WELCOME_IMG = r"E:\Bird-audio\welcome.jpg"
BIRD_INFO_IMAGES = [
    r"E:\Bird-audio\image1.jpg",
    r"E:\Bird-audio\images4.jpg",
    r"E:\Bird-audio\img2.jpg_large",
    r"E:\Bird-audio\img3.jpg",
    r"E:\Bird-audio\backgr.jpg",
]
SPECIES_IMAGES = [
    r"E:\Bird-audio\Inference_Images\Andean Guan_sound.jpg",
    r"E:\Bird-audio\Inference_Images\Baudo Guan_sound.jpg",
    r"E:\Bird-audio\Inference_Images\Blue-throated Piping Guan_sound.jpg",
    r"E:\Bird-audio\Inference_Images\Black-capped Tinamou_sound.jpg",
    r"E:\Bird-audio\Inference_Images\Cauca Guan_sound.jpg"
]

# -------------------- Streamlit Config --------------------
def streamlit_config():
    st.set_page_config(page_title='Bird Sound Classification', layout='wide')
    css = """
    <style>
    h1, h2, h3, h4 { color: white; text-shadow: 1px 1px 2px black; }
    .stButton>button { background-color: #8a2be2; color: white; font-weight: bold; }
    .stFileUploader>label, .stFileUploader>div { color: white; font-weight: bold; }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------------- Fullscreen Background --------------------
def set_fullscreen_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, 'rb') as img:
            b64 = base64.b64encode(img.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{b64}");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
            }}
            </style>
            """, unsafe_allow_html=True
        )

# -------------------- Pink Background --------------------
def set_pink_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffc0cb;
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------- Pages --------------------
def welcome_page():
    set_fullscreen_background(r"E:\Bird-audio\HD-wallpaper-birds-bird-flower-nature.jpg")
    st.markdown(
        "<h1 style='text-align:center;'>Welcome to Bird Species Classification Based on Audio</h1>",
        unsafe_allow_html=True
    )
    add_vertical_space(2)
    st.markdown(
        "<h3 style='text-align:center;'>Explore various species and identify birds from their sounds.</h3>",
        unsafe_allow_html=True
    )


def bird_info_page():
    set_pink_background()
    st.markdown("<h2>About Birds</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        Birds are warm-blooded vertebrates characterized by feathers, beaks, and the ability to lay eggs.
        They play key roles in ecosystems and have species-specific vocalizations.
        """,
        unsafe_allow_html=True
    )
    cols = st.columns(5)
    for idx, img_path in enumerate(BIRD_INFO_IMAGES):
        with cols[idx % 5]:
            try:
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading bird image {img_path}: {e}")


def species_info_page():
    set_pink_background()
    st.markdown("<h2>Bird Species Information</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        Here are some species you can classify:
        - Andean Guan
        - Baudo Guan
        - Blue-throated Piping Guan
        - Black-capped Tinamou
        - Cauca Guan
        """,
        unsafe_allow_html=True
    )
    cols = st.columns(5)
    for idx, img_path in enumerate(SPECIES_IMAGES):
        with cols[idx]:
            try:
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading species image {img_path}: {e}")


def audio_classification_page():
    set_pink_background()
    st.markdown("<h2>Upload and Classify Bird Sound</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        input_audio = st.file_uploader('', type=['mp3', 'wav'])
    if input_audio:
        with st.spinner('Processing audio...'):
            with col2:
                predict_audio(input_audio)

# -------------------- Inference --------------------
def predict_audio(audio_file):
    try:
        with open('prediction.json') as f:
            label_map = json.load(f)
        y, sr = librosa.load(audio_file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        feats = np.mean(mfcc, axis=1).reshape(1, -1, 1).astype('float32')
        model = tf.keras.models.load_model('model.h5')
        preds = model.predict(feats)
        idx = np.argmax(preds)
        species = label_map[str(idx)]
        conf = round(float(np.max(preds)) * 100, 2)
        st.markdown(f"<h3 style='text-align:center; color:lightgreen;'>Predicted Bird species is : {species}</h3>", unsafe_allow_html=True)
        #st.markdown(f"<h4 style='text-align:center; color:orange;'>Confidence: {conf}%</h4>", unsafe_allow_html=True)
        st.audio(audio_file, format='audio/wav')

        # ------------------ Display Species Image ------------------
        # Use the species string exactly as given (which includes '_sound')
        image_path = os.path.join(r"E:\Bird-audio\Inference_Images", f"{species}.jpg")
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                st.image(img, caption=species, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display image for {species}: {e}")
        else:
            st.warning(f"No image available for predicted species: {species}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# -------------------- Main --------------------
streamlit_config()

selected = option_menu(
    None,
    ['Welcome', 'Bird Info', 'Species Info', 'Audio Classification'],
    icons=['house', 'book', 'info-circle', 'mic'],
    orientation='horizontal'
)

if selected == 'Welcome':
    welcome_page()
elif selected == 'Bird Info':
    bird_info_page()
elif selected == 'Species Info':
    species_info_page()
else:
    audio_classification_page()
