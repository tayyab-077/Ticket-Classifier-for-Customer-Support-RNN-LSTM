import os
import warnings

# Disable GPU and all TensorFlow info logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=all, 1=info, 2=warnings, 3=errors
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimization logs

# Suppress Python warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*compile_metrics.*")

# TensorFlow logging fix (no deprecation warning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Now import everything else
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import numpy as np
import re, string, pickle


# Load resources
@st.cache_resource
def load_resources():
    model = load_model("ticket_classifier.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    with open("category_mapping.pkl", "rb") as f:
        category_mapping = pickle.load(f)
    return model, tokenizer, category_mapping

model, tokenizer, category_mapping  = load_resources()

# Clean text function
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\\S+|@\\w+|#\\w+", "", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    return t

# Streamlit UI
st.set_page_config(page_title="Ticket Classifier", page_icon="🎫")
st.title("🎫 Automated Ticket Classifier")
st.write("Paste a ticket description and see its predicted category.")

ticket = st.text_area("Ticket Description", height=150)

if st.button("Predict"):
    if ticket.strip():
        cleaned = clean_text(ticket)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=50)
        probs = model.predict(padded)
        idx = np.argmax(probs, axis=1)[0]
        # label = df['Topic_group'].cat.categories[idx]
        label = category_mapping[idx]

        confidence = probs[0][idx] * 100
        st.success(f"**Prediction:** {label}")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter a ticket description.")
