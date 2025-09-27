import streamlit as st
import numpy as np
import re, string
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

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
