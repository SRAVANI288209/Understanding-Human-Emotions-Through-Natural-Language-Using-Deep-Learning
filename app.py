
# ============================================================
# EMOTION DETECTION - STREAMLIT WEB APPLICATION
# ============================================================

import streamlit as st
import numpy as np
import pickle
import json
import re
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="💬",
    layout="centered"
)


# ============================================================
# TEXT PREPROCESSING
# IMPORTANT: Use the same clean_text() function as training.
# Replace this function with your exact notebook version
# if your training preprocessing was different.
# ============================================================

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# LOAD SAVED MODEL AND FILES
# ============================================================

@st.cache_resource
def load_assets():

    model = load_model(
        "best_fasttext_bilstm.keras",
        compile=False
    )

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    with open("label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)

    with open("model_config.json", "r") as file:
        config = json.load(file)

    return model, tokenizer, label_encoder, config


# ============================================================
# EMOTION PREDICTION FUNCTION
# ============================================================

def predict_emotion(sentence, model, tokenizer,
                    label_encoder, max_len):

    cleaned_sentence = clean_text(sentence)

    if not cleaned_sentence:
        return None, None

    sequence = tokenizer.texts_to_sequences(
        [cleaned_sentence]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    probabilities = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    emotion = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    score = float(probabilities[predicted_index])

    return emotion, (score, probabilities)


# ============================================================
# WEB APPLICATION
# ============================================================

st.title("💬 Emotion Detection System")

st.write(
    "Enter a sentence below to predict its emotional category "
    "using our trained FastText + BiLSTM model."
)

st.caption("Supported emotions: Anger, Fear, Joy and Sadness")

try:
    model, tokenizer, label_encoder, config = load_assets()
    max_len = config.get("max_len", 25)

except Exception as error:
    st.error(
        "Unable to load the model or supporting files. "
        "Check that all four saved files are in the project folder."
    )
    st.exception(error)
    st.stop()


# ============================================================
# USER INPUT
# ============================================================

sentence = st.text_area(
    "Enter your sentence",
    placeholder="Example: I am feeling very happy today.",
    height=120
)

predict_button = st.button(
    "🔍 Predict Emotion",
    type="primary",
    use_container_width=True
)


# ============================================================
# DISPLAY PREDICTION
# ============================================================

if predict_button:

    if not sentence.strip():
        st.warning("Please enter a sentence first.")

    else:
        with st.spinner("Analyzing your sentence..."):

            emotion, result = predict_emotion(
                sentence,
                model,
                tokenizer,
                label_encoder,
                max_len
            )

        if emotion is None:
            st.warning(
                "Please enter a sentence containing valid text."
            )

        else:
            score, probabilities = result

            st.divider()
            st.subheader("Prediction Result")

            st.success(f"Predicted Emotion: {emotion.upper()}")

            st.metric(
                label="Model Score",
                value=f"{score * 100:.2f}%"
            )

            st.caption(
                "Model score is the predicted class probability; "
                "it is not a guarantee of correctness."
            )

            # Show probabilities for all emotion classes
            probability_df = pd.DataFrame({
                "Emotion": label_encoder.classes_,
                "Probability (%)": probabilities * 100
            })

            probability_df = probability_df.sort_values(
                by="Probability (%)",
                ascending=False
            )

            st.subheader("Emotion Probability Distribution")

            st.bar_chart(
                probability_df.set_index("Emotion")
            )

            st.dataframe(
                probability_df.round(2),
                use_container_width=True,
                hide_index=True
            )

# ============================================================
# END OF APPLICATION
# ============================================================
