# unspoken_meaning_detector.py
# Run with: streamlit run unspoken_meaning_detector.py

import streamlit as st
import pandas as pd
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------- CONFIG ----------------
CSV_PATH = "unspoken_meaning_dataset_200rows.csv"
MODEL_PATH = "model.joblib"
MEANING_PATH = "meaning_dict.joblib"

# -------------- DATA LOADING ------------
@st.cache_data
def load_and_prepare_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV file not found: {CSV_PATH}")
        st.stop()

    df = pd.read_csv(CSV_PATH)

    # Make sure required columns exist
    required_cols = {"message", "label", "hidden_meaning"}
    if not required_cols.issubset(df.columns):
        st.error("CSV must contain: message, label, hidden_meaning")
        st.stop()

    # Clean text
    df["message"] = df["message"].astype(str).str.lower()
    df["message"] = df["message"].apply(
        lambda x: re.sub(r"[^\w\s]", "", x).strip()
    )

    # Remove empty rows that break training
    df = df.dropna(subset=["message", "label"])
    df = df[df["message"] != ""]

    # Meaning dictionary
    meaning_dict = df.groupby("label")["hidden_meaning"].first().to_dict()

    # Train-test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    return train_df, test_df, meaning_dict

# -------------- MODEL TRAINING ----------
@st.cache_resource
def train_or_load_model(force_retrain=False):

    if not force_retrain and os.path.exists(MODEL_PATH) and os.path.exists(MEANING_PATH):
        model = joblib.load(MODEL_PATH)
        meaning_dict = joblib.load(MEANING_PATH)
        st.success("Loaded saved model.")
        return model, meaning_dict

    st.info("Training new model...")

    train_df, test_df, meaning_dict = load_and_prepare_data()

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            multi_class="auto"
        ))
    ])

    with st.spinner("Training model..."):
        model.fit(train_df["message"], train_df["label"])

    joblib.dump(model, MODEL_PATH)
    joblib.dump(meaning_dict, MEANING_PATH)

    st.success("Training complete and model saved.")
    return model, meaning_dict

# ---------------- STREAMLIT APP ----------------
def main():
    st.title("Unspoken Meaning Detector")
    st.subheader("What people say vs what they really mean.")

    model, meaning_dict = train_or_load_model()

    message = st.text_area(
        "Enter your message:",
        height=120,
        placeholder="It's fine."
    )

    if st.button("Analyze"):
        if not message.strip():
            st.warning("Please enter a message.")
        else:
            clean_text = re.sub(r"[^\w\s]", "", message.lower().strip())

            pred = model.predict([clean_text])[0]
            probs = model.predict_proba([clean_text])[0]
            confidence = max(probs) * 100

            meaning = meaning_dict.get(pred, "Unknown meaning")

            st.write(f"**Original message:** {message}")
            st.write(
                f"**Prediction:** "
                f"{pred.capitalize().replace('_', '-')} "
                f"({confidence:.0f}%)"
            )
            st.write(f"**Real meaning:** {meaning}")

    st.markdown("---")
    st.write("Built with sklearn + Streamlit.")

    if st.button("Force retrain model"):
        train_or_load_model(force_retrain=True)
        st.rerun()

if __name__ == "__main__":
    main()
