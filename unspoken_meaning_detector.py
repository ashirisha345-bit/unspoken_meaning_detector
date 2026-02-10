# unspoken_meaning_detector.py


import streamlit as st
import pandas as pd
import re
import torch
import pickle
import os
import sys
from sklearn.model_selection import train_test_split

try:
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
except ImportError as e:
    st.error("Cannot import transformers / datasets libraries.\n"
             "Most likely: requirements.txt is missing or transformers failed to install.\n\n"
             "→ Make sure requirements.txt exists in GitHub root\n"
             "→ Try adding: torch --index-url https://download.pytorch.org/whl/cpu\n"
             "→ Then reboot the app")
    st.stop()

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────
CSV_PATH = "unspoken_meaning_dataset_200rows.csv"
MODEL_PATH      = "model.pt"
TOKENIZER_PATH  = "tokenizer.pkl"
ID2LABEL_PATH   = "id_to_label.pkl"
MEANING_PATH    = "meaning_dict.pkl"

MAX_LENGTH = 128
NUM_EPOCHS = 3
BATCH_SIZE = 8

# ────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ────────────────────────────────────────────────

@st.cache_data
def load_and_prepare_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"CSV file not found: {CSV_PATH}\nMake sure the file is in the GitHub repo root.")
        st.stop()

    df = pd.read_csv(CSV_PATH)
    df['message'] = df['message'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower().strip()))

    unique_labels = sorted(df['label'].unique())
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}

    df['label_id'] = df['label'].map(label_to_id)
    meaning_dict = df.groupby('label')['hidden_meaning'].first().to_dict()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label_id']
    )

    return train_df, test_df, label_to_id, id_to_label, meaning_dict, len(unique_labels)


@st.cache_resource
def get_tokenizer():
    return DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['message'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )


@st.cache_resource
def train_or_load_model(_num_labels, force_retrain=False):
    tokenizer = get_tokenizer()

    paths = [MODEL_PATH, TOKENIZER_PATH, ID2LABEL_PATH, MEANING_PATH]
    if not force_retrain and all(os.path.exists(p) for p in paths):
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=_num_labels
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        with open(ID2LABEL_PATH, 'rb') as f:
            id_to_label = pickle.load(f)
        with open(MEANING_PATH, 'rb') as f:
            meaning_dict = pickle.load(f)
        st.success("Loaded saved model files.")
        return model, tokenizer, id_to_label, meaning_dict

    st.info("No saved model found → training new model (may take 3–12 min on CPU)...")
    train_df, test_df, _, id_to_label, meaning_dict, num_labels = load_and_prepare_data()

    train_dataset = Dataset.from_pandas(train_df[['message', 'label_id']])
    test_dataset  = Dataset.from_pandas(test_df[['message', 'label_id']])

    tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_test  = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir='./temp_results',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="epoch",              # ← CHANGED HERE: was evaluation_strategy
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    with st.spinner("Training model... please wait"):
        trainer.train()

    torch.save(model.state_dict(), MODEL_PATH)
    with open(TOKENIZER_PATH,   'wb') as f: pickle.dump(tokenizer,   f)
    with open(ID2LABEL_PATH,    'wb') as f: pickle.dump(id_to_label, f)
    with open(MEANING_PATH,     'wb') as f: pickle.dump(meaning_dict, f)

    st.success("Training done → model saved!")
    return model, tokenizer, id_to_label, meaning_dict


def main():
    st.title("Unspoken Meaning Detector")
    st.subheader("What people say vs. what they really mean.")

    train_df, _, _, id_to_label, meaning_dict, num_labels = load_and_prepare_data()
    model, tokenizer, id_to_label, meaning_dict = train_or_load_model(num_labels)

    model.eval()

    message = st.text_area("Enter a message:", height=120, placeholder="It's fine...")

    if st.button("Analyze", type="primary"):
        if not message.strip():
            st.warning("Please enter a message.")
            return

        with st.spinner("Analyzing..."):
            clean_text = re.sub(r'[^\w\s]', '', message.lower().strip())
            inputs = tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1)
            pred_idx = probs.argmax().item()
            confidence = probs[0, pred_idx].item() * 100

            label = id_to_label[pred_idx]
            hidden_meaning = meaning_dict.get(label, "—")

            st.markdown("### Result")
            st.info(f"**Original:** {message}")
            st.subheader(f"{label.replace('_', ' ').title()} ({confidence:.0f}%)")
            st.markdown(f"**Real meaning:** {hidden_meaning}")

    st.markdown("---")
    st.caption("Built with DistilBERT + Streamlit • Small 200-row dataset")

    if st.button("Force retrain (slow)", type="secondary"):
        st.cache_resource.clear()
        st.rerun()


if __name__ == "__main__":
    main()
