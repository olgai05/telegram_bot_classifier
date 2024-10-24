import streamlit as st
import torch
import joblib
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Load tokenizer, model, and trained classifier
@st.cache_resource
def load_resources():
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    bert_model = AutoModel.from_pretrained("cointegrated/rubert-tiny2").to('cuda' if torch.cuda.is_available() else 'cpu')
    clf = joblib.load('logistic_regression_model2.pkl')  # Trained LogisticRegression model
    le = joblib.load('label_encoder.pkl')  # Trained LabelEncoder
    return tokenizer, bert_model, clf, le

# Function to embed the message using BERT
def embed_message(message, tokenizer, bert_model):
    inputs = tokenizer(message, padding=True, truncation=True, max_length=64, return_tensors="pt").to(bert_model.device)
    with torch.no_grad():
        output = bert_model(**inputs)
    cls_token = output.last_hidden_state[:, 0, :]  # CLS token embedding
    return torch.nn.functional.normalize(cls_token)[0].cpu().numpy()

# Function to predict label
def predict_label(message, tokenizer, bert_model, clf, le):
    # Get the embedding of the message
    features = embed_message(message, tokenizer, bert_model).reshape(1, -1)
    
    # Predict using the trained classifier
    pred_numeric = clf.predict(features)
    
    # Convert prediction back to original label
    pred_label = le.inverse_transform(pred_numeric)
    return pred_label[0]

# Streamlit App
st.title("Message Label Prediction App")

# Input text box for message
user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    if user_input:
        # Load resources
        tokenizer, bert_model, clf, le = load_resources()

        # Predict label
        prediction = predict_label(user_input, tokenizer, bert_model, clf, le)

        # Display the result
        st.write(f"Predicted Label: **{prediction}**")
    else:
        st.write("Please enter a message to classify.")
