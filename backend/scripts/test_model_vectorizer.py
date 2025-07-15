import pickle
import os

MODEL_PATH = '../data/credit_card_model.pkl'
VECTORIZER_PATH = '../data/credit_card_vectorizer.pkl'

# Load the vectorizer
if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}")
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

# Load the model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Example input (should match the format used during training)
title = "Fullz with CVV and SSN"
description = "Includes full credit card details and social security number"
category = "Cards and CVV"

# Combine text features as done in training
text = f"{title} {description} {category}"

# Transform the input text to the same feature space as training
X = vectorizer.transform([text])

# Predict the risk level
predicted_risk = model.predict(X)[0]
print("Predicted risk level:", predicted_risk)

# (Optional) Get prediction probabilities
if hasattr(model, 'predict_proba'):
    probs = model.predict_proba(X)[0]
    print("Probabilities:", dict(zip(model.classes_, probs))) 