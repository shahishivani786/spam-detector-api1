from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model and vectorizer
model = joblib.load('spam_detection_model (1).pkl')  # Path to your trained model
vectorizer = joblib.load('tfidf_vectorizer (1).pkl')  # Path to your vectorizer

# Initialize FastAPI app
app = FastAPI()

# Define the request body for the email text
class EmailRequest(BaseModel):
    email_text: str

# Define the prediction endpoint
@app.post("/predict/")
def predict(request: EmailRequest):
    # Preprocess the email text using the vectorizer
    email_vector = vectorizer.transform([request.email_text])
    
    # Make prediction
    prediction = model.predict(email_vector)
    
    # Return the result as "Spam" or "Not Spam"
    result = "Spam" if prediction == 1 else "Not Spam"
    return {"prediction": result}
