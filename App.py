import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
with open("C:/Users/Vinaykumar/OneDrive/Desktop/DS Projects/Financial Sentiment Analysis/New/model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the vectorizer
with open("C:/Users/Vinaykumar/OneDrive/Desktop/DS Projects/Financial Sentiment Analysis/New/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load the tokenizer
with open("C:/Users/Vinaykumar/OneDrive/Desktop/DS Projects/Financial Sentiment Analysis/New/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

st.title('Financial Sentiment Analysis')

# Create a text box for user input
text = st.text_area("Enter a financial text :", max_chars=1000, height=200)

# Create a button to perform analysis
if st.button('Analyze'):
    # Apply the vectorizer to the user input
    X = vectorizer.transform([text])

    # Convert the numpy array to a string
    text_str = text.tostring().decode('utf-8')

    # Make a prediction using the model
    prediction = model.predict(tokenizer.texts_to_sequences([text_str]))

    # Display the prediction
    if prediction == 1:
        st.success('Positive sentiment')
    elif prediction == 0:
        st.info('Neutral sentiment')
    else:
        st.error('Negative sentiment')

def predict(text):
    # Tokenize the text
    sequence = tokenizer.texts_to_sequences([text])

    # Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=250, dtype='int32', value=0)

    # Make a prediction using the model
    prediction = model.predict(padded_sequence, batch_size=1, verbose=2)[0]
    predicted_label = np.argmax(prediction)

    # Map the predicted label to a sentiment
    sentiments = ["Negative", "Neutral", "Positive"]
    return sentiments[predicted_label]

# # Test the predict function
# text = "This is a great product"
# prediction = predict(text)
# st.write(f"Prediction for '{text}': {prediction}")
