from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import streamlit as st
import tensorflow as tf

# Read the content of the text file
with open('Sherlock Holmes.txt', 'r') as f:
    data = f.read()

# Tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

model = tf.keras.models.load_model('nextword.h5')

# Define the function to predict the next word
def predict_next_word(text):
    # Tokenize the text
    sequences = tokenizer.texts_to_sequences([text])
    # Pad the sequences
    padded_sequences = pad_sequences(sequences, maxlen=18)
    # Predict the next word
    predictions = model.predict(padded_sequences)
    # Get the index of the predicted word
    predicted_word_indices = np.argsort(predictions[0])[-3:][::-1]  # Get top 3 indices
    predicted_words = [tokenizer.index_word[index] for index in predicted_word_indices]
    # Return the predicted word
    return predicted_words

# Create a streamlit app with styling
st.title('Next Word Prediction')
st.markdown(
    "This app predicts the next word based on the input text using a pre-trained model."
)

# Get the input text from the user
input_text = st.text_input('Enter a text:')

button_style = f"background: white; padding: 5px 10px; border: none; color: black; border-radius: 50px; cursor: pointer; font-size: 18px;"

if input_text:
    predicted_words = predict_next_word(input_text)

    # Create columns to display predicted words horizontally
    col1, col2, col3 = st.columns(3)  # Create three equally spaced columns

    with col1:
        if len(predicted_words) >= 1:
            st.markdown(f'<div style="{button_style}">{predicted_words[0]}</div>', unsafe_allow_html=True)  # Display the first word with styling

    with col2:
        if len(predicted_words) >= 2:
            st.markdown(f'<div style="{button_style}">{predicted_words[1]}</div>', unsafe_allow_html=True)  # Display the second word with styling

    with col3:
        if len(predicted_words) >= 3:
            st.markdown(f'<div style="{button_style}">{predicted_words[2]}</div>', unsafe_allow_html=True)  # Display the third word with styling
