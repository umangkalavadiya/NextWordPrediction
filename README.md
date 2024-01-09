# Next Word Prediction App
This GitHub repository contains a simple Streamlit web application for predicting the next word in a given text. The prediction is based on a pre-trained model using TensorFlow and Keras.
https://nextwordprediction.streamlit.app/
 # Usage
Clone the repository:

```bash
git clone https://github.com/your-username/next-word-prediction.git
```
cd next-word-prediction
Install the required dependencies:

```bash
pip install -r requirements.txt
```
Download the pre-trained model (nextword.h5) and the input text file (Sherlock Holmes.txt) and place them in the project directory.

Run the Streamlit app:

```bash
streamlit run app.py
```
Open the provided URL in your web browser and enter a text in the input box. The app will predict and display the top three next words.

# Dependencies
TensorFlow
Keras
NumPy
Streamlit

# Model Loading
The app loads a pre-trained model (nextword.h5) and tokenizes the input text using Keras's Tokenizer. The model predicts the next word based on the provided input.

# App Interface
The Streamlit app provides a user-friendly interface with a title, a description of its functionality, and an input box for entering text. Predicted words are displayed in three equally spaced columns with a stylish button-like appearance.

Feel free to explore, modify, and integrate this app into your projects! If you have any questions or suggestions, please open an issue or contribute to the repository.




