import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = load_model('emotion_recognition_model.h5')

# Load or define your label encoder
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
label_encoder = LabelEncoder()
label_encoder.fit(emotions)

# Color mapping for emotions
emotion_colors = {
    'neutral': 'gray',
    'calm': '#ADD8E6',  # light blue
    'happy': 'yellow',
    'sad': 'blue',
    'angry': 'red',
    'fearful': 'purple',
    'disgust': 'green',
    'surprised': 'orange'
}

# Constants
SAMPLE_RATE = 22050

# Function to extract features
def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

# Function to predict emotion
def predict_emotion(audio):
    feature = extract_features(audio)
    feature = np.expand_dims(np.expand_dims(feature, axis=0), axis=2)  # Reshape for model
    predictions = model.predict(feature)
    emotion = label_encoder.inverse_transform([np.argmax(predictions)])
    accuracy = np.max(predictions)
    return emotion[0], accuracy

st.title('Real-time Speech Emotion Recognition')

# File uploader widget
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Read audio file
    audio, _ = librosa.load(uploaded_file, sr=SAMPLE_RATE)

    # Predict emotion
    prediction, accuracy = predict_emotion(audio)

    # Display the prediction with color and larger text
    color = emotion_colors.get(prediction, 'black')  # Default to black if no color is found
    st.markdown(f"<h1 style='color:{color}; font-size:30px;'>{prediction}</h1>", unsafe_allow_html=True)
    st.write(f"Accuracy: {accuracy * 100:.2f}%")
