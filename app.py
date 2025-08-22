# app.py
# This script will serve as the main entry point for our Streamlit web application.
# Its purpose is to create a user interface where users can upload an audio
# file and see the predicted music genre from our trained CNN model.

# --- Import Necessary Libraries --- 
#For building the web app user interface import streamlit as st 
import streamlit as st
#For loading and using our trained Keras model import tensorflow as tf 
import tensorflow as tf
# For numerical operations and data manipulation
import numpy as np
#For audio processing (loading files, extracting features)
import librosa
#For loading the scaler object
import joblib

#-----Helper Function-------

# We use this decorator to cache the model and scaler loading process.
# This means the model and scaler are loaded only once when the app starts,
# significantly speeding up subsequent predictions.
@st.cache_data
def load_model_and_scaler():
    """
    Loads the pre-trained Keras CNN model and the StandardScaler object.
    The @st.cache_data decorator ensures this function is run only once.
    """
    try:
        # Load the Keras model. compile=False is a performance optimization for inference.
        model = tf.keras.models.load_model('music_genre_cnn.h5', compile=False)
        
        # Load the scaler object from the file.
        scaler = joblib.load('scaler.joblib')
        
        # Define the genre mapping based on the training labels.
        genre_mapping = {
            0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
            5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
        }
        
        return model, scaler, genre_mapping
    except FileNotFoundError as e:
        # If a file is not found, display an error in the app and stop execution.
        st.error(f"Error loading model or scaler file: {e}. Please ensure the files are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model/scaler: {e}")
        st.stop()


def extract_features(audio_file, sample_rate=22050, n_mfcc=13, n_chroma=12):
    """
    Extracts a feature vector from a single audio file.
    
    Args:
        audio_file: The file-like object from st.file_uploader.
        sample_rate (int): The sample rate for audio processing.
        n_mfcc (int): The number of MFCC coefficients to extract.
        n_chroma (int): The number of chroma bins to extract.
        
    Returns:
        np.array: A 1D NumPy array containing the 28 extracted features.
    """
    try:
        # Load the audio data from the file-like object.
        # Librosa can handle file objects directly.
        y, sr = librosa.load(audio_file, sr=sample_rate, duration=30)
        
        # --- Extract Features ---
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        chroma_mean = np.mean(chroma, axis=1)
        
        # Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = np.mean(spec_cent)
        
        # Spectral Rolloff
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_roll_mean = np.mean(spec_roll)
        
        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        # Concatenate all features into a single vector
        features = np.concatenate([
            mfccs_mean,
            chroma_mean,
            np.array([spec_cent_mean]),
            np.array([spec_roll_mean]),
            np.array([zcr_mean])
        ])
        
        return features

    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None
    

def predict_genre(audio_file):
    """
    The main prediction pipeline. It takes an audio file, processes it,
    and returns the predicted genre.

    Args:
        audio_file: The file-like object from st.file_uploader.

    Returns:
        str: The predicted genre name or an error message.
    """
    # Step 1: Load the pre-trained model, scaler, and genre mapping.
    # This is efficient due to Streamlit's caching.
    model, scaler, genre_mapping = load_model_and_scaler()

    # Step 2: Extract features from the uploaded audio file.
    features = extract_features(audio_file)

    # Step 3: Handle the case where feature extraction fails.
    if features is None:
        return "Error: Could not process audio file. Please try a different file."

    # Step 4: Scale the features.
    # The scaler expects a 2D array of shape (n_samples, n_features).
    # Our `features` variable is a 1D array of shape (28,), so we reshape it.
    try:
        features_reshaped = features.reshape(1, -1)
        features_scaled = scaler.transform(features_reshaped)
    except Exception as e:
        st.error(f"Error during feature scaling: {e}")
        return "Error: Feature scaling failed."

    # Step 5: Prepare the data for the CNN model.
    # The CNN expects a 3D array: (num_samples, num_timesteps, num_features/channels).
    # We add a new dimension to our scaled features array.
    # Shape changes from (1, 28) to (1, 28, 1).
    features_cnn = np.expand_dims(features_scaled, axis=-1)

    # Step 6: Make a prediction using the loaded model.
    # The model.predict() method returns an array of class probabilities.
    try:
        prediction_probs = model.predict(features_cnn)
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return "Error: Model prediction failed."

    # Step 7: Find the class with the highest probability.
    # np.argmax() returns the index of the maximum value in the array.
    predicted_index = np.argmax(prediction_probs)

    # Step 8: Map the predicted index to the genre name.
    predicted_genre = genre_mapping.get(predicted_index, "Unknown Genre")

    return predicted_genre

#------The Main Application logic-------

def main():
    st.set_page_config(
        page_title="🎵 Music Genre Classifier",
        page_icon="🎶",
        layout="centered"
    )

    # Custom CSS for background and fonts
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
        }
        .stButton>button {
            background-color: #6366f1;
            color: white;
            font-weight: bold;
        }
        .stSpinner {
            color: #6366f1 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("🎵 Music Genre Classification App")
    st.markdown(
        """
        <h3 style='color:#6366f1;'>Welcome!</h3>
        <p>This application uses a <b>Convolutional Neural Network (CNN)</b> to predict the genre of a music track.</p>
        <ul>
            <li>Upload a short <b>.wav</b> audio file</li>
            <li>Listen to your track</li>
            <li>Get instant genre prediction!</li>
        </ul>
        """, unsafe_allow_html=True
    )

    st.sidebar.image(
        "https://cdn.pixabay.com/photo/2016/11/29/09/32/music-1867128_1280.jpg",
        use_container_width=True
    )
    st.sidebar.header("About")
    st.sidebar.info(
        "This demo uses deep learning to classify music genres. "
        "Built with Streamlit, TensorFlow, and Librosa."
    )

    uploaded_file = st.file_uploader("🎤 Drag and drop your audio file here", type=['wav'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        st.divider()
        with st.spinner("🔎 Classifying your track..."):
            predicted_genre = predict_genre(uploaded_file)
        st.success("✅ Prediction complete!")
        st.subheader("🎼 Prediction Result")
        st.markdown(
            f"<h2 style='color:#6366f1;text-align:center;'>{predicted_genre.capitalize()}</h2>",
            unsafe_allow_html=True
        )

if __name__ == '__main__':
    main()