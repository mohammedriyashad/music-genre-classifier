# 🎵 Music Genre Classification

https://music-genre-classifier-37otycjsvhmxcgfkfsheha.streamlit.app/
A deep learning web application built with Streamlit that classifies music tracks into one of ten genres. Upload a `.wav` file and see the model predict its genre in real-time.

---

![App Screenshot](https://github.com/user-attachments/assets/a3d729ef-9c00-4f20-a075-58e5fddd8603)


## 📚 Project Overview

This project is an end-to-end music genre classification system. It takes raw audio files from the GTZAN dataset, processes them to extract key audio features, and uses these features to train and evaluate multiple machine learning models. The best-performing model, a Convolutional Neural Network (CNN), is then deployed as an interactive web application using Streamlit and containerized with Docker for portability and easy deployment.

### Key Features:

*   **Feature Extraction:** Processes audio files to extract 28 features, including MFCCs, Chroma, Spectral Centroid, and more.
*   **Model Training:** Implements and trains four different models: Logistic Regression, SVM, Random Forest, and a CNN.
*   **Model Evaluation:** Compares models using detailed classification reports and confusion matrices to select the best performer.
*   **Interactive Web App:** A user-friendly interface built with Streamlit that allows users to upload their own `.wav` files for classification.
*   **Containerized Deployment:** The entire application is containerized using Docker, making it portable and easy to deploy.

## 🛠️ Tech Stack

*   **Python:** The core programming language.
*   **Librosa:** For audio processing and feature extraction.
*   **Scikit-learn:** For traditional machine learning models and data preprocessing.
*   **TensorFlow/Keras:** For building and training the Convolutional Neural Network.
*   **Pandas & NumPy:** For data manipulation and numerical operations.
*   **Streamlit:** For building the interactive web application interface.
*   **Docker:** For containerizing the application for deployment.
*   **GitHub:** For version control and hosting the source code.
*   **Streamlit Community Cloud:** For hosting the live application.

## 🚀 Setup and Local Installation

To run this project on your local machine, please follow the steps below.

### 1. Clone the Repository

```bash
git clone https://github.com/[YOUR_GITHUB_USERNAME]/music-genre-classifier.git
cd music-genre-classifier

📂 Project Structure
├── app.py                   # Main Streamlit application script
├── Dockerfile               # Instructions for building the Docker container
├── requirements.txt         # List of Python dependencies
├── music_genre_cnn.h5       # The trained CNN model
├── scaler.joblib            # The fitted StandardScaler
└── README.md                # You are here!
