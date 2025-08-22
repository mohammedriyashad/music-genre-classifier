# Import necessary libraries
import os
#import json
import librosa
import numpy as np
import pandas as pd

# Define the path to the dataset
DATASET_PATH = "genres_original"

# Define the path where the processed data (features) will be saved
#JSON_PATH = "data.json"
CSV_PATH="features.csv"

# Define constants for audio processing
SAMPLE_RATE = 22050
TRACK_DURATION_SECONDS = 30
NUM_SEGMENTS = 10

# Define constants for MFCC extraction
NUM_MFCC = 13       # Number of MFCC coefficients to extract
N_FFT = 2048        # Window size for FFT
HOP_LENGTH = 512    # Number of samples to slide the FFT window
# ----------------------------------------

# Calculate the number of audio samples we expect per 30-second track
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION_SECONDS

def process_dataset(dataset_path, csv_path):
    """
    The main function to extract features from the dataset and save them to a JSON file.
    ...
    """
    
    data = {
        "mapping": [],
        "labels": [],
        "features": []
    }

    print("Starting feature extraction...")

    for i, genre_folder in enumerate(sorted(os.listdir(dataset_path))):
        genre_path = os.path.join(dataset_path, genre_folder)

        if os.path.isdir(genre_path):
            data["mapping"].append(genre_folder)
            print(f"\nProcessing genre: {genre_folder}")

            for filename in sorted(os.listdir(genre_path)):
                if filename.endswith(".wav"):
                    file_path = os.path.join(genre_path, filename)

                    
                    try:
                                  # Load the audio file
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                        
                        # --- THIS IS THE NEW CODE BLOCK TO ADD ---

                        # We'll add a simple check to ensure the loaded signal is long enough
                        # to be split into our desired number of segments.
                        # SAMPLES_PER_TRACK was calculated as SAMPLE_RATE * TRACK_DURATION_SECONDS
                        if len(signal) >= SAMPLES_PER_TRACK:
                            
                            # Calculate the number of samples per segment
                            num_samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)

                            for s in range(NUM_SEGMENTS):
                                start_sample = s * num_samples_per_segment
                                end_sample = start_sample + num_samples_per_segment
                                segment = signal[start_sample:end_sample]

                                # Extract MFCCs from the segment.
                                mfccs = librosa.feature.mfcc(y=segment, 
                                                             sr=sr, 
                                                             n_mfcc=NUM_MFCC, 
                                                             n_fft=N_FFT, 
                                                             hop_length=HOP_LENGTH)
                                mfccs_processed = np.mean(mfccs, axis=1)
                                # Extract Chroma Features from the segment.
                                # The result 'chroma' is a 2D NumPy array of shape (12, num_frames).
                                chroma = librosa.feature.chroma_stft(y=segment,
                                                                     sr=sr,
                                                                     n_fft=N_FFT,
                                                                     hop_length=HOP_LENGTH)
                                
                                # Aggregate the chroma features over time by taking the mean.
                                # The result 'chroma_processed' is a 1D array of shape (12,).
                                chroma_processed = np.mean(chroma, axis=1)
                                # In later steps, we will combine mfccs_processed and chroma_processed
                                # into a single feature vector.
                                # Extract Spectral Centroid.
                                # The result 'spectral_centroid' is a 1D array with one value per frame.
                                spectral_centroid = librosa.feature.spectral_centroid(y=segment,
                                                                                      sr=sr,
                                                                                      n_fft=N_FFT,
                                                                                      hop_length=HOP_LENGTH)
                                
                                # Aggregate the spectral centroid over time by taking the mean.
                                # The result 'spectral_centroid_processed' is a single float value.
                                spectral_centroid_processed = np.mean(spectral_centroid)

                                # --- THIS IS THE NEW CODE BLOCK TO ADD ---
                                # Extract Spectral Rolloff.
                                # The result 'spectral_rolloff' is a 1D array with one value per frame.
                                spectral_rolloff = librosa.feature.spectral_rolloff(y=segment,
                                                                                    sr=sr,
                                                                                    n_fft=N_FFT,
                                                                                    hop_length=HOP_LENGTH)
                                
                                # Aggregate the spectral rolloff over time by taking the mean.
                                # The result 'spectral_rolloff_processed' is a single float value.
                                spectral_rolloff_processed = np.mean(spectral_rolloff)
                                # Extract Zero-Crossing Rate.
                                # The result 'zcr' is a 1D array with one value per frame.
                                zcr = librosa.feature.zero_crossing_rate(y=segment,
                                                                         hop_length=HOP_LENGTH)
                                
                                # Aggregate the ZCR over time by taking the mean.
                                # The result 'zcr_processed' is a single float value.
                                zcr_processed = np.mean(zcr)
                                # Combine all features into a single feature vector.
                                # np.hstack is a NumPy function that stacks arrays horizontally.
                                feature_vector = np.hstack((mfccs_processed, 
                                                             chroma_processed, 
                                                             spectral_centroid_processed, 
                                                             spectral_rolloff_processed, 
                                                             zcr_processed))
                                
                                # Store the feature vector and its corresponding label
                                data["features"].append(feature_vector.tolist())
                                data["labels"].append(i) # 'i' is the genre index from the outer loop
                        

                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
                        # If a file can't be loaded, we print an error and continue to the next file.
                        continue
    # Convert the lists into a pandas DataFrame
    print("\nConverting data to pandas DataFrame...")
    
    # Create a DataFrame from the feature vectors.
    # Each inner list in data["features"] will become a row in the DataFrame.
    features_df = pd.DataFrame(data["features"])
    
    # Add the labels as a new column to the DataFrame.
    # The column will be named 'genre_label'.
    features_df["genre_label"] = data["labels"]
    
    # Display the first 5 rows of the new DataFrame to verify.
    '''print("DataFrame created successfully. Here are the first 5 rows:")
    print(features_df.head())'''

     # Save the DataFrame to a CSV file
    print(f"Saving DataFrame to {csv_path}...")
    features_df.to_csv(csv_path, index=False)
    

    # After the loop, we will save the data to the specified JSON file
    # with open(json_path, "w") as fp:
    #     json.dump(data, fp, indent=4)
    
    
    print("\nFeature extraction complete. The file 'features.csv' has beeen created.")

if __name__ == "__main__":
    process_dataset(DATASET_PATH, CSV_PATH)