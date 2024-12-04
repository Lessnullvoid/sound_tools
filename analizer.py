#analize the spectral and harmonic components of the audio file 

import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d import Axes3D 
from IPython.display import display


# Load the audio file
file_path = 'peaks.wav'
y, sr = librosa.load(file_path)

# Function to analyze the duration of the audio file in milliseconds
def analyze_duration(y, sr):
    duration_seconds = librosa.get_duration(y=y, sr=sr)
    duration_milliseconds = duration_seconds * 1000
    print(f"Duration: {duration_milliseconds:.2f} ms")

# Call the function to analyze the duration
analyze_duration(y, sr)


# Compute the Short-Time Fourier Transform (STFT)
D = librosa.stft(y)

# Convert the amplitude to decibels
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Create a DataFrame for the spectrogram
spectrogram_df = pd.DataFrame(S_db)

# Compute the chroma feature
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

# Create a DataFrame for the chroma feature
chroma_df = pd.DataFrame(chroma)

# Compute the dominant frequencies
frequencies = np.linspace(0, sr/2, int(1 + D.shape[0]//2))
dominant_frequencies = frequencies[np.argmax(np.abs(D), axis=0)]

# Calculate the mean of the dominant frequencies
mean_dominant_frequency = np.mean(dominant_frequencies)

# Identify the most present frequency
most_present_frequency = np.bincount(dominant_frequencies.astype(int)).argmax()

# Print the dominant frequencies as a list
print("Dominant Frequencies:", dominant_frequencies.tolist())
print("Mean Dominant Frequency:", mean_dominant_frequency)
print("Most Present Frequency:", most_present_frequency)


# Print the dominant frequencies as a list
print("Dominant Frequencies:", dominant_frequencies.tolist())

# Create a DataFrame for the dominant frequencies
dominant_freq_df = pd.DataFrame(dominant_frequencies, columns=['Dominant Frequency'])

# Compute the harmonic component of the audio signal
harmonic, _ = librosa.effects.hpss(y)

# Compute the STFT of the harmonic component
D_harmonic = librosa.stft(harmonic)

# Convert the amplitude to decibels
S_db_harmonic = librosa.amplitude_to_db(np.abs(D_harmonic), ref=np.max)

# Plot the spectrogram, chroma feature, dominant frequencies, and harmonic in a 2x2 grid
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))  # Adjusted figure size

# Plot the spectrogram as a 3D plot in the 2x2 grid
ax1 = fig.add_subplot(221, projection='3d')  # Use ax1 for the 3D plot

# Create meshgrid for time and frequency
time = np.arange(S_db.shape[1])
frequency = np.arange(S_db.shape[0])
T, F = np.meshgrid(time, frequency)

# Plot the surface
ax1.plot_surface(T, F, S_db, cmap='magma')

ax1.set_title('3D Spectrogram', fontsize=8)  # Adjusted font size
ax1.set_xlabel('Time', fontsize=6)           # Adjusted font size
ax1.set_ylabel('Frequency', fontsize=6)      # Adjusted font size
ax1.set_zlabel('Amplitude (dB)', fontsize=6) # Adjusted font size

# Plot the chroma feature
sns.heatmap(chroma_df, cmap='coolwarm', cbar_kws={'label': 'Intensity'}, ax=ax2)
ax2.set_title('Chroma Feature', fontsize=8)  # Adjusted font size
ax2.set_xlabel('Time', fontsize=6)            # Adjusted font size
ax2.set_ylabel('Chroma', fontsize=6)          # Adjusted font size

# Plot the dominant frequencies
ax3.plot(dominant_freq_df)
ax3.set_title('Dominant Frequencies', fontsize=8)  # Adjusted font size
ax3.set_xlabel('Time', fontsize=6)                  # Adjusted font size
ax3.set_ylabel('Frequency (Hz)', fontsize=6)        # Adjusted font size

# Plot the harmonic component as a 3D plot
ax4 = fig.add_subplot(224, projection='3d')  # Use ax4 for the 3D plot

# Create meshgrid for time and frequency for harmonic
time_harmonic = np.arange(S_db_harmonic.shape[1])
frequency_harmonic = np.arange(S_db_harmonic.shape[0])
T_harmonic, F_harmonic = np.meshgrid(time_harmonic, frequency_harmonic)

# Plot the surface for harmonic
ax4.plot_surface(T_harmonic, F_harmonic, S_db_harmonic, cmap='viridis')

ax4.set_title('3D Harmonic', fontsize=8)  # Adjusted font size
ax4.set_xlabel('Time', fontsize=6)        # Adjusted font size
ax4.set_ylabel('Frequency', fontsize=6)   # Adjusted font size
ax4.set_zlabel('Amplitude (dB)', fontsize=6)  # Adjusted font size

plt.tight_layout()
plt.show()