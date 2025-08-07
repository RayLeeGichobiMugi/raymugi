#!/usr/bin/env python3
"""
Complete Audio Recognition Pipeline using Machine Learning
Demonstrates preprocessing, feature extraction, and classification
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class AudioRecognitionPipeline:
    """Complete pipeline for audio recognition using ML"""
    
    def __init__(self, sample_rate=22050, duration=5.0, n_mels=128, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_len = int(sample_rate * duration)
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_audio(self, file_path):
        """Load and preprocess audio file"""
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Pad or truncate to fixed length
            if len(audio) < self.max_len:
                # Pad with zeros
                audio = np.pad(audio, (0, self.max_len - len(audio)), mode='constant')
            else:
                # Truncate
                audio = audio[:self.max_len]
                
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_features(self, audio):
        """Extract comprehensive audio features"""
        features = {}
        
        # 1. Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # 2. Mel-scaled spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_mels=self.n_mels, 
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_spectrogram'] = log_mel_spec
        
        # 3. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        features['chroma_std'] = np.std(chroma, axis=1)
        
        # 4. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # 5. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 6. Tempo and rhythmic features
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features['tempo'] = tempo
        except:
            features['tempo'] = 0
            
        return features
    
    def create_traditional_feature_vector(self, features):
        """Create feature vector for traditional ML models"""
        feature_vector = []
        
        # Add all scalar features
        feature_vector.extend(features['mfcc_mean'])
        feature_vector.extend(features['mfcc_std'])
        feature_vector.extend(features['chroma_mean'])
        feature_vector.extend(features['chroma_std'])
        feature_vector.append(features['spectral_centroid_mean'])
        feature_vector.append(features['spectral_centroid_std'])
        feature_vector.append(features['spectral_rolloff_mean'])
        feature_vector.append(features['spectral_rolloff_std'])
        feature_vector.append(features['zcr_mean'])
        feature_vector.append(features['zcr_std'])
        feature_vector.append(features['tempo'])
        
        return np.array(feature_vector)
    
    def build_cnn_model(self, input_shape, num_classes):
        """Build CNN model for spectrogram-based classification"""
        model = keras.Sequential([
            # Input layer
            keras.layers.Input(shape=input_shape),
            
            # Add channel dimension for Conv2D
            keras.layers.Reshape(input_shape + (1,)),
            
            # First convolutional block
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Second convolutional block
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Third convolutional block
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            
            # Global average pooling
            keras.layers.GlobalAveragePooling2D(),
            
            # Dense layers
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def visualize_audio_features(self, audio, features, title="Audio Analysis"):
        """Visualize audio waveform and extracted features"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Waveform
        time = np.linspace(0, len(audio) / self.sample_rate, len(audio))
        axes[0, 0].plot(time, audio)
        axes[0, 0].set_title('Waveform')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        
        # Mel-spectrogram
        librosa.display.specshow(
            features['mel_spectrogram'], sr=self.sample_rate, 
            hop_length=self.hop_length, x_axis='time', y_axis='mel', ax=axes[0, 1]
        )
        axes[0, 1].set_title('Mel-Spectrogram')
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        librosa.display.specshow(
            mfccs, sr=self.sample_rate, hop_length=self.hop_length, 
            x_axis='time', ax=axes[1, 0]
        )
        axes[1, 0].set_title('MFCCs')
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        librosa.display.specshow(
            chroma, sr=self.sample_rate, hop_length=self.hop_length, 
            x_axis='time', y_axis='chroma', ax=axes[1, 1]
        )
        axes[1, 1].set_title('Chroma Features')
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        time_frames = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=self.sample_rate, hop_length=self.hop_length)
        axes[2, 0].plot(time_frames, spectral_centroids)
        axes[2, 0].set_title('Spectral Centroid')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Hz')
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        time_frames = librosa.frames_to_time(np.arange(len(zcr)), sr=self.sample_rate, hop_length=self.hop_length)
        axes[2, 1].plot(time_frames, zcr)
        axes[2, 1].set_title('Zero Crossing Rate')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Rate')
        
        plt.tight_layout()
        return fig
    
    def train_traditional_ml_model(self, X, y):
        """Train traditional ML model (Random Forest)"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test)
        accuracy = rf_model.score(X_test, y_test)
        
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return rf_model, X_test, y_test, y_pred
    
    def train_cnn_model(self, X, y, epochs=50, batch_size=32):
        """Train CNN model on spectrograms"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        # Build and train model
        model = self.build_cnn_model(X_train.shape[1:], num_classes)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"CNN Test Accuracy: {test_accuracy:.4f}")
        
        return model, history, X_test, y_test

def demonstrate_audio_recognition():
    """Demonstrate the complete audio recognition pipeline"""
    
    # Initialize pipeline
    pipeline = AudioRecognitionPipeline()
    
    # Example: Create synthetic audio data for demonstration
    # In practice, you would load real audio files
    print("Creating synthetic audio data for demonstration...")
    
    # Generate synthetic audio samples
    samples = []
    labels = []
    
    # Sine wave (representing tonal sounds)
    for freq in [440, 880, 1320]:  # Different frequencies
        for _ in range(10):  # 10 samples each
            t = np.linspace(0, pipeline.duration, pipeline.max_len)
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
            # Add some noise
            audio += 0.1 * np.random.normal(size=audio.shape)
            samples.append(audio)
            labels.append(f"tone_{freq}Hz")
    
    # Noise (representing non-tonal sounds)
    for _ in range(30):
        audio = np.random.normal(0, 0.3, pipeline.max_len)
        samples.append(audio)
        labels.append("noise")
    
    print(f"Generated {len(samples)} audio samples with {len(set(labels))} classes")
    
    # Extract features
    print("Extracting features...")
    traditional_features = []
    spectrograms = []
    
    for i, audio in enumerate(samples):
        features = pipeline.extract_features(audio)
        
        # Traditional ML features
        feature_vector = pipeline.create_traditional_feature_vector(features)
        traditional_features.append(feature_vector)
        
        # Spectrogram for CNN
        spectrograms.append(features['mel_spectrogram'])
        
        if i == 0:  # Visualize first sample
            fig = pipeline.visualize_audio_features(audio, features, f"Sample Audio Analysis - {labels[i]}")
            plt.savefig('/workspace/audio_analysis_example.png', dpi=150, bbox_inches='tight')
            plt.close()
    
    # Convert to numpy arrays
    X_traditional = np.array(traditional_features)
    X_spectrograms = np.array(spectrograms)
    y = np.array(labels)
    
    print(f"Traditional features shape: {X_traditional.shape}")
    print(f"Spectrogram features shape: {X_spectrograms.shape}")
    
    # Train traditional ML model
    print("\n" + "="*50)
    print("Training Traditional ML Model (Random Forest)")
    print("="*50)
    rf_model, X_test_trad, y_test_trad, y_pred_trad = pipeline.train_traditional_ml_model(X_traditional, y)
    
    # Train CNN model
    print("\n" + "="*50)
    print("Training CNN Model")
    print("="*50)
    cnn_model, history, X_test_cnn, y_test_cnn = pipeline.train_cnn_model(X_spectrograms, y, epochs=30)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/workspace/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*50)
    print("Pipeline demonstration completed!")
    print("Generated files:")
    print("- audio_analysis_example.png: Feature visualization")
    print("- training_history.png: Training progress")
    print("="*50)

if __name__ == "__main__":
    demonstrate_audio_recognition()