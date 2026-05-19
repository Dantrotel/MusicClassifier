# pyrefly: ignore [missing-import]
import librosa
import numpy as np

def extract_features(file_path):
    """
    Extrae exactamente 355 características musicales del archivo de audio usando librosa.
    Está diseñado para coincidir a la perfección con la extracción realizada 
    durante el entrenamiento (Notebook 01b).
    
    Desglose:
    1. MFCCs (20x2 = 40 features)
    2. Chroma STFT (12x2 = 24 features)
    3. Spectral Contrast (7x2 = 14 features)
    4. Spectral Centroid (1x2 = 2 features)
    5. Spectral Rolloff (1x2 = 2 features)
    6. Zero Crossing Rate (1x2 = 2 features)
    7. RMS Energy (1x2 = 2 features)
    8. Tempo (1 feature)
    9. Tonnetz (6x2 = 12 features)
    10. Mel-spectrogram (128x2 = 256 features)
    
    Total: 40 + 24 + 14 + 2 + 2 + 2 + 2 + 1 + 12 + 256 = 355
    """
    try:
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30.0)
        
        features = []
        
        # 1. MFCC (20 coeficientes -> mean & var = 40)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for mean, var in zip(np.mean(mfcc, axis=1), np.var(mfcc, axis=1)):
            features.extend([mean, var])
            
        # 2. Chroma STFT (12 dimensiones -> mean & var = 24)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for mean, var in zip(np.mean(chroma, axis=1), np.var(chroma, axis=1)):
            features.extend([mean, var])
            
        # 3. Spectral Contrast (7 dimensiones -> mean & var = 14)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for mean, var in zip(np.mean(contrast, axis=1), np.var(contrast, axis=1)):
            features.extend([mean, var])
            
        # 4. Spectral Centroid (1 dimensión -> mean & var = 2)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([np.mean(centroid), np.var(centroid)])
        
        # 5. Spectral Rolloff (1 dimensión -> mean & var = 2)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([np.mean(rolloff), np.var(rolloff)])
        
        # 6. Zero Crossing Rate (1 dimensión -> mean & var = 2)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.var(zcr)])
        
        # 7. RMS Energy (1 dimensión -> mean & var = 2)
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.var(rms)])
        
        # 8. Tempo (1 dimensión escalar = 1)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(float(np.mean(tempo)))
        
        # 9. Tonnetz (6 dimensiones -> mean & var = 12)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        for mean, var in zip(np.mean(tonnetz, axis=1), np.var(tonnetz, axis=1)):
            features.extend([mean, var])
            
        # 10. Mel-spectrogram (128 dimensiones -> mean & var = 256)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        for mean, var in zip(np.mean(mel_db, axis=1), np.var(mel_db, axis=1)):
            features.extend([mean, var])
        
        assert len(features) == 355, f"Error: Se extrajeron {len(features)} features en lugar de 355"
        
        return np.array(features)
        
    except Exception as e:
        raise ValueError(f"Error procesando el archivo de audio con librosa: {str(e)}")
