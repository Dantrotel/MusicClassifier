import joblib
import json
import os
import numpy as np
from features import extract_features

# Ruta base relativa a este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar artefactos de Machine Learning en memoria al arrancar
try:
    model = joblib.load(os.path.join(BASE_DIR, 'model_final.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler_final.pkl'))
    
    with open(os.path.join(BASE_DIR, 'genres_final.json'), 'r') as f:
        genre_mapping = json.load(f)
except FileNotFoundError as e:
    print(f"Advertencia: No se encontraron los archivos del modelo: {e}")
    model = None
    scaler = None
    genre_mapping = {}

def predict_genre(file_path: str) -> dict:
    """
    1. Extrae features del audio
    2. Aplica el StandardScaler
    3. Predice el género usando el modelo cargado
    4. Retorna el género predicho y las probabilidades
    """
    if not model or not scaler:
        raise RuntimeError("Los modelos no están cargados. Asegúrate de tener model.pkl y scaler.pkl")
        
    # 1. Extraer features
    features = extract_features(file_path)
    
    # 2. Escalar características (necesita ser un array 2D de 1 muestra)
    features_2d = features.reshape(1, -1)
    
    try:
        features_scaled = scaler.transform(features_2d)
    except ValueError as e:
        # Este error puede ocurrir si librosa extrajo menos o más columnas que las usadas al entrenar
        raise ValueError(f"Las dimensiones de las características no coinciden con el modelo. {str(e)}")
    
    # 3. Predecir
    predicted_idx = int(model.predict(features_scaled)[0])
    predicted_genre = genre_mapping[predicted_idx] if predicted_idx < len(genre_mapping) else "Desconocido"
    
    # 4. Probabilidades (si el modelo las soporta)
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_scaled)[0]
        for idx, prob in enumerate(probs):
            g_name = genre_mapping[idx] if idx < len(genre_mapping) else f"Clase_{idx}"
            probabilities[g_name] = round(float(prob), 4)
    else:
        # Para SVM con probability=False
        probabilities = {predicted_genre: 1.0}
        
    return {
        "genre": predicted_genre,
        "probabilities": probabilities
    }
