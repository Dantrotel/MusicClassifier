import os
import shutil
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict
from pydantic import BaseModel
from predict import predict_genre, genre_mapping
from dotenv import load_dotenv

load_dotenv()

from rag.chains import explain_genre, chat_about_music, get_recommendations

router = APIRouter()

# Aseguramos que exista la carpeta temporal dentro de backend/
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Recibe un archivo de audio (.mp3 o .wav), lo guarda temporalmente,
    extrae sus características, hace la predicción y limpia el temporal.
    """
    # 1. Validar la extensión
    if not file.filename.lower().endswith(('.mp3', '.wav')):
        raise HTTPException(status_code=400, detail="Formato de archivo no soportado. Sube un .mp3 o .wav")
        
    temp_path = os.path.join(TEMP_DIR, file.filename)
    
    # 2. Guardar archivo temporal
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception:
        raise HTTPException(status_code=500, detail="Error guardando el archivo de audio en el servidor.")
        
    # 3. Predicción
    try:
        result = predict_genre(temp_path)
    except ValueError as e:
        # Errores en procesamiento de librosa o mismatch de dimensiones
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Error del servidor (modelos no cargados)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # Otros errores
        raise HTTPException(status_code=500, detail=f"Error inesperado en la predicción: {str(e)}")
    finally:
        # 4. Limpieza automática del temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return result

@router.get("/genres")
def get_genres() -> Dict[str, str]:
    """
    Devuelve la lista de géneros disponibles entrenados en el modelo.
    """
    if not genre_mapping:
        raise HTTPException(status_code=500, detail="El diccionario de géneros no está cargado.")
    return genre_mapping

class ExplainRequest(BaseModel):
    genre: str
    probabilities: dict[str, float]

class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []

class RecommendRequest(BaseModel):
    history: list[dict]

@router.post("/explain")
async def explain_endpoint(req: ExplainRequest):
    try:
        explanation = explain_genre(req.genre, req.probabilities)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        response = chat_about_music(req.question, req.history)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommend")
async def recommend_endpoint(req: RecommendRequest):
    if not req.history:
        raise HTTPException(status_code=400, detail="El historial está vacío")
    try:
        recommendations = get_recommendations(req.history)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
