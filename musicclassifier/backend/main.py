# pyrefly: ignore [missing-import]
from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
from router import router

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="MusicClassifier API",
    description="API de Machine Learning para clasificar géneros musicales",
    version="1.0.0"
)

# Configurar CORS para permitir comunicación con el Frontend (React/Vite en puerto 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Origen del Frontend
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los headers
)

# Registrar las rutas de nuestra API
app.include_router(router)

@app.get("/")
def read_root():
    return {
        "message": "Bienvenido a la API de MusicClassifier. Ve a /docs para ver la documentación interactiva y probar los endpoints."
    }
