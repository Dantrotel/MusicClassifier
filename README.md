# MusicClassifier

Clasificador de géneros musicales con Machine Learning, FastAPI, React y RAG con LangChain + Gemini.

## Arquitectura

Frontend (React + Vite + Tailwind)
↓
Backend (FastAPI)
├── /predict    → Clasificación con SVM
├── /explain    → Explicación con RAG
├── /chat       → Chatbot musical
└── /recommend  → Recomendaciones
↓
Capa RAG (LangChain + Gemini + ChromaDB)


## Stack Tecnológico

| Capa | Tecnologías |
|---|---|
| Frontend | React, Vite, Tailwind CSS, Recharts |
| Backend | FastAPI, Uvicorn |
| ML | scikit-learn (SVM), librosa, XGBoost |
| RAG | LangChain, ChromaDB, Google Gemini |
| Dataset | FMA Small (8.000 tracks, 8 géneros) |

## Instalación

### Backend
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend
```bash
cd musicclassifier/frontend
pnpm install
```

## ⚙️ Configuración

Crea un archivo `.env` en `musicclassifier/backend/`:

GEMINI_API_KEY=tu_api_key_aqui


## Ejecución

### Backend
```bash
cd musicclassifier/backend
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd musicclassifier/frontend
pnpm dev
```

## Géneros Clasificados

Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock

## Métricas del Modelo

| Modelo | F1 Macro | Test Accuracy |
|---|---|---|
| SVM (final) | 0.57 | 58% |

## Funcionalidades RAG

- **Explicación de predicciones** — El LLM explica por qué una canción pertenece a un género
- **Chatbot musical** — Responde preguntas sobre géneros usando Wikipedia como contexto
- **Recomendaciones** — Sugiere artistas y subgéneros basándose en el historial