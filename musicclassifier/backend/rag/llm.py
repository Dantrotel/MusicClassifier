import os
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("La variable de entorno GEMINI_API_KEY no está definida. Por favor, configúrala antes de ejecutar.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key= os.getenv("GEMINI_API_KEY")
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key= os.getenv("GEMINI_API_KEY")
)
