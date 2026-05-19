import os
from pathlib import Path
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma
from .llm import embeddings

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

GENRES = [
    "Electronic music",
    "Experimental music",
    "Folk music",
    "Hip hop music",
    "Instrumental music",
    "World music",
    "Pop music",
    "Rock music"
]

def build_vectorstore():
    print("Construyendo el vectorstore desde Wikipedia...")
    documents = []
    
    for genre in GENRES:
        print(f"Descargando artículo para: {genre}...")
        try:
            loader = WikipediaLoader(query=genre, load_max_docs=1, lang="en")
            docs = loader.load()
            if docs:
                documents.extend(docs)
                print(f"✅ {genre} descargado exitosamente.")
            else:
                print(f"⚠️ No se encontraron documentos para {genre}.")
        except Exception as e:
            print(f"❌ Error al descargar {genre}: {e}")
            continue
            
    if not documents:
        print("⚠️ No se descargó ningún documento. El vectorstore estará vacío.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Se crearon {len(chunks)} chunks de texto.")
    
    print("Persistiendo en ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DB_DIR)
    )
    print("✅ Vectorstore construido y persistido exitosamente.")
    return vectorstore

def load_vectorstore():
    if CHROMA_DB_DIR.exists() and any(CHROMA_DB_DIR.iterdir()):
        print("Cargando vectorstore existente...")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DB_DIR),
            embedding_function=embeddings
        )
        return vectorstore
    else:
        print("El vectorstore no existe. Construyendo uno nuevo...")
        return build_vectorstore()

def get_retriever():
    vectorstore = load_vectorstore()
    if vectorstore:
        return vectorstore.as_retriever(search_kwargs={"k": 3})
    return None
