from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .llm import llm
from .vectorstore import get_retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def explain_genre(genre, features_dict):
    try:
        retriever = get_retriever()
        if not retriever:
            return "No se pudo cargar el contexto para la explicación."
        
        top_features = sorted(features_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        features_str = ", ".join([f"{k} ({v})" for k, v in top_features])

        prompt = ChatPromptTemplate.from_template(
            "Eres un experto en música y procesamiento de audio. "
            "Se ha predicho que una canción pertenece al género {genre}. "
            "Los 5 features acústicos más relevantes extraídos son: {features}.\n\n"
            "Basándote en el siguiente contexto extraído de Wikipedia:\n"
            "{context}\n\n"
            "Explica en español, en 3 o 4 oraciones, por qué una canción con estas características musicales "
            "probablemente pertenece al género {genre}. Sé claro y conciso."
        )

        chain = prompt | llm | StrOutputParser()
        docs = retriever.invoke(genre)
        context = format_docs(docs)
        
        response = chain.invoke({
            "genre": genre,
            "features": features_str,
            "context": context
        })
        
        return response
    except Exception as e:
        return f"Ocurrió un error al generar la explicación: {str(e)}"

def chat_about_music(question, history=[]):
    try:
        retriever = get_retriever()
        if not retriever:
            return "El sistema no está disponible en este momento."

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Eres un asistente amable experto en géneros musicales. "
             "Solo puedes responder preguntas relacionadas con música, géneros musicales, instrumentos y artistas. "
             "Si la pregunta del usuario no está relacionada con la música, responde amablemente que solo puedes hablar sobre música. "
             "Usa el siguiente contexto recuperado de Wikipedia para complementar tu respuesta si es útil:\n\n{context}\n\n"
             "Responde siempre en español."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        chain = prompt | llm | StrOutputParser()
        docs = retriever.invoke(question)
        context = format_docs(docs)

        formatted_history = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted_history.append(("human", content))
            else:
                formatted_history.append(("ai", content))

        response = chain.invoke({
            "context": context,
            "history": formatted_history,
            "question": question
        })
        return response
    except Exception as e:
        return f"Ocurrió un error al procesar tu pregunta: {str(e)}"

def get_recommendations(history):
    try:
        if not history:
            return "No hay suficientes predicciones en el historial para hacer recomendaciones."

        genre_counts = {}
        for item in history:
            g = item.get("genre")
            if g:
                genre_counts[g] = genre_counts.get(g, 0) + 1
        
        if not genre_counts:
            return "No se encontraron géneros válidos en el historial."

        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        genres_list = [g[0] for g in top_genres]
        genres_str = ", ".join(genres_list)

        retriever = get_retriever()
        if not retriever:
            return "No se pudo cargar el contexto para las recomendaciones."

        docs = []
        for g in genres_list:
            docs.extend(retriever.invoke(g))
        context = format_docs(docs)

        prompt = ChatPromptTemplate.from_template(
            "Eres un experto recomendador de música. "
            "El usuario ha estado analizando canciones y sus géneros más frecuentes han sido: {genres}.\n\n"
            "Basándote en estos gustos y usando el siguiente contexto como referencia (opcional):\n"
            "{context}\n\n"
            "Recomienda en español 3 artistas específicos o subgéneros que el usuario debería explorar. "
            "Explica brevemente por qué le podrían gustar basándote en su historial. "
            "No menciones la palabra 'contexto' o 'Wikipedia' en tu respuesta."
        )

        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({
            "genres": genres_str,
            "context": context
        })
        return response
    except Exception as e:
        return f"Ocurrió un error al generar las recomendaciones: {str(e)}"
