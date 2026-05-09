from dotenv import load_dotenv
import os
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma

from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)

from datasets import Dataset


# =========================================
# CONFIGURACIÓN
# =========================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


# =========================================
# CARGA DE DOCUMENTOS
# =========================================

docs = []

loader = PyPDFLoader("documento_rag_ejemplo.pdf")

docs.extend(loader.load())

print(f"\nDocumentos cargados: {len(docs)}")


# =========================================
# CHUNKING
# =========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print(f"Chunks generados: {len(chunks)}")


# =========================================
# EMBEDDINGS
# =========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=API_KEY
)


# =========================================
# CREAR BASE VECTORIAL
# =========================================

if not os.path.exists("chroma_db"):

    print("\nCreando base vectorial...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    print("Base vectorial creada")

else:

    print("\nLa base vectorial ya existe")


# =========================================
# CARGAR CHROMA
# =========================================

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# =========================================
# LLM
# =========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=API_KEY,
    temperature=0
)


# =========================================
# PREGUNTAS
# =========================================

preguntas = [

    # Tipo 1
    "¿Qué es una clave primaria?",
    "¿Para qué sirve la cláusula WHERE?",

    # Tipo 2
    "¿Cómo se identifica de manera única un registro?",
    "¿Cómo se filtran datos específicos en SQL?",

    # Tipo 3
    "¿Cómo se relacionan dos tablas y qué elemento se utiliza para hacerlo?",
    "Explica cómo funciona una consulta SQL utilizando SELECT, FROM y WHERE.",

    # Tipo 4
    "¿Qué es una derivada parcial?",
    "¿Cómo funciona la fotosíntesis?"
]


# =========================================
# RESPUESTAS ESPERADAS
# =========================================

ground_truths = [

    "Una clave primaria identifica de manera única cada registro.",

    "WHERE permite filtrar resultados en SQL.",

    "Una clave primaria permite identificar un registro de manera única.",

    "Los datos pueden filtrarse utilizando la cláusula WHERE.",

    "Las tablas se relacionan mediante claves foráneas.",

    "SELECT obtiene datos, FROM indica la tabla y WHERE filtra resultados.",

    "La pregunta está fuera del dominio del asistente.",

    "La pregunta está fuera del dominio del asistente."
]


# =========================================
# GENERAR RESPUESTAS
# =========================================

answers = []
contexts = []

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

for pregunta in preguntas:

    docs_recuperados = retriever.invoke(pregunta)

    contexto = [
        doc.page_content
        for doc in docs_recuperados
    ]

    contexto_unido = "\n".join(contexto)

    print("\n" + "=" * 60)
    print(f"PREGUNTA: {pregunta}")

    print("\nFRAGMENTOS RECUPERADOS:\n")

    for i, doc in enumerate(docs_recuperados, 1):

        print(f"[{i}]")
        print(doc.page_content[:300])
        print()


    prompt = f"""
Eres un tutor académico especializado en bases de datos.

Responde únicamente usando el contexto proporcionado.

Si la respuesta no está en el contexto responde:
"La pregunta está fuera del dominio del asistente"

CONTEXTO:
{contexto_unido}

PREGUNTA:
{pregunta}
"""


    response = llm.invoke(prompt)

    respuesta = response.content

    print("RESPUESTA:\n")
    print(respuesta)

    answers.append(respuesta)
    contexts.append(contexto)


# =========================================
# DATASET RAGAS
# =========================================

dataset = Dataset.from_dict({

    "question": preguntas,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})


# =========================================
# EVALUACIÓN
# =========================================

resultado = evaluate(

    dataset=dataset,

    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision
    ]
)


# =========================================
# RESULTADOS
# =========================================

df = resultado.to_pandas()

print("\n")
print("=" * 60)
print("RESULTADOS RAGAS")
print("=" * 60)

print(df)


# =========================================
# GUARDAR CSV
# =========================================

df.to_csv(
    "resultados_ragas.csv",
    index=False
)

print("\nArchivo generado:")
print("resultados_ragas.csv")