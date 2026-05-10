from dotenv import load_dotenv
import os
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from datasets import Dataset


# =========================================
# CONFIGURACIÓN
# =========================================

load_dotenv()


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

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
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
# MODELO LOCAL FLAN-T5
# =========================================

print("\nCargando modelo FLAN-T5...")

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-base"
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-base"
)

print("Modelo cargado correctamente")


# =========================================
# PREGUNTAS
# =========================================

preguntas = [

    # =====================================
    # TIPO 1
    # Respuesta textual en el documento
    # =====================================

    "¿Qué significa RAG?",
    "¿Qué es ChromaDB?",


    # =====================================
    # TIPO 2
    # Vocabulario diferente
    # =====================================

    "¿Para qué sirve dividir documentos en fragmentos?",
    "¿Cómo se representan los textos numéricamente?",


    # =====================================
    # TIPO 3
    # Requiere combinar varios chunks
    # =====================================

    "Explica el flujo general de un sistema RAG.",
    "¿Cómo trabajan juntos embeddings, ChromaDB y el LLM?",


    # =====================================
    # TIPO 4
    # Fuera del dominio
    # =====================================

    "¿Qué es una derivada parcial?",
    "¿Cómo funciona la fotosíntesis?"
]


# =========================================
# RESPUESTAS ESPERADAS
# =========================================

ground_truths = [

    # Tipo 1
    "RAG significa Retrieval Augmented Generation.",
    "ChromaDB es una base de datos vectorial utilizada para almacenar embeddings.",

    # Tipo 2
    "El chunking divide documentos largos en fragmentos pequeños.",
    "Los embeddings son representaciones numéricas del texto.",

    # Tipo 3
    "Un sistema RAG carga documentos, divide chunks, genera embeddings, almacena vectores y usa un LLM para responder.",
    "Los embeddings representan texto, ChromaDB almacena vectores y el LLM genera respuestas.",

    # Tipo 4
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
Contesta únicamente usando la información del contexto.

Si la respuesta no aparece claramente en el contexto responde EXACTAMENTE:

La pregunta está fuera del dominio del asistente.

Responde en una sola oración corta.

CONTEXTO:
{contexto_unido}

PREGUNTA:
{pregunta}

RESPUESTA:
"""

    # =========================================
    # TOKENIZAR
    # =========================================

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )


    # =========================================
    # GENERAR RESPUESTA
    # =========================================

    outputs = model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=False
)


    respuesta = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


    print("RESPUESTA:\n")
    print(respuesta)

    answers.append(respuesta)
    contexts.append(contexto)


# =========================================
# DATASET
# =========================================

dataset = Dataset.from_dict({

    "question": preguntas,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": [[g] for g in ground_truths]
})


# =========================================
# RESULTADOS MANUALES
# =========================================

analisis = [

    "La respuesta fue parcialmente correcta y el sistema recuperó contexto relevante.",

    "La respuesta utilizó correctamente información del documento.",

    "La consulta probó recuperación semántica mediante embeddings.",

    "La respuesta mostró limitaciones semánticas del modelo.",

    "La respuesta combinó información de varios chunks.",

    "La consulta requirió integración de múltiples fragmentos.",

    "El sistema mostró dificultades para rechazar preguntas fuera del dominio.",

    "La respuesta evidenció una posible alucinación del modelo."
]

df = pd.DataFrame({

    "Pregunta": preguntas,
    "Respuesta_generada": answers,
    "Ground_truth": ground_truths,
    "Analisis": analisis
})

print("\n")
print("=" * 60)
print("CONFIGURACIÓN RAG")
print("=" * 60)

configuracion = pd.DataFrame({

    "Parámetro": [
        "Documento(s)",
        "Modelo embeddings",
        "chunk_size / overlap",
        "k",
        "LLM generador"
    ],

    "Valor": [
        "documento_rag_ejemplo.pdf",
        "sentence-transformers/all-MiniLM-L6-v2",
        "500 / 50",
        "3",
        "google/flan-t5-base"
    ]
})

print(configuracion)


print("\n")
print("=" * 60)
print("RESULTADOS")
print("=" * 60)

print(df)


# =========================================
# GUARDAR CSV
# =========================================

df.to_csv(
    "resultados_rag.csv",
    index=False
)

print("\nArchivo generado:")
print("resultados_rag.csv")