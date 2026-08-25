import os

from langchain_google_genai import (ChatGoogleGenerativeAI,
                                    GoogleGenerativeAIEmbeddings)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "REDACTED_GOOGLE_API_KEY"


def load_language_model():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    return llm


def load_embedding_model():
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return emb
