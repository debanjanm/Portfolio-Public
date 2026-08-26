from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from common.model import load_embedding_model


class RetrieveQuery:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def load_vectorstore(self, db_load_path):
        self.vector_store = FAISS.load_local(
            db_load_path, self.embedding_model, allow_dangerous_deserialization=True
        )

    def retrieve_context(self, query):
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 2}
        )
        context = self.retriever.invoke(query)
        return context


if __name__ == "__main__":
    vectorstore_path = "../artifacts/database/PDFs/"

    retrieve_query = RetrieveQuery(load_embedding_model())
    retrieve_query.load_vectorstore(vectorstore_path)
    query = "I am Depressed"
    results = retrieve_query.retrieve_context(query)
    for res in results:
        print(f"* {res.page_content}")
