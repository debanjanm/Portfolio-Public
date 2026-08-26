from uuid import uuid4

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from common.model import load_embedding_model


class IndexData:
    def __init__(self):
        self.embedding_model = load_embedding_model()

    def load_document(self, document_path):
        self.document_path = document_path
        self.document = PyMuPDFLoader(self.document_path)

    def split_document(self):
        self.documents = self.document.load()

    def create_vectorstore(self):
        index = faiss.IndexFlatL2(len(self.embedding_model.embed_query("hello world")))
        self.vector_store = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        uuids = [str(uuid4()) for _ in range(len(self.documents))]
        self.vector_store.add_documents(documents=self.documents, ids=uuids)

    def save_vectorstore(self, db_save_path):
        self.vector_store.save_local(db_save_path)


if __name__ == "__main__":
    document_path = "../artifacts/dataset/PDFs/DepressionGuide-web.pdf"
    vectorstore_path = "../artifacts/database/PDFs/"

    index_data = IndexData()
    index_data.load_document(document_path)
    index_data.split_document()
    index_data.create_vectorstore()
    index_data.save_vectorstore(vectorstore_path)
