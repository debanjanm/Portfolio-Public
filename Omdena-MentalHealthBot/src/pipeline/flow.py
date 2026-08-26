from common.config import PDF_DB_PATH
from common.model import load_embedding_model, load_language_model
from core.generator import GenerateResponse
from core.prompter import rag_prompt
from core.retriever import RetrieveQuery
from utils.memory_utils import get_conversation_history


class MentalChatbot:
    def __init__(self):
        self.embedding_model = load_embedding_model()
        self.language_model  = load_language_model()

        self.retriever = RetrieveQuery(self.embedding_model)
        print(f"PDF_DB_PATH:{PDF_DB_PATH}")
        self.retriever.load_vectorstore(PDF_DB_PATH)

        self.generator = GenerateResponse(self.language_model)

    def retrieving(self, query):
        return self.retriever.retrieve_context(query)

    def generating(self, query, context):
        return self.generator.generate_response(rag_prompt, query, context, self.user_id, self.conversation_id, get_conversation_history)

    def executing(self, query, user_id, conversation_id):
        self.user_id = user_id
        self.conversation_id = conversation_id

        context = self.retrieving(query)

        response = self.generating(query, context)
        return response