from langchain.memory import ConversationBufferMemory

from common.config import PDF_DB_PATH
from common.model import load_embedding_model, load_language_model
from core.generator import GenerateResponse
from core.prompter import rag_prompt
from core.retriever import RetrieveQuery
from utilities.memory_utils import (get_conversation_history,
                                      post_process_chat_history)


class MentalChatbot:
    def __init__(self):
        self.embedding_model = load_embedding_model()
        self.language_model  = load_language_model()

        self.retriever = RetrieveQuery(self.embedding_model)
        print(f"PDF_DB_PATH:{PDF_DB_PATH}")
        self.retriever.load_vectorstore(PDF_DB_PATH)

        self.generator = GenerateResponse(self.language_model)
    
    def indexing():
        pass

    def retrieving(self, query):
        self.retriever.retrieve_context(query)

    def prompting():
        pass

    def generating(self, query, context):
        return self.generator.generate_response(rag_prompt, query, context, self.user_id, self.conversation_id, get_conversation_history) 

    def executing(self, query, user_id, conversation_id):
        self.user_id = user_id 
        self.conversation_id = conversation_id

        memory = ConversationBufferMemory(chat_memory=get_conversation_history(user_id, conversation_id), memory_key="history", k=2, return_messages=True)
        memory_dict = post_process_chat_history(memory)

        context = self.retrieving(query)

        response = self.generating(query, context)
        return response