from langchain_community.chat_message_histories import SQLChatMessageHistory
from common.config import SQLITE_DB_PATH

def get_conversation_history(user_id, conversation_id):

    connection = "sqlite:///" + SQLITE_DB_PATH
    chat_message_history = SQLChatMessageHistory(
        f"{user_id}--{conversation_id}", connection=connection
    )
    return chat_message_history