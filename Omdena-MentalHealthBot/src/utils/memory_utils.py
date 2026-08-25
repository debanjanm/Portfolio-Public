from langchain_community.chat_message_histories import SQLChatMessageHistory
from common.config import SQLITE_DB_PATH

def get_conversation_history(user_id, conversation_id):

    connection = "sqlite:///" + SQLITE_DB_PATH
    chat_message_history = SQLChatMessageHistory(
        f"{user_id}--{conversation_id}", connection=connection
    )
    return chat_message_history

def post_process_chat_history(memory):
    chat_memory = memory.load_memory_variables({}).get("history")
    memory_dict = {}
    if chat_memory == []:
        return memory_dict
    else:
        for i in range(0, len(chat_memory), 2):
            memory_dict.update(
                {   
                    f"Human_Message_{i}": chat_memory[i].content,
                    f"AI_Message_{i}": chat_memory[i+1].content,
                }
            )
        return memory_dict