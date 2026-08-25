from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Using the information in the context, give a comprehensive answer to the question.",
        ),
        MessagesPlaceholder(variable_name="history", n_messages=6),
        (
            "user",
            "Context: {context} --- Now here is the question you need to answer. Question: {question}",
        ),
    ]
)
