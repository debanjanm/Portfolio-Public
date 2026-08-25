from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory


class GenerateResponse:
    def __init__(self, language_model):
        self.language_model = language_model

    def generate_response(
        self,
        prompt_template,
        query,
        context,
        user_id,
        conversation_id,
        get_conversation_history,
    ):
        runnable = prompt_template | self.language_model

        runnable_with_message_history = RunnableWithMessageHistory(
            runnable,
            get_conversation_history,
            input_messages_key="question",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for the user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="Conversation ID",
                    description="Unique identifier for the conversation.",
                    default="",
                    is_shared=True,
                ),
            ],
        )

        config = {
            "configurable": {"user_id": user_id, "conversation_id": conversation_id}
        }

        ai_msg = runnable_with_message_history.invoke(
            {"question": query, "context": context}, config
        )

        return ai_msg.content
