##========================================================================================##
import os

from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import (ChatMessageHistory, ConversationBufferMemory,
                              ConversationSummaryBufferMemory)
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
##========================================================================================##
from rag.generator import load_cllm

llm = load_cllm()

# array_of_files
from rag.retriever import instantiate_rag

retriever = instantiate_rag()

##========================================================================================##
# Define system and user message templates
with open(
    os.path.join(script_dir, "prompts", "system_message_template.txt"), "r"
) as file:
    system_message_template = file.read().replace("\n", "")

with open(os.path.join(script_dir, "prompts", "user_mesage_template.txt"), "r") as file:
    user_message_template = file.read().replace("\n", "")

with open(
    os.path.join(script_dir, "prompts", "condense_question_prompt.txt"), "r"
) as file:
    condense_question_prompt = file.read().replace("\n", "")

# Create message templates
system_message = SystemMessagePromptTemplate.from_template(system_message_template)
user_message = HumanMessagePromptTemplate.from_template(user_message_template)

# Compile messages into a chat prompt template
messages = [system_message, user_message]
chatbot_prompt = ChatPromptTemplate.from_messages(messages)


##========================================================================================##
history = ChatMessageHistory()
# Provide the chat history when initializing the ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=ConversationSummaryBufferMemory(
        memory_key="chat_history",
        input_key="question",
        llm=llm,
        max_token_limit=40,
        return_messages=True,
    ),
    return_source_documents=False,
    chain_type="stuff",
    max_tokens_limit=100,
    condense_question_prompt=PromptTemplate.from_template(condense_question_prompt),
    combine_docs_chain_kwargs={"prompt": chatbot_prompt},
    verbose=True,
    return_generated_question=False,
)

##========================================================================================##


def LLM_generator(question: str):
    answer = qa({"question": question, "chat_history": history.messages})["answer"]
    print("##------##")
    return answer


##========================================================================================##

# Implement Classification

from analyzer.categorizer import (corelation_analysis, pattern_classification,
                                  sentiment_class)

# is_depressed = sentiment_class(conversation_buffer)

# random
# Initialize the 2D list
is_depressed = [[]]

# Assign a probability value to the cell at row 0 and column 1
is_depressed[0].append(0.75)

# Check if the probability value is greater than 0.5 and print the result
if is_depressed[0][0] > 0.5:
    print("Not so depressed")
else:
    print("is_depressed")

##========================================================================================##
