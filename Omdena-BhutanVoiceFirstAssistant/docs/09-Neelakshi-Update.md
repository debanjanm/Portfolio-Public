If I understood correctly:
1) colab notebook is using KNOWLEDGE_BASE: List[KnowledgeChunk] = [.....] to answer query. There are no saved chunks from actual research document in vector database.
This explains why did I receive only particular document list and office address in response.

2) For Task 2:  CLASSIFIER_PROMPT_TEMPLATE or the REQUEST_TYPES / SERVICES are listed in colab notebook and not in data_retrieval.py file on github.

My question is what is expected in "Edit the prompt or labels to improve accuracy" task?

Example: when I added "tour" in 
class KeywordClassifier(BaseClassifier): OUT_OF_SCOPE_WORDS = [...]
it did not classify my query "how to plan tour in Bhutan?" as out-of-scope. Instead it is classified in-scope, safe, general-info, permits.

Please need your guidance.