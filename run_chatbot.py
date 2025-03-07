import chromadb
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.langchain import LangchainEmbedding
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embedding_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("faq")

# Create LlamaIndex VectorStore
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the index
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embedding_model)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.3  # 0.3 low and 0.7 high

def find_best_match(user_query):
    query_embedding = embedding_model.get_query_embedding(user_query)
    results = chroma_collection.query(query_embeddings=[query_embedding], n_results=1)

    if not results["ids"] or not results["ids"][0]:  # Ensure there is a valid match
        return None, None, 0.0  # No match found, confidence is 0

    best_question = results["ids"][0][0]  # Extract the matched question ID
    best_answer = results["metadatas"][0][0]["answer"].replace("\\n", "\n")  # Restore line breaks
    confidence = results["distances"][0][0]  # Extract the similarity score (distance)

    # Convert distance to confidence (higher distance = lower confidence)
    confidence = 1 - confidence  # Normalize to a 0-1 range

    return best_question, best_answer, confidence

class Bot:
    def __init__(self):
        self.last_question = None
        self.last_answer = None

    def answer_question(self, user_query):
        if self.last_question and ("explain" in user_query.lower() or "further" in user_query.lower()):
            return self.last_answer  # Use last answer for follow-up questions

        best_question, full_answer, confidence = find_best_match(user_query)

        # Check if the confidence is below the threshold
        if not best_question or confidence < CONFIDENCE_THRESHOLD:
            return "I'm sorry, I don't have enough information to answer that question. Please try rephrasing or ask a different question."

        self.last_question = user_query
        self.last_answer = full_answer

        return full_answer  # Returns the **entire** answer

# Initialize chatbot
chatbot = Bot()

# Interactive Chat
print('Reedy: Hello! How may I help you?\nType "exit" to quit.')
while True:
    user_query = input('\nYou: ')
    if user_query.lower() == 'exit':
        break
    response = chatbot.answer_question(user_query)
    print(f"Reedy: {response}")
