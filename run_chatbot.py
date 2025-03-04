import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")  
collection = chroma_client.get_or_create_collection("faq")

def find_best_match(user_query):
    """Finds the closest question using vector search."""
    query_embedding = embedding_model.encode(user_query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    if not results["ids"] or not results["ids"][0]:  # Ensure there is a valid match
        return None, None

    best_question = results["ids"][0][0]  # Extract the matched question ID
    best_answer = results["metadatas"][0][0]["answer"]  # Retrieve the answer
    return best_question, best_answer

class Bot:
    def __init__(self):
        """Initialize chatbot with no need for reprocessing."""
        self.last_question = None
        self.last_answer = None

    def answer_question(self, user_query):
        """Finds the best match and returns the full answer."""
        if self.last_question and ("explain" in user_query.lower() or "further" in user_query.lower()):
            return self.last_answer  # Use last answer for follow-up questions

        best_question, full_answer = find_best_match(user_query)
        if not best_question:
            return "That question is outside my knowledge scope."

        self.last_question = user_query
        self.last_answer = full_answer

        return full_answer  #Returns the entire answer

chatbot = Bot()

print('Reedy: Hello! How may I help you?\nType "exit" to quit.')
while True:
    user_query = input('\nYou: ')
    if user_query.lower() == 'exit':
        break
    response = chatbot.answer_question(user_query)
    print(f"Reedy: {response}")
