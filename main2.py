import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Stores data persistently
collection = chroma_client.get_or_create_collection("faq")

def load_document(file_path):
    """Loads text from a given TXT document."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def process_document(text):
    """Extracts Q&A pairs and stores them in ChromaDB."""
    lines = text.split("\n")
    question = None
    answer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        if line.endswith("?"):  # Question detected
            if question:  # Store previous Q&A pair
                add_to_vector_db(question, " ".join(answer).strip())
            question = line
            answer = []
        else:
            answer.append(line)  # Collect answer lines

    if question:  # Store last Q&A pair
        add_to_vector_db(question, " ".join(answer).strip())

def add_to_vector_db(question, answer):
    """Adds question-answer pair to the vector database."""
    embedding = embedding_model.encode(question).tolist()  # Convert to vector
    collection.add(ids=[question], embeddings=[embedding], metadatas=[{"answer": answer}])

def find_best_match(user_query):
    """Finds the closest question using vector search."""
    query_embedding = embedding_model.encode(user_query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    if not results["ids"]:  # No match found
        return None, None

    best_question = results["ids"][0][0]
    best_answer = results["metadatas"][0][0]["answer"]
    return best_question, best_answer

class Bot:
    def __init__(self, document):
        process_document(document)  # Store FAQs in vector database
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

        return full_answer  # Returns the **entire** answer

# Load and preprocess document
file_path = r'c:\Users\cudiamam\Documents\Code\Reedy\FAQ.txt'
document = load_document(file_path)

# Initialize chatbot
chatbot = Bot(document)

# Interactive Chat
print('Reedy: Hello! How may I help you?\nType "exit" to quit.')
while True:
    user_query = input('\nYou: ')
    if user_query.lower() == 'exit':
        break
    response = chatbot.answer_question(user_query)
    print(f"Reedy: {response}")
