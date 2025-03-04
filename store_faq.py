import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
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
    """Adds question-answer pair to the vector database if not already present."""
    existing = collection.get(ids=[question])  # Fetch existing entry

    if existing and existing["ids"]:  # If question already exists, skip it
        print(f"[INFO] Skipping duplicate question: {question}")
        return

    embedding = embedding_model.encode(question).tolist()  # Convert to vector
    collection.add(ids=[question], embeddings=[embedding], metadatas=[{"answer": answer}])
    print(f"[SUCCESS] Added question: {question}")

# Run this only once to store FAQs
file_path = r'c:\Users\cudiamam\Documents\Code\Reedy\FAQ.txt'
document = load_document(file_path)

print("[INFO] Storing FAQ data in ChromaDB...")
process_document(document)
print("[INFO] FAQ data stored successfully!")

#print("[DEBUG] Checking stored questions in ChromaDB...")
#all_data = collection.get()
#print("[DEBUG] Stored questions:", all_data["ids"])
