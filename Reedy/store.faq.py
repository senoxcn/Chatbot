import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("faq")

# Create LlamaIndex VectorStore
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load and process the FAQ document
def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def process_document(text):
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
    existing = chroma_collection.get(ids=[question])  # Fetch existing entry

    if existing and existing["ids"]:  # If question already exists, skip it
        print(f"[INFO] Skipping duplicate question: {question}")
        return

    embedding = embedding_model.encode(question).tolist()  # Convert to vector
    formatted_answer = answer.replace("\n", "\\n")  # Ensure line breaks are stored correctly
    chroma_collection.add(ids=[question], embeddings=[embedding], metadatas=[{"answer": formatted_answer}])
    print(f"[SUCCESS] Added question: {question}")

# Run this only once to store FAQs
file_path = r'c:\Users\cudiamam\Documents\Reedy\FAQ.txt'
document = load_document(file_path)

print("[INFO] Storing FAQ data in ChromaDB...")
process_document(document)
print("[INFO] FAQ data stored successfully!")