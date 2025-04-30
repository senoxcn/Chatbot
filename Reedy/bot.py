import os
import requests
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.storage_context import StorageContext
from django.conf import settings
from sentence_transformers import SentenceTransformer

# Check if ChromaDB directory exists, if not, populate it
chroma_db_path = os.path.join(settings.BASE_DIR, "chroma_db")
if not os.path.exists(chroma_db_path):
    print("[INFO] ChromaDB not found. Running store_faq to populate the database...")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    chroma_collection = chroma_client.get_or_create_collection("faq")

    faq_file_path = os.path.join(settings.BASE_DIR, "chatbot", "data", "FAQ.txt")

    def load_document(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except FileNotFoundError:
            print(f"[ERROR] FAQ file not found at {file_path}")
            exit(1)

    def process_document(text):
        lines = text.split("\n")
        question = None
        answer = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.endswith("?"):
                if question:
                    add_to_vector_db(question, " ".join(answer).strip())
                question = line
                answer = []
            else:
                answer.append(line)
        if question:
            add_to_vector_db(question, " ".join(answer).strip())

    def add_to_vector_db(question, answer):
        existing = chroma_collection.get(ids=[question])
        if existing and existing["ids"]:
            print(f"[WARNING] Skipping duplicate: {question}")
            return
        embedding = embedding_model.encode(question).tolist()
        chroma_collection.add(ids=[question], embeddings=[embedding], metadatas=[{"answer": answer}])
        print(f"[INFO] Added: {question}")

    document = load_document(faq_file_path)
    if document:
        print("[INFO] Storing FAQ data in ChromaDB...")
        process_document(document)
        print("[INFO] FAQ data stored successfully!")

# Load embedding model for queries
embedding_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
chroma_collection = chroma_client.get_or_create_collection("faq")

# Create LlamaIndex VectorStore
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the index
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embedding_model)

class LLM:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name

    def complete(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "ocp-apim-subscription-key": settings.OPENAI_API_KEY,
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }

        try:
            response = requests.post(settings.OPENAI_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            return f"Request Error: {str(e)}"
        except KeyError:
            return "Unexpected response format. Please check the API response."

class Bot:
    def __init__(self):
        self.index = index
        self.llm = LLM(model_name="gpt-4o-mini")

    def answer_question(self, user_query):
        # Encode the user query into an embedding
        query_embedding = embedding_model.get_text_embedding(user_query)

        # Using ChromaDB Direct Query (faster for FAQs)
        search_results = chroma_collection.query(
            query_embeddings=[query_embedding],  # Pass as a list
            n_results=3,  # Retrieve top 3 results
            include=["distances", "metadatas", "documents"]
        )

        # Extract context from search results
        if search_results["ids"] and search_results["ids"][0]:
            context = "\n".join([
                f"Q: {search_results['ids'][0][i]}\nA: {search_results['metadatas'][0][i]['answer']}"
                for i in range(len(search_results["ids"][0]))
            ])
        else:
            context = "That question is outside my knowledge scope."

        # Calculate confidence scores
        confidence_scores = [1 - dist for dist in search_results["distances"][0]] if search_results["distances"] and search_results["distances"][0] else []

        # Determine if query is ambiguous based on confidence scores
        is_ambiguous = False
        if len(confidence_scores) > 0:
            # If the top result has low confidence or multiple results have similar confidence
            if confidence_scores[0] < 0.2 or (len(confidence_scores) > 1 and confidence_scores[0] - confidence_scores[1] < 0.1):
                is_ambiguous = True

        # Format context for the model prompt
        context = "\n".join([
            f"Q: {search_results['ids'][0][i]}\nA: {search_results['metadatas'][0][i]['answer']}"
            for i in range(len(search_results["ids"][0]))
        ]) if search_results["ids"] and search_results["ids"][0] else "That question is outside my knowledge scope."


        ambiguity_info = "This query appears to be ambiguous or vague. Please ask for clarification or suggest related topics." if is_ambiguous else ""

        prompt = f"""
        ## Task and Context
        You are an assistant for the employees on Reed Elsevier.
        You handle inquiries about shuttle service request tool, admin and facility service desk, travel request tool, room reservation tool, IT equipment request and deployment, and other common queries.
        Answer the question: {user_query}
        Use the following context to answer the question: {context}

        ## Ambiguity Assessment
        {ambiguity_info}
        DO NOT count as ambiguous question if it is simple or common. Make sure that it is still inside the scope.

        ## Style Guide
        Speak in an informative and friendly way.
        Provide concise and accurate information.
        When handling ambiguous queries, DO NOT provide a direct answer. Instead:
            a. Explain that the query could have multiple interpretations
            b. Suggest related 2-3 topics topics based on {context} and {user_query}
            c. Ask for clarification
        Maintain the conversation history.
        For questions outside the scope of {context}, reply with 'That question is outside my knowledge scope.'
        Avoid using jargon unless necessary, and explain any technical terms used.
        Always ensure the user feels understood and assisted.
        """

        response = self.llm.complete(prompt)
        return response

# Initialize the Bot
chatbot = Bot()