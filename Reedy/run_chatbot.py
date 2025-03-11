import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from dotenv import load_dotenv
import os
import requests
from intents import intent_map, detect_intent

load_dotenv()

#OpenAI API
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load embedding model
embedding_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("faq")

# Create LlamaIndex VectorStore
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the index
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embedding_model)

# Initialize GPT-4o Mini Model
class LLM:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name

    def complete(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "ocp-apim-subscription-key": OPENAI_API_KEY,
        }
        payload = {
            "model": self.model_name,
            "messages": [
                #{"role": "system", "content": prompt},
                {"role": "user", "content": prompt},
            ],
            #"temperature": 0.7,    #lower value, more focused. higher values make it more creative
            #"max_tokens": 500      #limits the words/texts in the response
        }

        try:
            response = requests.post(OPENAI_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()  # Raise error if request fails
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            return f"Request Error: {str(e)}"
        except KeyError:
            return "Unexpected response format. Please check the API response."

# Bot Class
class Bot:
    def __init__(self):
        self.index = index
        self.llm = LLM(model_name="gpt-4o-mini")  # Instantiate the LLM class

    def answer_question(self, user_query):
        # Encode the user query into an embedding
        query_embedding = embedding_model.get_text_embedding(user_query)
        detected_intent = detect_intent(user_query)

        if detected_intent:
            context = intent_map[detected_intent]["context"]

            prompt = f"""
            Speak in an informative and friendly way.
            Provide concise and accurate information.

            User's Question: {user_query}
            Context Information: {context}
            """
            
            response = self.llm.complete(prompt)
            return response.strip()

        # Using ChromaDB Direct Query
        search_results = chroma_collection.query(
            query_embeddings=[query_embedding],  # Pass as a list
            n_results=3  # Retrieve top 3 results
        )

        # Extract context from search results
        if search_results["ids"] and search_results["ids"][0]:
            context = "\n".join([
                f"Q: {search_results['ids'][0][i]}\nA: {search_results['metadatas'][0][i]['answer']}"
                for i in range(len(search_results["ids"][0]))
            ])
        else:
            context = "That question is outside my knowledge scope."



        prompt = f"""
        ## Task and Context
        You are an assistant for the employees on Reed Elsevier.
        You handle inquiries about shuttle service request tool, admin and facility service desk, travel request tool, room reservation tool, IT equipment request and deployment, and other common queries.
        Answer the question: {user_query}
        Use the following context to answer the question: {context}

        ## Style Guide
        Speak in an informative and friendly way.
        Provide concise and accurate information.
        If the answer is not found, suggest related topics or ask for clarification.
        For questions outside the scope of context, reply with 'That question is outside my knowledge scope.'
        Avoid using jargon unless necessary, and explain any technical terms used.
        Always ensure the user feels understood and assisted.
        """

        response = self.llm.complete(prompt)
        return response

# Initialize chatbot
chatbot = Bot()

# User Chat
print('Reedy: Hello! How may I help you?\nType "exit" to quit.')
while True:
    user_query = input('\nYou: ')
    if user_query.lower() == 'exit':
        break
    response = chatbot.answer_question(user_query)
    print(f"Reedy: {response}")
