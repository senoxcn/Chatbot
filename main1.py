from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from ctransformers import AutoModelForCausalLM
import PyPDF2
import os

def load_document(file_path):
    """Load the document"""
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''.join(page.extract_text() for page in reader.pages if page.extract_text())
            return text
    else:
        raise ValueError('Unsupported file format')


def preprocess_text(text):
    """Clean and preprocess text by removing extra spaces."""
    return ' '.join(text.split())

#file paths
file_path = r'c:\Users\cudiamam\Documents\Code\Reedy\FAQ.pdf'
model_path = r'C:\Users\cudiamam\Documents\Code\Reedy\capybarahermes-2.5-mistral-7b.Q4_K_M.gguf'


class Bot:
    def __init__(self, document, model_path):
        self.document = document
        self.index = self.build_index(model_path)

    def build_index(self, model_path):
        """Build a vector store index using LlamaIndex."""
        with open("temp_document.txt", "w", encoding="utf-8") as f:
            f.write(self.document)

        reader = SimpleDirectoryReader(input_files=["temp_document.txt"])
        documents = reader.load_data()

        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_file=model_path,  # `model_file` for GGUF models
            model_type="mistral",  
            temperature=0.7,
            max_new_tokens=256
        )


        Settings.llm = llm

        index = VectorStoreIndex.from_documents(documents)
        return index

    def answer_question(self, user_query):
        """Answer user questions using the indexed document."""
        query_engine = self.index.as_query_engine()
        response = query_engine.query(user_query)
        return response.response  # Extract the answer text

document = load_document(file_path)
document = preprocess_text(document)

chatbot = Bot(document, model_path)

print('Reedy: Hello! How may I help you?\nType "exit" to quit.')
while True:
    user_query = input('\nYou: ')
    if user_query.lower() == 'exit':
        break
    response = chatbot.answer_question(user_query)
    print(f"Reedy: {response}")
