from transformers import pipeline
import torch

#Hugging Face - question-answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def load_document(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif file_path.endswith('.pdf'):
        import PyPDF2
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            return text
    elif file_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(file_path)
        return df.to_string()
    else:
        raise ValueError('Unsupported file format')


def preprocess_text(text):
    text = ' '.join(text.split())
    return text

class Bot:
    def __init__(self, document):
        self.document = document

    def answer_question(self, user_query):
        # Use the Hugging Face QA pipeline to find the answer
        result = qa_pipeline(question=user_query, context=self.document)
        return result['answer']


file_path = r'c:\Users\cudiamam\Documents\Code\Reedy\FAQ.pdf'
document = load_document(file_path)
document = preprocess_text(document)  #Clean the text

chatbot = Bot(document)

print('Reedy: Hello! How may I help you?.\nType "exit" to quit.')
while True:
    user_query = input('\nYou: ')
    if user_query.lower() == 'exit':
        break
    response = chatbot.answer_question(user_query)
    print(f"Reedy: {response}")
