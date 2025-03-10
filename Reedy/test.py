import ollama

client = ollama.Client()

model = "llama3.1"
prompt = "What is Python?"

response = client.generate(model=model, prompt=prompt)

print("Response: ")
print(response.response)