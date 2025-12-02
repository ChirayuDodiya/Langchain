# from langchain_huggingface import HuggingFaceEmbeddings

# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# text = "Delhi is the capital of India"

# vector = embedding.embed_query(text)

# print(str(vector))

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

document = ["Delhi is the capital of India",
    "Mumbai is the capital of Maharashtra",
    "Chennai is the capital of Tamil Nadu"]

vector = embedding.embed_documents(document)

print(str(vector))