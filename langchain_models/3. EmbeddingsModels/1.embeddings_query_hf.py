from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "Muskaan is Love"

result = embedding.embed_query(text)  # it will generate the 384 dimension embeddings

print(result)