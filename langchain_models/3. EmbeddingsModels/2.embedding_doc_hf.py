from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

document = ["CampusX is great channel",
            "Nitish Sir is very good teacher",
            "And I am learning the GENAI"
]

result = embedding.embed_documents(document)  # it will generate the 384 dimension embeddings

print(result)