from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

text = "Muskaan is Love"

result = embedding.embed_query(text)  # it will generate the 384 dimension embeddings

print(result)