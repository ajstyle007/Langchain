from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert"),
    ("human", "Now tell me about {topic} briefly")
])

prompt = chat_template.invoke({"domain" : "Hip-Hop", "topic" : "Eminem"})

result = model.invoke(prompt)

print(result.content)
print(prompt)