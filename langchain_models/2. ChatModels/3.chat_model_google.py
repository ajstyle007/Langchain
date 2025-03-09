from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")

result = model.invoke("Who is the PM of India?")

print(result.content)


# for google api key we have to visit the following website

# https://aistudio.google.com/apikey?_gl=1*2ao1ig*_ga*MTE1OTczNTgzMy4xNzQxMjUwODUx*_ga_P1DBVKWT6V*MTc0MTI1MDg1MC4xLjEuMTc0MTI1MDg1OC41Mi4wLjEzOTkyNTgzMTM.