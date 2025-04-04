from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= "text-generation"
)

model = ChatHuggingFace(llm = llm)

result = model.invoke("What is the meaning of love?")

print(result.content)


# from langchain.llms import HuggingFaceHub
# from dotenv import load_dotenv

# load_dotenv()

# llm = HuggingFaceHub(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  
#     model_kwargs={"temperature": 0.7, "max_length": 100},
# )

# result = llm.invoke("What is the capital of India?")

# print(result)