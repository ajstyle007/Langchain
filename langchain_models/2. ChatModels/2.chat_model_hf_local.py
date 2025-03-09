from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from dotenv import load_dotenv
import os

os.environ["HF_HOME"] = "D:/langchain/huggingface_cache"

llm = HuggingFacePipeline.from_model_id(
    model_id= "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task= "text-generation",
    pipeline_kwargs = dict(
        temperature = 0.5,
        do_sample = True,
        max_new_tokens = 100
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("who is the PM of India?")

print(result.content)