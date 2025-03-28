from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task= "text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact 1", description="fact 1 about the topic"),
    ResponseSchema(name="fact 2", description="fact 2 about the topic"),
    ResponseSchema(name="fact 3", description="fact 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template= "Give 3 fact abou the {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction" : parser.get_format_instructions()}
)

# prompt = template.invoke({"topic" : "Light Freezing"})

chain = template | model | parser

# result = model.invoke(prompt)
result = chain.invoke({"topic" : "Light Freezing"})

# final_result = parser.parse(result.content)
# print(final_result)

print(result)