from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task= "text-generation"
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 questions answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes" : prompt1 | model1 | parser,
    "quiz" : prompt2 | model2 | parser
})

merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

text = """
The gym is a place where people come to build strength, improve endurance,
 and maintain a healthy lifestyle. It is equipped with various machines, free weights,
   and cardio equipment to support different fitness goals. Many gyms also offer group classes,
     such as yoga, spinning, and high-intensity interval training, creating a motivating 
     environment for members. Regular exercise not only enhances physical health but also 
     boosts mental well-being by reducing stress and increasing energy levels. Whether 
     someone is lifting weights, running on a treadmill, or stretching after a workout, 
     the gym provides a space for self-improvement and discipline.

     The gym is more than just a place to work out—it’s a space for self-discipline,
       growth, and motivation. Whether lifting weights, running on the treadmill, or joining
         a group fitness class, every visit is a step toward better health. The sound of 
         weights clanking, upbeat music, and the determination in people’s eyes create an 
         atmosphere of energy and focus. Beyond physical fitness, the gym also promotes mental
           strength, relieving stress and boosting confidence. It’s a place where goals are set, 
           challenges are overcome, and progress is made—one workout at a time.
"""

result = chain.invoke({"text" : text})

print(result)

chain.get_graph().print_ascii()