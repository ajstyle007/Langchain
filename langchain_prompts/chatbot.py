from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")

# chat_history = [] # there is one issue with appending a list is that out language model 
# does not understand that which is ai message or which is human message so we will use the 
# langchain messages class which has three methods SystemMessage, HumanMessage, AIMessage

chat_history = [
    SystemMessage(content = "You are a Smart AI Assistant")
]

while True:

    user_input = input("You: ")
    chat_history.append(HumanMessage(content = user_input))

    if user_input == "exit":
        break

    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content = result.content))
    print(result.content)

print(chat_history)
