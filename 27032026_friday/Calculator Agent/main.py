import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_2 = os.getenv("GROQ_MODEL_2")

@tool(
    "Addition",
    description = "This tool returns the addition of all the provided numbers"
)
def add(numbers: list[float]):
    """This tool returns the addition of all the provided numbers"""

    return sum(numbers)

llm = ChatGroq(
    api_key = GROQ_API_KEY,
    model = GROQ_MODEL_2,
    temperature = 0.3
)

agent = create_agent(
    model = llm,
    tools = [add],
    system_prompt = "You have to behave as an expert calculator agent"
)

user_input = input("Enter your query : ")
query = {
    "messages": [HumanMessage(
        content = user_input
    )]
}

response = agent.invoke(query)

print(f"\nResponse : {response["messages"][-1].content}")