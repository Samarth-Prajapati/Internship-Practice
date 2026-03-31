from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_ADVANCE")

llm = ChatOllama(
    model = OLLAMA_MODEL,
    temperature = 0.3,
)

@tool()
def print_message(message: str) -> str:
    """Print a message to the console."""
    print(f"\n[Agent Tool Execution] : {message}")
    return "Message printed successfully."

agent = create_agent(
    model = llm,
    tools = [print_message],
    system_prompt = "You are a helpful assistant. Use your tools to complete tasks.",
    middleware = [TodoListMiddleware()],
)

user_input = {
    "messages": [
        {
            "role": "user",
            "content": "First, print 'Hello'. Second, print 'Working on it'. Finally, print 'Done'."
        }
    ]
}

response = agent.invoke(user_input)

print("\nFinal Response")
print(response["messages"][-1].content)