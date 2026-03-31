from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
import os
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

llm = ChatOllama(
    model = OLLAMA_MODEL,
    temperature = 0.3
)

@tool(
    "your_read_email_tool",
    description = "Function to read an email by its ID."
)
def your_read_email_tool(email_id: str) -> str:
    """Function to read an email by its ID."""
    return f"Email content for ID : {email_id}"

@tool(
    "your_send_email_tool",
    description = "Function to send an email."
)
def your_send_email_tool(recipient: str, subject: str) -> str:
    """Function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent = create_agent(
    model = llm,
    tools = [
        your_read_email_tool,
        your_send_email_tool
    ],
    checkpointer = InMemorySaver(),
    middleware = [
        HumanInTheLoopMiddleware(
            interrupt_on = {
                "your_send_email_tool": {
                    "allowed_decisions": [
                        "approve",
                        "edit",
                        "reject"
                    ],
                },
                "your_read_email_tool": False,
            }
        ),
    ],
)

while True:
    query = input("Enter your query : ")
    if query.lower() in ["exit", "quit"]:
        break

    user_input = {
        "messages": [
            HumanMessage(
                content = query
            )
        ]
    }

    config = {
        "configurable": {
            "thread_id": "thread_1"
        }
    }

    response = agent.invoke(
        user_input,
        config = config
    )

    state = agent.invoke(
        Command(
            resume = {
                "decisions": [
                    {
                        "type": "approve"
                    }
                ]
            }
        ),
        config = config
    )

    print(f"\nResponse : {state["messages"][-1].content}\n")