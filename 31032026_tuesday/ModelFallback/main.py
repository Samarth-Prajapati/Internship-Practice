import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL_ADVANCE")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL_2")

llm = ChatOllama(
    model = OLLAMA_MODEL,
    temperature = 0.3
)

fallback_model = ChatGroq(
    model = GROQ_MODEL,
    api_key = GROQ_API_KEY,
    temperature = 0.3
)

agent = create_agent(
    model = llm,
    checkpointer = MemorySaver(),
    middleware = [
        ModelFallbackMiddleware(fallback_model)
    ]
)

query = {
    "messages": [
        HumanMessage(
            "Who is the Prime Minister of India ?"
        )
    ]
}

config = {
    "configurable": {
        "thread_id": "thread_1"
    }
}

response = agent.invoke(
    query,
    config = config
)
print(response["messages"][-1].content)