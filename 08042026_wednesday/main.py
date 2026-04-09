from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

llm = ChatOllama(
    model = "llama3.2:latest",
    temperature = 0.3,
    base_url = OLLAMA_BASE_URL,
)

response = llm.invoke(
    [
        HumanMessage(
            content = "Explain transformers in simple terms"
        )
    ]
)

print(response.content)