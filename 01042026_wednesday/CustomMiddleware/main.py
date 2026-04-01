import os
import time
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_ollama import ChatOllama

load_dotenv()

OLLAMA_MODEL_ADVANCE = os.getenv("OLLAMA_MODEL_ADVANCE")

model = ChatOllama(
    model = OLLAMA_MODEL_ADVANCE,
)

@wrap_model_call
def timing_middleware(request, handler):
    print("\nBefore model call ")
    start = time.time()

    # handler returns a ModelResponse object
    response = handler(request)
    end = time.time()

    print("After model call")
    print(f"Execution time : {end - start:.2f} s")
    if response.result and len(response.result) > 0:
        response.result[0].content += "\n\n[Processed through middleware]"

    return response

agent = create_agent(
    model = model,
    middleware = [timing_middleware]
)

user_input = {
    "messages": [
        {
            "role": "user",
            "content": "Explain what is machine learning in simple words"
        }
    ]
}

result = agent.invoke(user_input)

for msg in result["messages"]:
    print(f"\n{msg.type.upper()} : {msg.content}")