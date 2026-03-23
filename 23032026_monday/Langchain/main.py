import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key = GROQ_API_KEY,
    temperature = 0.3,
    model = "llama-3.1-8b-instant",
    max_tokens = 1000

)

result = llm.invoke("Explain Generative AI ?")
print(f"Output = {result.content}")