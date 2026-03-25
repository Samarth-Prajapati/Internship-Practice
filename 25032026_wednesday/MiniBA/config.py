import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL_1 = os.getenv("GROQ_MODEL_1")

    llm = ChatGroq(
        api_key = GROQ_API_KEY,
        model = GROQ_MODEL_1,
        temperature = 0.3
    )

except Exception as error:
    print(error)