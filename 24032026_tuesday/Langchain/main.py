import os
import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

st.set_page_config(
    page_title = "Chatbot",
    layout = "centered"
)
st.title("Tavily Search Chatbot")
st.write("Search anything on web using Tavily Search")

tavily_client = TavilyClient(
    api_key = TAVILY_API_KEY
)

@tool(
    "SearchEngine",
    description = "Search Anything on Web"
)
def search(
        user_query: str
):
    return tavily_client.search(user_query)

model = ChatGroq(
    api_key = GROQ_API_KEY,
    model = GROQ_MODEL,
    temperature = 0.3
)

llm = model.bind_tools([search], tool_choice = "auto")

query = st.text_input(
    "Enter your query",
    placeholder = "Who is Samarth ?",
)

if query:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    st.session_state.messages.append({"role": "User", "content": query})
    with st.chat_message("User"):
        st.write(query)

    with st.spinner("Thinking"):     # Type: Ignore
        result = llm.invoke(query)

        with st.expander("Metadata"):
            st.write(result)

        if result.tool_calls:
            tool_call = result.tool_calls[0]
            with st.expander("Tool Call"):
                st.write(tool_call)
            search_results = search.invoke(tool_call["args"])
            with st.expander("Search Results"):
                st.write(search_results)

            tool_message = ToolMessage(
                content = str(search_results),
                tool_call_id = tool_call["id"]
            )
            with st.expander("Tool Message"):
                st.write(tool_message)

            final_result = llm.invoke([
                HumanMessage(
                    content = query
                ),
                result,
                tool_message
            ])
            st.session_state.messages.append({"role": "AI", "content": final_result.content})
            with st.chat_message("AI"):
                st.write(final_result.content)

        else:
            st.session_state.messages.append({"role": "AI", "content": result.content})
            with st.chat_message("AI"):
                st.write(result.content)
