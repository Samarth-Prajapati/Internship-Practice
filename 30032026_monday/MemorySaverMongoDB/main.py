import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing import TypedDict, List, Annotated
from langgraph.checkpoint.memory import MemorySaver
from langchain_mongodb import MongoDBChatMessageHistory
from langgraph.graph import StateGraph, END, add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

MONGO_URI = os.getenv("MONGO_URI")
SESSION_ID = "session1"
THRESHOLD = 5

llm = ChatOllama(
    model = OLLAMA_MODEL,
    temperature = 0.3
)

checkpointer = MemorySaver()

mongo_history = MongoDBChatMessageHistory(
    connection_string = MONGO_URI,
    session_id = SESSION_ID,
    database_name = "ChatHistory",
    collection_name = "chat_history"
)

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

msg_id = set()

def agent_workflow(state: State) -> State:
    messages = list(state.get(
        "messages",
        []
    ))

    while len(messages) > THRESHOLD:
        oldest = messages.pop(0)

        if oldest.id not in msg_id:
            msg_id.add(oldest.id)

            if isinstance(oldest, HumanMessage):
                mongo_history.add_user_message(oldest.content)

            elif isinstance(oldest, AIMessage):
                mongo_history.add_ai_message(oldest.content)

    long_term = mongo_history.messages

    context = []
    context.extend(long_term)
    context.extend(messages)

    print(f"Context : {context}")
    response = llm.invoke(context)

    messages.append(
        AIMessage(
            content = response.content
        )
    )

    return {
        "messages": messages
    }

graph = StateGraph(State)

graph.add_node("agent_workflow", agent_workflow)
graph.set_entry_point("agent_workflow")
graph.add_edge("agent_workflow", END)

agent = graph.compile(
    checkpointer = checkpointer
)

config = {
    "configurable":
        {
            "thread_id": "1"
        }
}

print("ChatBot with Memory ('exit' to quit)\n")

while True:
    user_input = input("You : ")

    if user_input.lower() == "exit":
        break

    result = agent.invoke(
        {
            "messages":
                [
                    HumanMessage(
                        content = user_input
                    )
                ]
        },
        config = config
    )

    print("AI : ", result["messages"][-1].content)