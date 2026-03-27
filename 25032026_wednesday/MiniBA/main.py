from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from config import llm
from tools import requirements, user_story, tasks
from prompt_loader import prompt_loader
from pdf_loader import load_pdf

def main():
    """
    Main Function that executes agent
    Returns - response
    -------
    """

    tools = [requirements, user_story, tasks]

    prompt = prompt_loader("prompt/agent_workflow.md")
    data = load_pdf("25032026_wednesday/MiniBA/data.pdf")

    user_query = {
        "messages": [
            HumanMessage(
                content = f"""
                    1. Gather Functional and Non Functional Requirements
                    2. Generate User Stories
                    3. Generate Detailed tasks
                        
                    Data : {data}
                 """
            )
        ]
    }

    agent = create_agent(
        model = llm,
        tools = tools,
        system_prompt = prompt,
    )
    response = agent.invoke(user_query)

    return response["messages"][-1].content

if __name__ == "__main__":
    print(main())
