from langchain_classic.agents import initialize_agent, AgentType
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
    data = load_pdf("data.pdf")

    user_query = f"""
    1. Gather Functional and Non Functional Requirements
    2. Generate User Stories
    3. Generate Detailed tasks
    
    Data : {data}
    """

    agent = initialize_agent(
        llm = llm,
        tools = tools,
        prompt=prompt,
        agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose = True
    )

    return agent.invoke(user_query)

if __name__ == "__main__":
    main()
