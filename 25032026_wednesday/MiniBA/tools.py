from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from prompt_loader import prompt_loader
from config import llm

@tool(
    "RequirementsGenerator",
    description = "Gather functional and non functional requirements from the provided data"
)
def requirements(data: str):
    """
    Gather functional and non-functional requirements from the provided data
    Parameters
    ----------
    data - input data

    Returns - gathered requirements from data
    -------
    """

    try:
        if not data:
            return f"Error Loading Data/PDF"

        prompt_data = prompt_loader("prompt/requirements.md")
        prompt = PromptTemplate.from_template(prompt_data)

        chain = prompt|llm
        response = chain.invoke(
            {
                "content" : data
            }
        )

        return response.content

    except Exception as error:
        return error

@tool(
    "UserStoryGenerator",
    description = "Generates AGILE User Story from the gathered requirements"
)
def user_story(data: str):
    """
    Generates AGILE User Story from the gathered requirements
    Parameters
    ----------
    data - gathered requirements data

    Returns - generated AGILE User Story
    -------
    """

    try:
        if not data:
            return f"Error Loading Requirements"

        prompt_data = prompt_loader("prompt/user_story.md")
        prompt = PromptTemplate.from_template(prompt_data)

        chain = prompt | llm
        response = chain.invoke(
            {
                "requirements": data
            }
        )

        return response.content

    except Exception as error:
        return error

@tool(
    "TaskGenerator",
    description = "Breaks down User Stories into Development Tasks"
)
def tasks(data: str):
    """
    Breaks down User Stories into Development Tasks
    Parameters
    ----------
    data - gathered user stories data

    Returns - generated development tasks
    -------
    """

    try:
        if not data:
            return f"Error Loading User Stories"

        prompt_data = prompt_loader("prompt/tasks.md")
        prompt = PromptTemplate.from_template(prompt_data)

        chain = prompt | llm
        response = chain.invoke(
            {
                "user_stories": data
            }
        )

        return response.content

    except Exception as error:
        return error