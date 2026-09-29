from crewai import Agent, LLM

from tools import academic_search


def create_academic_researcher(llm: LLM) -> Agent:
    """
    Creates the Academic Researcher agent.

    This agent focuses on:
    - Research papers
    - Academic studies
    - Scholarly evidence
    - DOI / publication information
    - Evidence quality
    """

    return Agent(
        role="Academic Research Specialist",

        goal=(
            "Find reliable academic and scholarly evidence related to the "
            "research topic. Identify important studies, research findings, "
            "authors, publication years, and DOI information when available."
        ),

        backstory=(
            "You are an academic research specialist. "
            "You carefully distinguish peer-reviewed research from general "
            "internet content. You never invent papers, authors, findings, "
            "statistics, or citations. When evidence is limited, clearly say so."
        ),

        tools=[academic_search],

        llm=llm,

        allow_delegation=False,

        verbose=True,
    )
