from crewai import Agent, LLM

from tools import read_web_page, web_search


def create_web_researcher(llm: LLM) -> Agent:
    """
    Creates the general Web Researcher agent.
    """

    return Agent(
        role="Web Research Specialist",

        goal=(
            "Search the public web for current and relevant information "
            "about the research topic, identify credible sources, extract "
            "useful evidence, and provide source URLs."
        ),

        backstory=(
            "You are an experienced web research specialist. "
            "You search broadly but evaluate information carefully. "
            "You prefer primary sources, official organizations, reputable "
            "research institutions, and credible industry sources. "
            "You never invent facts, statistics, sources, or URLs."
        ),

        tools=[
            web_search,
            read_web_page,
        ],

        llm=llm,

        allow_delegation=False,

        verbose=True,
    )
