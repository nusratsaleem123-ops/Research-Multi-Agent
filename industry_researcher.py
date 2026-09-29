from crewai import Agent, LLM

from tools import industry_search


def create_industry_researcher(llm: LLM) -> Agent:
    """
    Creates the Industry Researcher agent.

    Focus:
    - Companies
    - Startups
    - Commercial products
    - Market applications
    - Competitive landscape
    - Industry gaps
    """

    return Agent(
        role="Industry and Market Research Specialist",

        goal=(
            "Investigate existing commercial solutions, companies, startups, "
            "products, business models, market applications, and competitive "
            "gaps related to the research topic."
        ),

        backstory=(
            "You are an industry research analyst who studies companies, "
            "products, startups, markets, and technology adoption. "
            "You verify important claims using available sources and never "
            "invent company information."
        ),

        tools=[industry_search],

        llm=llm,

        allow_delegation=False,

        verbose=True,
    )
