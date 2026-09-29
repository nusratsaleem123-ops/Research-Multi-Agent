from crewai import Agent, LLM


def create_research_manager(llm: LLM) -> Agent:
    """
    Creates the Research Manager agent.

    The manager does not perform the main research itself.
    It creates the research strategy and guides the other agents.
    """

    return Agent(
        role="Senior Research Manager",

        goal=(
            "Plan a rigorous multi-source research investigation. "
            "Break the research question into clear subproblems and "
            "coordinate the work of web, academic, and industry researchers."
        ),

        backstory=(
            "You are a senior research manager experienced in coordinating "
            "multi-disciplinary research teams. You know how to break broad "
            "questions into smaller research tasks, identify evidence "
            "requirements, detect missing information, and maintain research "
            "quality. You never fabricate evidence."
        ),

        llm=llm,

        allow_delegation=False,

        verbose=True,
    )
