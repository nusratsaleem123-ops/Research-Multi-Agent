from crewai import Agent, LLM


def create_synthesizer(llm: LLM) -> Agent:
    """
    Creates the final Research Synthesizer agent.
    """

    return Agent(
        role="Research Synthesis and Report Specialist",

        goal=(
            "Combine findings from web, academic, and industry researchers "
            "into one accurate, structured, evidence-based research report."
        ),

        backstory=(
            "You are a senior research editor and evidence synthesizer. "
            "You compare findings from multiple sources, identify agreements "
            "and disagreements, distinguish facts from interpretation, "
            "identify evidence gaps, and produce clear research reports. "
            "You never create fake citations or unsupported statistics."
        ),

        llm=llm,

        allow_delegation=False,

        verbose=True,
    )
