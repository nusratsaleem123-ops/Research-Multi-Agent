import os

from crewai import Crew, LLM, Process, Task

from academic_researcher import create_academic_researcher
from industry_researcher import create_industry_researcher
from research_manager import create_research_manager
from synthesizer import create_synthesizer
from web_researcher import create_web_researcher


MODEL_NAME = "groq/openai/gpt-oss-120b"


def create_llm() -> LLM:
    """
    Create the Groq LLM used by all research agents.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing."
        )

    return LLM(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.1,
    )


def run_research(
    topic: str,
    research_question: str = "",
    research_depth: str = "Standard",
):

    llm = create_llm()

    # -----------------------------------------------------
    # Create Agents
    # -----------------------------------------------------

    manager = create_research_manager(llm)

    web_researcher = create_web_researcher(llm)

    academic_researcher = create_academic_researcher(llm)

    industry_researcher = create_industry_researcher(llm)

    synthesizer = create_synthesizer(llm)

    # -----------------------------------------------------
    # Task 1 - Research Planning
    # -----------------------------------------------------

    planning_task = Task(
        description=f"""
Create a structured research plan for the following topic:

TOPIC:
{topic}

RESEARCH QUESTION:
{research_question or "No additional question was provided."}

RESEARCH DEPTH:
{research_depth}

Your plan must identify:

1. Main research questions
2. Important subtopics
3. What evidence should be collected
4. What sources should be considered
5. What the Web Researcher should investigate
6. What the Academic Researcher should investigate
7. What the Industry Researcher should investigate
8. What should be compared
9. What gaps should be identified

Do not answer the research yet.
Create a clear investigation plan for the other researchers.
""",

        expected_output=(
            "A structured research plan containing research questions, "
            "subtopics, evidence requirements, and instructions for each "
            "specialized researcher."
        ),

        agent=manager,
    )

    # -----------------------------------------------------
    # Task 2 - Web Research
    # -----------------------------------------------------

    web_task = Task(
        description=f"""
Research the following topic using the available web research tools:

TOPIC:
{topic}

RESEARCH QUESTION:
{research_question or "No additional question was provided."}

Use the research plan provided by the Research Manager.

Investigate:

- Existing solutions
- Existing companies
- Existing products
- Relevant organizations
- Current developments
- Important statistics
- Recent news where relevant
- Publicly available evidence
- Problems and gaps

IMPORTANT:

Do not invent facts.

For important claims, provide the source title and URL.

Clearly separate:
- Verified information
- Claims made by sources
- Your interpretation
""",

        expected_output=(
            "A structured web research report with findings, source titles, "
            "URLs, evidence, and identified gaps."
        ),

        agent=web_researcher,

        context=[planning_task],
    )

    # -----------------------------------------------------
    # Task 3 - Academic Research
    # -----------------------------------------------------

    academic_task = Task(
        description=f"""
Conduct academic research on:

TOPIC:
{topic}

RESEARCH QUESTION:
{research_question or "No additional question was provided."}

Follow the research plan from the Research Manager.

Find relevant academic evidence.

Focus on:

- Research papers
- Peer-reviewed studies where available
- Academic findings
- Researchers/authors
- Publication years
- DOI information
- Research conclusions
- Limitations
- Evidence that supports or challenges important claims

Do not invent citations.

If a paper cannot be verified, do not present it as a verified paper.
""",

        expected_output=(
            "A structured academic research report containing verified "
            "academic sources, findings, publication information, and limitations."
        ),

        agent=academic_researcher,

        context=[planning_task],
    )

    # -----------------------------------------------------
    # Task 4 - Industry Research
    # -----------------------------------------------------

    industry_task = Task(
        description=f"""
Investigate the industry and market side of this topic:

TOPIC:
{topic}

RESEARCH QUESTION:
{research_question or "No additional question was provided."}

Use the Research Manager's plan.

Investigate:

- Existing startups
- Existing companies
- Commercial products
- Industry solutions
- Market applications
- Business models
- Target customers
- Competitive gaps
- Technology adoption
- Geographic differences
- Opportunities for new solutions

Do not invent company information.

For important claims, provide source titles and URLs.
""",

        expected_output=(
            "A structured industry research report covering existing "
            "commercial solutions, companies, market applications, "
            "competitive gaps, and opportunities."
        ),

        agent=industry_researcher,

        context=[planning_task],
    )

    # -----------------------------------------------------
    # Task 5 - Synthesis
    # -----------------------------------------------------

    synthesis_task = Task(
        description=f"""
Create the final research report for:

TOPIC:
{topic}

RESEARCH QUESTION:
{research_question or "No additional question was provided."}

Combine the findings from:

1. Research Manager
2. Web Researcher
3. Academic Researcher
4. Industry Researcher

The final report should contain:

# Executive Summary

# Research Question

# Existing Solutions

# Academic Evidence

# Industry / Market Landscape

# Important Findings

# Problems and Gaps

# Opportunities

# Evidence Quality / Limitations

# Sources

For every important factual claim, identify the supporting source.

Do not invent statistics, companies, papers, citations, or URLs.

Clearly distinguish facts from interpretation.

The final report should be useful for someone making a research,
startup, product, or hackathon decision.
""",

        expected_output=(
            "A comprehensive but readable final research report with "
            "evidence, source references, existing solutions, gaps, "
            "limitations, and opportunities."
        ),

        agent=synthesizer,

        context=[
            planning_task,
            web_task,
            academic_task,
            industry_task,
        ],
    )

    # -----------------------------------------------------
    # Create Crew
    # -----------------------------------------------------

    crew = Crew(
        agents=[
            manager,
            web_researcher,
            academic_researcher,
            industry_researcher,
            synthesizer,
        ],

        tasks=[
            planning_task,
            web_task,
            academic_task,
            industry_task,
            synthesis_task,
        ],

        process=Process.sequential,

        verbose=True,
    )

    # -----------------------------------------------------
    # Run Crew
    # -----------------------------------------------------

    result = crew.kickoff()

    return result
