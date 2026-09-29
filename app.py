import os

import streamlit as st

from crew import run_research


st.set_page_config(
    page_title="AI Research Multi-Agent",
    page_icon="🔬",
    layout="wide",
)


st.title("🔬 AI Research Multi-Agent")
st.caption(
    "CrewAI + Groq GPT-OSS 120B + Web + Academic + Industry Research"
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("Research Settings")

    research_depth = st.selectbox(
        "Research Depth",
        [
            "Quick",
            "Standard",
            "Deep",
        ],
        index=1,
    )

    st.info(
        "The system uses multiple specialized AI researchers "
        "and combines their findings into one final research report."
    )


# ---------------------------------------------------------
# Main input
# ---------------------------------------------------------

st.subheader("Research Topic")

topic = st.text_area(
    "What do you want the AI research team to investigate?",
    placeholder=(
        "Example: AI-based solutions for reducing food waste "
        "in Pakistan and globally"
    ),
    height=120,
)


research_question = st.text_area(
    "Optional research question",
    placeholder=(
        "Example: What existing solutions already exist, "
        "what are their gaps, and what opportunities remain?"
    ),
    height=100,
)


# ---------------------------------------------------------
# Research button
# ---------------------------------------------------------

if st.button(
    "🚀 Start Multi-Agent Research",
    type="primary",
    use_container_width=True,
):

    if not topic.strip():
        st.warning("Please enter a research topic first.")
        st.stop()

    # -----------------------------------------------------
    # Read Groq API key from Streamlit Secrets
    # -----------------------------------------------------

    groq_key = None

    if "GROQ_API_KEY" in st.secrets:
        groq_key = st.secrets["GROQ_API_KEY"]

    if not groq_key:
        groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        st.error(
            "GROQ_API_KEY is missing. "
            "Add it in Streamlit Cloud → Settings → Secrets."
        )
        st.stop()

    os.environ["GROQ_API_KEY"] = groq_key

    # -----------------------------------------------------
    # Run research
    # -----------------------------------------------------

    with st.status(
        "AI research team is working...",
        expanded=True,
    ) as status:

        st.write("🧠 Research Manager is planning the investigation...")
        st.write("🌐 Web Researcher is searching the web...")
        st.write("🎓 Academic Researcher is checking scholarly evidence...")
        st.write("🏭 Industry Researcher is investigating market solutions...")
        st.write("📝 Synthesizer is preparing the final report...")

        try:

            result = run_research(
                topic=topic.strip(),
                research_question=research_question.strip(),
                research_depth=research_depth,
            )

            status.update(
                label="Research completed successfully!",
                state="complete",
            )

        except Exception as exc:

            status.update(
                label="Research failed",
                state="error",
            )

            st.error("The research workflow encountered an error.")

            st.exception(exc)

            st.stop()

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    st.divider()

    st.subheader("📊 Final Research Report")

    st.markdown(str(result))

    # -----------------------------------------------------
    # Download
    # -----------------------------------------------------

    st.download_button(
        label="⬇️ Download Research Report",
        data=str(result),
        file_name="research_report.txt",
        mime="text/plain",
        use_container_width=True,
    )
