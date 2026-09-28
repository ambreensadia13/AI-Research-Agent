import os
import re

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from ddgs import DDGS


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM DUCKDUCKGO SEARCH TOOL
# ============================================================

class SearchInput(BaseModel):
    query: str = Field(
        ...,
        description="A focused web search query about the research topic.",
    )


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"

    description: str = (
        "Search the public web using DuckDuckGo. "
        "Use this tool to find current facts, statistics, "
        "official pages, news, research papers, and other "
        "useful information."
    )

    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        """
        Search DuckDuckGo and return formatted search results.
        """

        if not query or not query.strip():
            return "Search query was empty."

        try:
            results = []

            with DDGS() as search:

                search_results = search.text(
                    query.strip(),
                    max_results=6,
                    safesearch="moderate",
                )

                for item in search_results:

                    title = item.get("title", "Untitled")
                    url = item.get("href", "")
                    body = item.get("body", "")

                    results.append(
                        f"TITLE: {title}\n"
                        f"URL: {url}\n"
                        f"SUMMARY: {body}\n"
                    )

            if not results:
                return (
                    "No search results were found. "
                    "Try a different search query."
                )

            return "\n".join(results)

        except Exception as exc:

            return (
                "DuckDuckGo search failed.\n"
                f"Error: {type(exc).__name__}: {exc}"
            )


# ============================================================
# API KEY
# ============================================================

def get_groq_key() -> str:
    """
    Get the Groq API key from Streamlit Secrets
    or environment variables.
    """

    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = ""

    return secret_key or os.getenv("GROQ_API_KEY", "")


# ============================================================
# REPORT CLEANING
# ============================================================

def clean_report(text: str) -> str:
    """
    Remove accidental code fences from the generated report.
    """

    if not text:
        return ""

    text = str(text).strip()

    # Remove opening markdown code fence
    text = re.sub(
        r"^```(?:markdown|md)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove closing code fence
    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


# ============================================================
# BUILD CREWAI RESEARCH CREW
# ============================================================

def build_research_crew(
    topic: str,
    report_length: str,
):
    """
    Create the CrewAI research agent and research task.
    """

    groq_key = get_groq_key()

    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add your Groq API key to Streamlit Secrets."
        )

    # --------------------------------------------------------
    # GROQ LLM
    # --------------------------------------------------------

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_key,
        temperature=0.2,
        reasoning_effort="medium",
        max_tokens=12000,
    )

    # --------------------------------------------------------
    # SEARCH TOOL
    # --------------------------------------------------------

    search_tool = DuckDuckGoSearchTool()

    # --------------------------------------------------------
    # RESEARCH AGENT
    # --------------------------------------------------------

    researcher = Agent(
        role="AI Research Analyst",

        goal=(
            "Research the user's topic carefully using web search "
            "and produce a factual, well-organized research report "
            "with traceable source URLs."
        ),

        backstory=(
            "You are a meticulous research analyst. "
            "You search the web before making factual claims. "
            "You prefer primary and authoritative sources when "
            "available. You compare information across multiple "
            "sources, distinguish facts from opinions, and never "
            "invent citations. You write clearly for a general audience."
        ),

        tools=[search_tool],

        llm=llm,

        allow_delegation=False,

        verbose=False,

        max_iter=10,
    )

    # --------------------------------------------------------
    # RESEARCH TASK
    # --------------------------------------------------------

    task = Task(
        description=f"""
Research the following topic:

{topic}

Your job is to independently research this topic on the public
web and then write a complete research report.

RESEARCH REQUIREMENTS:

1. Search multiple different queries instead of relying on only
   one search.

2. Prefer official documentation, government sources, academic
   papers, reputable organizations, and established publications
   where relevant.

3. Cross-check important factual claims.

4. Include concrete dates when they matter.

5. Do not invent facts, statistics, studies, or URLs.

6. Clearly distinguish established facts from expert interpretation
   or reported claims.

7. Include source URLs directly after important sourced claims.

8. If reliable information is unavailable, say so instead of guessing.

REPORT LENGTH:

{report_length}

USE THIS STRUCTURE:

# Research Report: [Topic]

## Executive Summary

Provide a concise overview of the research.

## Key Findings

Present the most important findings with source URLs.

## Background

Explain the topic clearly for a beginner.

## Detailed Analysis

Break the subject into useful sections.

Use headings and bullet points where appropriate.

## Current Developments

Discuss recent developments when relevant.

Include dates and source URLs.

## Benefits and Opportunities

Describe documented benefits or opportunities
without exaggeration.

## Risks, Limitations, and Challenges

Describe evidence-based limitations,
risks, uncertainties, and challenges.

## Practical Takeaways

Provide useful conclusions based on the research.

## Sources

List the main URLs used in the research.

Put one URL per line.

IMPORTANT RULES:

- Do not claim that you personally visited a webpage unless
  the search tool returned information from that webpage.

- Do not make up sources.

- Do not make up URLs.

- Do not invent statistics.

- Do not invent studies.

- Do not include a separate "AI-generated sources" section.

- Write the final answer in clean Markdown.

- Keep the report factual and understandable.

- When sources disagree, clearly mention the disagreement.

""",

        expected_output=(
            "A complete, factual Markdown research report with "
            "a clear structure, useful analysis, and traceable "
            "source URLs in the Sources section."
        ),

        agent=researcher,
    )

    # --------------------------------------------------------
    # CREW
    # --------------------------------------------------------

    crew = Crew(
        agents=[researcher],

        tasks=[task],

        process=Process.sequential,

        verbose=False,
    )

    return crew


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main title */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
        line-height: 1.2;
    }


    /* Subtitle */

    .subtitle {
        font-size: 18px;
        opacity: 0.8;
        margin-bottom: 25px;
    }


    /* Information box */

    .info-box {
        padding: 18px;
        border-radius: 14px;

        background:
            linear-gradient(
                135deg,
                #0f172a,
                #075985
            );

        border:
            1px solid
            rgba(56, 189, 248, 0.35);

        margin-bottom: 20px;

        color: white;
    }


    /* Buttons */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }


    /* Text area */

    textarea {
        border-radius: 10px !important;
    }


    /* Research report */

    .research-report {
        padding: 25px;
        border-radius: 15px;

        background:
            rgba(15, 23, 42, 0.85);

        border:
            1px solid
            rgba(56, 189, 248, 0.25);
    }


    /* Sidebar */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0f172a,
                #111827
            );
    }


    /* Mobile */

    @media (max-width: 768px) {

        .main-title {
            font-size: 30px;
        }

        .subtitle {
            font-size: 15px;
        }

        .info-box {
            padding: 15px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔎 AI Research Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Research topics using CrewAI + DuckDuckGo + Groq GPT-OSS 120B."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    """
    <div class="info-box">

        <b>How it works</b><br><br>

        1. Enter a research topic.<br>
        2. CrewAI sends the research agent to search the web.<br>
        3. DuckDuckGo provides relevant web results.<br>
        4. The agent analyzes and organizes the information.<br>
        5. Groq GPT-OSS 120B generates the final report.<br>
        6. The report includes source URLs.

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    # --------------------------------------------------------
    # REPORT LENGTH
    # --------------------------------------------------------

    report_length = st.selectbox(
        "Report length",
        [
            "Short — about 600–900 words",
            "Medium — about 1,000–1,500 words",
            "Detailed — about 1,800–2,500 words",
        ],
        index=1,
    )

    st.divider()

    # --------------------------------------------------------
    # EXAMPLE TOPICS
    # --------------------------------------------------------

    st.markdown("### Example Topics")

    examples = [
        "How are AI agents changing software development?",
        "The current state of renewable energy technology",
        "How large language models work",
        "Cybersecurity threats facing small businesses",
        "The impact of AI on education",
        "How AI is changing healthcare",
        "The future of autonomous AI agents",
        "How generative AI affects content creation",
        "The role of AI in cybersecurity",
        "How quantum computing may affect AI",
    ]

    selected_example = st.selectbox(
        "Choose an example",
        ["None"] + examples,
    )

    st.divider()

    # --------------------------------------------------------
    # TECHNOLOGY INFO
    # --------------------------------------------------------

    st.caption(
        "Model: Groq — openai/gpt-oss-120b"
    )

    st.caption(
        "Search: DuckDuckGo"
    )

    st.caption(
        "Framework: CrewAI"
    )


# ============================================================
# TOPIC INPUT
# ============================================================

default_topic = (
    ""
    if selected_example == "None"
    else selected_example
)

topic = st.text_area(
    "Research Topic",
    value=default_topic,
    placeholder=(
        "Example: How will AI agents change web development?"
    ),
    height=120,
)


# ============================================================
# RESEARCH BUTTON
# ============================================================

research_button = st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN RESEARCH
# ============================================================

if research_button:

    # --------------------------------------------------------
    # VALIDATE TOPIC
    # --------------------------------------------------------

    if not topic.strip():

        st.warning(
            "Please enter a research topic first."
        )

        st.stop()

    # --------------------------------------------------------
    # VALIDATE API KEY
    # --------------------------------------------------------

    if not get_groq_key():

        st.error(
            "GROQ_API_KEY is not configured."
        )

        st.info(
            "Add your Groq API key to Streamlit "
            "Secrets and try again."
        )

        st.stop()

    # --------------------------------------------------------
    # RUN CREW
    # --------------------------------------------------------

    try:

        with st.status(
            "Researching your topic...",
            expanded=True,
        ) as status:

            st.write(
                "🔎 Searching the web with DuckDuckGo..."
            )

            crew = build_research_crew(
                topic=topic.strip(),
                report_length=report_length,
            )

            st.write(
                "🧠 Analyzing and organizing the findings..."
            )

            result = crew.kickoff(
                inputs={
                    "topic": topic.strip()
                }
            )

            report = clean_report(
                str(result)
            )

            status.update(
                label="Research completed!",
                state="complete",
                expanded=False,
            )

        # ----------------------------------------------------
        # DISPLAY REPORT
        # ----------------------------------------------------

        st.success(
            "Your research report is ready."
        )

        st.markdown(
            "## 📄 Research Report"
        )

        st.markdown(
            '<div class="research-report">',
            unsafe_allow_html=True,
        )

        st.markdown(report)

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # DOWNLOAD REPORT
        # ----------------------------------------------------

        st.download_button(
            label="⬇️ Download Report",
            data=report,
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------

    except Exception as exc:

        st.error(
            "The research agent could not complete the request."
        )

        st.code(
            f"{type(exc).__name__}: {exc}",
            language="text",
        )

        st.info(
            "If you are deploying on Streamlit Cloud, "
            "make sure GROQ_API_KEY is added under "
            "App Settings → Secrets."
        )


# ============================================================
# INITIAL MESSAGE
# ============================================================

else:

    st.info(
        "Enter a topic above and click "
        "**Start Research** to generate a report."
    )
