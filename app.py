import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from ddgs import DDGS
import os
import re


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# CUSTOM DUCKDUCKGO TOOL
# ============================================================

class SearchInput(BaseModel):
    query: str = Field(
        ...,
        description="A focused web search query about the research topic."
    )


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"
    description: str = (
        "Search the public web with DuckDuckGo. "
        "Use this tool to find current facts, statistics, official pages, "
        "news, research papers, and other useful sources."
    )
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        if not query or not query.strip():
            return "Search query was empty."

        try:
            results = []
            with DDGS() as search:
                for item in search.text(
                    query.strip(),
                    max_results=6,
                    safesearch="moderate",
                ):
                    title = item.get("title", "Untitled")
                    url = item.get("href", "")
                    body = item.get("body", "")

                    results.append(
                        f"TITLE: {title}\n"
                        f"URL: {url}\n"
                        f"SUMMARY: {body}\n"
                    )

            if not results:
                return "No search results were found. Try a different query."

            return "\n".join(results)

        except Exception as exc:
            return (
                "DuckDuckGo search failed. "
                f"Error: {type(exc).__name__}: {exc}"
            )


# ============================================================
# HELPERS
# ============================================================

def get_groq_key() -> str:
    """Read the Groq key from Streamlit Secrets or environment variables."""
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        secret_key = ""

    return secret_key or os.getenv("GROQ_API_KEY", "")


def clean_report(text: str) -> str:
    """Clean a few accidental formatting artifacts from model output."""
    text = text.strip()

    # Remove accidental code fences around the complete report.
    text = re.sub(r"^```(?:markdown|md)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def build_research_crew(topic: str, report_length: str):
    groq_key = get_groq_key()

    if not groq_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets."
        )

    # Groq hosts the OpenAI GPT-OSS 120B model.
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_key,
        temperature=0.2,
        reasoning_effort="medium",
        max_tokens=12000,
    )

    search_tool = DuckDuckGoSearchTool()

    researcher = Agent(
        role="AI Research Analyst",
        goal=(
            "Research the user's topic carefully using web search and "
            "produce a factual, well-organized research report with "
            "traceable source URLs."
        ),
        backstory=(
            "You are a meticulous research analyst. You search the web "
            "before making factual claims, prefer primary and authoritative "
            "sources when available, compare information across multiple "
            "sources, distinguish facts from opinions, and never invent "
            "citations. You write for a general audience."
        ),
        tools=[search_tool],
        llm=llm,
        allow_delegation=False,
        verbose=False,
        max_iter=10,
    )

    task = Task(
        description=f"""
Research the following topic:

{topic}

Your job is to independently research this topic on the public web and
then write a complete research report.

Research requirements:
1. Search multiple different queries instead of relying on one search.
2. Prefer official documentation, government sources, academic papers,
   reputable organizations, and established publications where relevant.
3. Cross-check important factual claims.
4. Include concrete dates when they matter.
5. Do not invent facts, statistics, studies, or URLs.
6. Clearly distinguish established facts from expert interpretation or
   reported claims.
7. Include the source URL directly after important sourced claims.
8. If reliable information is unavailable, say so instead of guessing.

Report length:
{report_length}

Use this structure:

# Research Report: [Topic]

## Executive Summary
A concise overview.

## Key Findings
The most important findings with source URLs.

## Background
Explain the topic clearly for a beginner.

## Detailed Analysis
Break the subject into useful sections. Use headings and bullet points
where appropriate.

## Current Developments
Discuss recent developments when relevant. Include dates and source URLs.

## Benefits and Opportunities
Describe documented benefits or opportunities without exaggeration.

## Risks, Limitations, and Challenges
Describe evidence-based limitations and uncertainties.

## Practical Takeaways
Give useful conclusions based on the research.

## Sources
List the main URLs used in the research, one per line.

Important:
- Do not claim that you personally visited a page unless the search tool
  returned it.
- Do not make up a source.
- Do not include a separate "AI-generated sources" section.
- Write the final answer in clean Markdown.
""",
        expected_output=(
            "A complete, factual Markdown research report with a clear "
            "structure and traceable URLs in the Sources section."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    return crew


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            opacity: 0.8;
            margin-bottom: 25px;
        }

        .info-box {
            padding: 18px;
            border-radius: 14px;
            background: linear-gradient(135deg, #0f172a, #075985);
            border: 1px solid rgba(56, 189, 248, 0.35);
            margin-bottom: 20px;
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 700;
        }

        textarea {
            border-radius: 10px !important;
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
    "Research a topic with CrewAI + DuckDuckGo + Groq GPT-OSS 120B."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
        <b>How it works</b><br>
        1. Enter a research topic.<br>
        2. The CrewAI research agent searches DuckDuckGo.<br>
        3. The agent checks and organizes the information.<br>
        4. Groq GPT-OSS 120B writes the final report with source URLs.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

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

    st.markdown("### Example topics")
    examples = [
        "How are AI agents changing software development?",
        "The current state of renewable energy technology",
        "How large language models work",
        "Cybersecurity threats facing small businesses",
        "The impact of AI on education",
    ]

    selected_example = st.selectbox(
        "Choose an example",
        ["None"] + examples,
    )

    st.divider()

    st.caption("Model: Groq — openai/gpt-oss-120b")
    st.caption("Search: DuckDuckGo")
    st.caption("Framework: CrewAI")


# ============================================================
# INPUT
# ============================================================

default_topic = "" if selected_example == "None" else selected_example

topic = st.text_area(
    "Research topic",
    value=default_topic,
    placeholder="Example: How will AI agents change web development?",
    height=120,
)

research_button = st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True,
)


# ============================================================
# RUN RESEARCH
# ============================================================

if research_button:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
        st.stop()

    if not get_groq_key():
        st.error(
            "GROQ_API_KEY is not configured. "
            "Add your Groq API key to Streamlit Secrets and try again."
        )
        st.stop()

    try:
        with st.status("Researching your topic...", expanded=True) as status:
            st.write("🔎 Searching the web with DuckDuckGo...")
            crew = build_research_crew(topic.strip(), report_length)

            st.write("🧠 Analyzing and organizing the findings...")
            result = crew.kickoff(
                inputs={"topic": topic.strip()}
            )

            report = clean_report(str(result))

            status.update(
                label="Research completed!",
                state="complete",
                expanded=False,
            )

        st.success("Your research report is ready.")

        st.markdown("## 📄 Research Report")
        st.markdown(report)

        st.download_button(
            label="⬇️ Download Report",
            data=report,
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    except Exception as exc:
        st.error("The research agent could not complete the request.")
        st.code(
            f"{type(exc).__name__}: {exc}",
            language="text",
        )
        st.info(
            "If you are deploying on Streamlit Cloud, first check that "
            "GROQ_API_KEY is present in App Settings → Secrets."
        )
else:
    st.info(
        "Enter a topic above and click **Start Research** to generate a report."
    )
