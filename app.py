import os
import re
import time
import html as html_lib

import streamlit as st


# ============================================================
# CREWAI CACHE BREAKPOINT FIX
# IMPORTANT:
# This must run BEFORE importing CrewAI.
# ============================================================

try:
    import crewai.llms.cache as _crewai_cache

    _crewai_cache.mark_cache_breakpoint = lambda message: message

except Exception:
    pass


# ============================================================
# CREWAI IMPORTS
# ============================================================

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool


# ============================================================
# DUCKDUCKGO
# ============================================================

from ddgs import DDGS


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN PAGE
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at top left,
                rgba(56, 189, 248, 0.10),
                transparent 35%
            ),
            radial-gradient(
                circle at top right,
                rgba(139, 92, 246, 0.10),
                transparent 35%
            ),
            #0f172a;
        color: #f8fafc;
    }


    /* ========================================================
       MAIN CONTENT
       ======================================================== */

    .main .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       HEADINGS
       ======================================================== */

    h1 {
        color: #f8fafc !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    h2, h3 {
        color: #f8fafc !important;
    }

    p {
        color: #cbd5e1;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        padding: 1.5rem 0 1rem 0;
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        color: #f8fafc;
    }

    .hero-title span {
        background: linear-gradient(
            90deg,
            #38bdf8,
            #8b5cf6
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 800px;
    }


    /* ========================================================
       INFO BOX
       ======================================================== */

    .info-box {
        background: linear-gradient(
            135deg,
            rgba(7, 89, 133, 0.90),
            rgba(30, 41, 59, 0.96)
        );

        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin: 1.2rem 0 1.8rem 0;
        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.20);
    }

    .info-title {
        color: #ffffff;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .info-step {
        color: #e2e8f0;
        font-size: 0.96rem;
        line-height: 1.7;
        margin: 0.25rem 0;
    }


    /* ========================================================
       INPUT
       ======================================================== */

    .stTextInput label {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }

    .stTextInput input {
        background-color: #111827 !important;
        color: #ffffff !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }

    .stTextInput input:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px #38bdf8 !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 0.65rem 1rem;
        font-weight: 700;
        color: white;
        background: linear-gradient(
            90deg,
            #0284c7,
            #7c3aed
        );
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow:
            0 8px 20px rgba(56, 189, 248, 0.20);
    }


    /* ========================================================
       REPORT CARD
       ======================================================== */

    .report-card {
        background: #ffffff;
        color: #111827;
        border-radius: 16px;
        padding: 1.6rem;
        margin-top: 1.2rem;
        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.25);
        line-height: 1.75;
    }

    .report-card h1,
    .report-card h2,
    .report-card h3,
    .report-card h4 {
        color: #111827 !important;
    }

    .report-card p,
    .report-card li {
        color: #1f2937 !important;
    }


    /* ========================================================
       SOURCE CARD
       ======================================================== */

    .source-card {
        background: #111827;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
    }

    .source-title {
        color: #f8fafc;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    .source-url {
        color: #38bdf8;
        font-size: 0.85rem;
        word-break: break-word;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #111827;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status-text {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin-top: 0.5rem;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {

        .main .block-container {
            padding: 1rem;
        }

        .hero-title {
            font-size: 2rem;
        }

        .hero-subtitle {
            font-size: 0.95rem;
        }

        .info-box {
            padding: 1rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "report" not in st.session_state:
    st.session_state.report = ""

if "sources" not in st.session_state:
    st.session_state.sources = []

if "last_topic" not in st.session_state:
    st.session_state.last_topic = ""


# ============================================================
# GROQ API KEY
# ============================================================

def get_groq_key():
    """
    Get Groq API key from Streamlit Secrets.
    """

    try:
        key = st.secrets.get("GROQ_API_KEY")

        if key:
            return key.strip()

    except Exception:
        pass

    key = os.getenv("GROQ_API_KEY")

    if key:
        return key.strip()

    return None


# ============================================================
# DUCKDUCKGO SEARCH TOOL
# ============================================================

class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="The web search query."
    )


class DuckDuckGoSearchTool(BaseTool):

    name: str = "DuckDuckGo Web Search"

    description: str = (
        "Search the public web using DuckDuckGo. "
        "Use this tool to find current and relevant information."
    )

    args_schema: type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str) -> str:

        results = []

        try:

            with DDGS() as ddgs:

                search_results = ddgs.text(
                    query,
                    max_results=5
                )

                for item in search_results:

                    title = item.get("title", "")
                    body = item.get("body", "")
                    href = item.get("href", "")

                    if not title and not body:
                        continue

                    results.append(
                        f"TITLE: {title}\n"
                        f"SUMMARY: {body}\n"
                        f"URL: {href}"
                    )

        except Exception as e:

            return (
                "DuckDuckGo search failed. "
                f"Error: {str(e)}"
            )

        if not results:
            return "No useful web results were found."

        return "\n\n".join(results)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_report(text):
    """
    Clean unwanted HTML and Markdown artifacts
    from the final report.
    """

    if not text:
        return ""

    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode HTML entities
    text = html_lib.unescape(text)

    # Remove code fences
    text = re.sub(r"```(?:markdown|text)?", "", text)
    text = text.replace("```", "")

    # Remove excessive markdown emphasis
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)

    # Remove horizontal rules
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove repeated blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# CLEAN URL
# ============================================================

def clean_url(url):
    if not url:
        return ""

    return str(url).strip()


# ============================================================
# BUILD RESEARCH CREW
# ============================================================

def build_research_crew(topic):

    groq_key = get_groq_key()

    if not groq_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found. "
            "Add GROQ_API_KEY to Streamlit Secrets."
        )

    # --------------------------------------------------------
    # GROQ LLM
    # --------------------------------------------------------

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_key,

        # Keep token usage below the free/on-demand TPM limit.
        max_tokens=2500,

        # Lower reasoning reduces unnecessary token consumption.
        reasoning_effort="low",

        temperature=0.2,
    )

    # --------------------------------------------------------
    # SEARCH TOOL
    # --------------------------------------------------------

    search_tool = DuckDuckGoSearchTool()

    # --------------------------------------------------------
    # RESEARCH AGENT
    # --------------------------------------------------------

    researcher = Agent(
        role="Web Researcher",

        goal=(
            "Research the user's topic using reliable public web "
            "sources and identify the most relevant factual information."
        ),

        backstory=(
            "You are a careful web researcher. "
            "You search the public web, compare information, "
            "and focus on useful and relevant facts."
        ),

        tools=[search_tool],

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )

    # --------------------------------------------------------
    # WRITER AGENT
    # --------------------------------------------------------

    writer = Agent(
        role="Research Report Writer",

        goal=(
            "Turn the research findings into a concise, clear, "
            "well-structured research report."
        ),

        backstory=(
            "You are a professional research writer. "
            "You summarize information clearly without unnecessary "
            "repetition or unsupported claims."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )

    # --------------------------------------------------------
    # RESEARCH TASK
    # --------------------------------------------------------

    research_task = Task(

        description=f"""
Research the following topic:

{topic}

Use DuckDuckGo to find relevant public web information.

Focus on:
- Important facts
- Recent information when available
- Relevant statistics or dates
- Different perspectives when appropriate
- Reliable and useful sources

Do not produce an unnecessarily long response.

Return concise research notes containing:
1. Key findings
2. Important facts
3. Useful source URLs

Topic:
{topic}
""",

        expected_output=(
            "Concise research notes with key factual findings "
            "and useful source URLs."
        ),

        agent=researcher,
    )

    # --------------------------------------------------------
    # REPORT TASK
    # --------------------------------------------------------

    report_task = Task(

        description=f"""
Using the research findings, write a concise report about:

{topic}

The report should contain:

Title

Overview

Key Findings

Important Details

Conclusion

Sources

Requirements:

- Use clear and simple language.
- Focus only on information relevant to the topic.
- Do not invent facts.
- Do not use HTML tags.
- Do not use code fences.
- Avoid excessive Markdown formatting.
- Do not repeat the same information.
- Keep the report reasonably concise.
- Include source URLs at the end when available.

Do not create a very long report.
""",

        expected_output=(
            "A clean, concise research report with a title, "
            "overview, key findings, important details, conclusion, "
            "and source URLs."
        ),

        agent=writer,

        context=[research_task],
    )

    # --------------------------------------------------------
    # CREW
    # --------------------------------------------------------

    crew = Crew(

        agents=[
            researcher,
            writer
        ],

        tasks=[
            research_task,
            report_task
        ],

        process=Process.sequential,

        verbose=False,
    )

    return crew


# ============================================================
# EXTRACT SOURCES
# ============================================================

def extract_sources(text):

    if not text:
        return []

    urls = re.findall(
        r"https?://[^\s<>\]\)]+",
        text
    )

    cleaned = []

    for url in urls:

        url = url.rstrip(
            ".,;:!?)]}"
        )

        if url not in cleaned:
            cleaned.append(url)

    return cleaned[:10]


# ============================================================
# RATE LIMIT MESSAGE
# ============================================================

def friendly_error(error):

    error_text = str(error)

    lower = error_text.lower()

    if (
        "rate limit" in lower
        or "rate_limit" in lower
        or "tokens per minute" in lower
        or "tpm" in lower
    ):

        wait_match = re.search(
            r"try again in ([0-9.]+)s",
            error_text,
            flags=re.IGNORECASE
        )

        if wait_match:

            seconds = float(wait_match.group(1))

            return (
                "Groq's token-per-minute limit was reached. "
                f"Please wait about {round(seconds)} seconds and try again."
            )

        return (
            "Groq's token-per-minute limit was reached. "
            "Please wait a short time and try again."
        )

    if "api key" in lower:

        return (
            "The Groq API key could not be authenticated. "
            "Please check GROQ_API_KEY in Streamlit Secrets."
        )

    return (
        "The research agent could not complete the request.\n\n"
        f"{error_text}"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚙️ Settings")

    st.markdown(
        "### Research Engine"
    )

    st.info(
        "DuckDuckGo is used to search the public web."
    )

    st.markdown(
        "### AI Model"
    )

    st.info(
        "Groq GPT-OSS 120B"
    )

    st.markdown(
        "### Output"
    )

    st.info(
        "Concise research report with sources"
    )

    st.markdown("---")

    st.markdown(
        "### 🔐 API Key"
    )

    if get_groq_key():

        st.success(
            "Groq API key detected."
        )

    else:

        st.error(
            "Groq API key not found."
        )

        st.caption(
            "Add GROQ_API_KEY to Streamlit Secrets."
        )


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🔎 <span>AI Research Agent</span>
        </div>

        <div class="hero-subtitle">
            Research any topic using DuckDuckGo and generate
            a clear AI-powered research report with sources.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HOW IT WORKS
# ============================================================

st.markdown(
    '<div class="info-box">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-title">💡 How It Works</div>',
    unsafe_allow_html=True
)

steps = [
    "Enter a research topic.",
    "CrewAI creates the research task.",
    "DuckDuckGo searches the public web.",
    "The research agent analyzes the information.",
    "Groq GPT-OSS 120B generates the report.",
    "Sources are included in the final report.",
]

for number, step in enumerate(steps, start=1):

    st.markdown(
        f'<div class="info-step">{number}. {step}</div>',
        unsafe_allow_html=True
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# TOPIC INPUT
# ============================================================

st.markdown("### 🔍 Research Topic")

topic = st.text_input(
    "Enter your research topic",
    placeholder=(
        "Example: Artificial Intelligence in Education"
    ),
    label_visibility="collapsed",
)


# ============================================================
# RESEARCH BUTTON
# ============================================================

research_button = st.button(
    "🚀 Start Research",
    use_container_width=True,
)


# ============================================================
# RUN RESEARCH
# ============================================================

if research_button:

    if not topic.strip():

        st.warning(
            "Please enter a research topic first."
        )

        st.stop()

    if not get_groq_key():

        st.error(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Secrets and redeploy."
        )

        st.stop()

    topic = topic.strip()

    st.session_state.report = ""
    st.session_state.sources = []
    st.session_state.last_topic = topic

    progress_box = st.empty()

    try:

        # ----------------------------------------------------
        # STATUS 1
        # ----------------------------------------------------

        progress_box.info(
            "🔎 Searching the web with DuckDuckGo..."
        )

        time.sleep(0.3)

        # ----------------------------------------------------
        # BUILD CREW
        # ----------------------------------------------------

        crew = build_research_crew(topic)

        # ----------------------------------------------------
        # STATUS 2
        # ----------------------------------------------------

        progress_box.info(
            "🧠 Researching and analyzing the information..."
        )

        # ----------------------------------------------------
        # RUN CREW
        # ----------------------------------------------------

        result = crew.kickoff()

        # ----------------------------------------------------
        # GET RESULT
        # ----------------------------------------------------

        raw_report = ""

        if hasattr(result, "raw"):

            raw_report = result.raw

        else:

            raw_report = str(result)

        # ----------------------------------------------------
        # CLEAN REPORT
        # ----------------------------------------------------

        report = clean_report(raw_report)

        # ----------------------------------------------------
        # EXTRACT SOURCES
        # ----------------------------------------------------

        sources = extract_sources(raw_report)

        st.session_state.report = report
        st.session_state.sources = sources

        progress_box.success(
            "✅ Research completed successfully."
        )

    except Exception as e:

        progress_box.empty()

        st.error(
            friendly_error(e)
        )


# ============================================================
# DISPLAY REPORT
# ============================================================

if st.session_state.report:

    st.markdown("---")

    st.markdown(
        f"### 📄 Research Report"
    )

    st.caption(
        f"Topic: {st.session_state.last_topic}"
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    st.markdown(
        '<div class="report-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        st.session_state.report
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # SOURCES
    # --------------------------------------------------------

    if st.session_state.sources:

        st.markdown("### 🔗 Sources")

        for index, source in enumerate(
            st.session_state.sources,
            start=1
        ):

            safe_source = html_lib.escape(source)

            st.markdown(
                f"""
                <div class="source-card">

                    <div class="source-title">
                        Source {index}
                    </div>

                    <div class="source-url">
                        {safe_source}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# EMPTY STATE
# ============================================================

elif not research_button:

    st.markdown("### 📚 Try a research topic")

    example_topics = [
        "Artificial Intelligence in Education",
        "Generative AI and the Future of Software Development",
        "Cybersecurity threats in 2026",
        "Impact of AI on healthcare",
        "Future of renewable energy",
    ]

    for example in example_topics:

        st.markdown(
            f"• {example}"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Research Agent • CrewAI • DuckDuckGo • Groq GPT-OSS 120B"
)
