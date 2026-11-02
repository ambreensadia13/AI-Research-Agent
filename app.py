import os
import re
import time
import html as html_lib

import streamlit as st


# ============================================================
# IMPORTS
# ============================================================

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool

from pydantic import BaseModel, Field

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
# CSS
# ============================================================

st.markdown(
    """
    <style>

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

    .main .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1,
    h2,
    h3 {
        color: #f8fafc !important;
    }

    p {
        color: #cbd5e1;
    }

    /* HERO */

    .hero {
        padding: 1rem 0 1.2rem 0;
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
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
