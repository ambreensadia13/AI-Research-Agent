# 🔎 AI Research Agent

A beginner-friendly AI Research Agent built with:

- CrewAI
- Groq
- OpenAI GPT-OSS 120B
- DuckDuckGo search
- Streamlit
- GitHub
- Streamlit Community Cloud

The app uses **one CrewAI agent**. The agent searches the web with DuckDuckGo,
researches the requested topic, checks the information, and writes a structured
Markdown report with source URLs.

## Project structure

```text
ai-research-agent/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── .gitignore
├── README.md
│
└── .streamlit/
    └── secrets.toml.example
```

## 1. Get a Groq API key

Create a Groq API key from the Groq console.

Do not put your real API key inside `app.py`.

The application reads:

```text
GROQ_API_KEY
```

from Streamlit Secrets.

## 2. Run locally

Use Python 3.12.

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create:

```text
.streamlit/secrets.toml
```

Put your key inside:

```toml
GROQ_API_KEY = "your-real-groq-api-key"
```

Then start Streamlit:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

## 3. Upload to GitHub

Create a new GitHub repository, for example:

```text
ai-research-agent
```

Upload:

```text
app.py
requirements.txt
runtime.txt
.gitignore
README.md
.streamlit/secrets.toml.example
```

Do NOT upload:

```text
.streamlit/secrets.toml
```

Your `.gitignore` already protects it.

## 4. Deploy to Streamlit Community Cloud

1. Open Streamlit Community Cloud.
2. Sign in with GitHub.
3. Click **Create app**.
4. Select your GitHub repository.
5. Select the `main` branch.
6. Set the main file to:

```text
app.py
```

7. Open **Advanced settings**.
8. Choose Python 3.12 if the option is shown.
9. In the Secrets box, paste:

```toml
GROQ_API_KEY = "your-real-groq-api-key"
```

10. Click **Save** / **Deploy**.

## 5. How the project works

The application has four main parts:

### Streamlit

Provides the user interface.

### CrewAI

Creates the AI research agent and task.

### DuckDuckGo

Provides free web search without requiring a search API key.

### Groq GPT-OSS 120B

Processes the research and writes the final report.

The CrewAI model configuration is:

```python
llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=groq_key,
    temperature=0.2,
    reasoning_effort="medium",
    max_tokens=12000,
)
```

## 6. Beginner explanation

Think of the application like this:

```text
You enter a topic
       ↓
Streamlit receives the topic
       ↓
CrewAI gives the task to the research agent
       ↓
Agent searches DuckDuckGo
       ↓
Agent analyzes the search results
       ↓
Groq GPT-OSS 120B writes the report
       ↓
Streamlit displays the report
       ↓
You can download the Markdown report
```

## 7. Common problems

### GROQ_API_KEY is missing

Check:

```text
.streamlit/secrets.toml
```

locally, or the **Secrets** section of your Streamlit Cloud app.

### The app works locally but not on Streamlit Cloud

Check the Cloud logs and make sure:

- `requirements.txt` is in the repository root.
- `app.py` is the selected entrypoint.
- `GROQ_API_KEY` is present in Streamlit Secrets.
- Python is set to 3.12.

### DuckDuckGo returns no results

Search engines can temporarily rate-limit automated requests. Try the
research again or use a more specific topic.

### CrewAI installation fails

Make sure you are using Python 3.12. Current CrewAI releases require Python
3.10 or newer and below Python 3.14.

## 8. Security

Never commit:

```text
.streamlit/secrets.toml
```

Never hard-code:

```text
GROQ_API_KEY
```

inside `app.py`.

Use Streamlit Secrets instead.

## 9. Future improvements

Once the basic project works, you can add:

- PDF report export
- DOCX report export
- Research history
- Source cards
- Website scraping
- Citation verification
- Topic categories
- Multi-agent research
- Research comparison
- Saved reports
