from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableLambda
from config_loader import load_config
from langchain_ollama import OllamaLLM

import httpx
from langchain_core.exceptions import OutputParserException

#uv run streamlit run app.py

config = load_config()

def is_ollama_running(host: str = "http://localhost:11434") -> bool:
    """Quick health check to see if Ollama is reachable."""
    try:
        r = httpx.get(f"{host}/api/tags", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False

# LLM selection
if config.llm.provider == "ollama":
    # llm = Ollama(model=config.llm.model, temperature=config.llm.temperature) # OLD style, deprecated
    llm = OllamaLLM(
    model=config.llm.model,
    temperature=config.llm.temperature,
)

else:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model=config.openai.model, temperature=config.llm.temperature)

# Summarization prompt
summary_prompt = PromptTemplate.from_template("""
Summarize the following YouTube transcript in a clear, structured way.
Highlight key ideas, arguments, and actionable insights.

Transcript:
{transcript}
""")

# Topic extraction prompt
topic_prompt = PromptTemplate.from_template("""
Extract the main subject of the following summary.
Return only a short phrase.

Summary:
{summary}
""")

# Steps as runnables
load_video = RunnableLambda(
    lambda url: YoutubeLoader.from_youtube_url(
        url, 
        add_video_info=False # config.youtube.add_video_info # pytube doesn't work well with LangChain's YoutubeLoader
    ).load()
)

extract_text = RunnableLambda(lambda docs: docs[0].page_content)

to_prompt_input = RunnableLambda(lambda transcript: {"transcript": transcript})

# Build pipeline using |
summarizer = (
    load_video
    | extract_text
    | to_prompt_input
    | summary_prompt
    | llm
)

topic_extractor = (
    RunnableLambda(lambda summary: {"summary": summary})
    | topic_prompt
    | llm
)

def summarize_youtube(url: str):
    # 1. Check if Ollama is running BEFORE invoking the chain
    if not is_ollama_running():
        return (
            "❌ Ollama is not running.\n\n"
            "Please start Ollama first (e.g., run `ollama serve`) "
            "and then try again."
        )

    # 2. Try running the summarizer
    try:
        return summarizer.invoke(url)

    except httpx.ConnectError:
        return (
            "❌ Could not connect to Ollama.\n\n"
            "Make sure Ollama is running and accessible at `localhost:11434`."
        )

    except OutputParserException as e:
        return f"⚠️ The model returned an unexpected output format:\n{e}"

    except Exception as e:
        return f"⚠️ Unexpected error while summarizing:\n{e}"


def extract_topic(summary: str):
    return topic_extractor.invoke(summary) #.content
