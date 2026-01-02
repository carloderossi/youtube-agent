from config_loader import load_config
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

config = load_config()

# LLM selection
if config.llm.provider == "ollama":
    llm = OllamaLLM(
        model=config.llm.model,
        temperature=config.llm.temperature
    )
else:
    llm = ChatOpenAI(
        model=config.openai.model,
        temperature=config.llm.temperature
    ) 

# Search tool
search = DuckDuckGoSearchRun()

# Prompt
normalize_prompt = PromptTemplate.from_template("""
You are a Deep Search Agent.

Normalize the following raw search results into JSON.
For each item, extract:

- title
- url
- author (if available)
- timestamp or publication date
- type: article, blog, repo, video, book, course, other
- highlight: true if it's a YouTube video

Raw results:
{raw}

Return JSON only.
""")

# Pipeline (modern LC 1.x style)
pipeline = (
    RunnableLambda(lambda topic: search.run(topic))
    | RunnableLambda(lambda raw: {"raw": raw})
    | normalize_prompt
    | llm
)

def deep_search(topic: str):
    return pipeline.invoke(topic)