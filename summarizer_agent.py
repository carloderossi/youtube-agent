from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableLambda
from config_loader import load_config
from langchain_community.llms import Ollama


config = load_config()

# LLM selection
if config.llm.provider == "ollama":
    from langchain_community.llms import Ollama
    llm = Ollama(model=config.llm.model, temperature=config.llm.temperature)

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
    return summarizer.invoke(url) #.content

def extract_topic(summary: str):
    return topic_extractor.invoke(summary) #.content
