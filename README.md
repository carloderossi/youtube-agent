# YouTube Summarizer Agent

#### A local, privacy‑preserving YouTube summarizer powered by LangChain 1.x, Ollama, Streamlit, and yt‑dlp.

---

# Overview

The YouTube Summarizer Agent extracts transcripts from YouTube videos, enriches them with metadata, and generates structured summaries using a local LLM (via Ollama) or OpenAI.

It is designed for:

- Long videos (lectures, podcasts, interviews)
- High‑quality summarization using chunking
- Local inference with Ollama
- A simple Streamlit UI
- A clean LangChain 1.x architecture

---

# Features

## Local LLM Summarization (OllamaLLM)
Uses the modern `langchain-ollama` integration (no deprecated APIs).
Optionally you can configure OpenAI call, or extend it easily to whatever else.

## Chunked Summarization Pipeline
Splits long transcripts into overlapping chunks, summarizes each, and combines them.

## Reliable Metadata Extraction (yt‑dlp)
Replaces pytube with a stable extractor for:

- Title
- Channel
- Upload date
- Description

## Transcript Loading
Uses LangChain’s `YoutubeLoader` with `youtube-transcript-api`.

## Streamlit Frontend
Simple UI for entering a YouTube URL and viewing:

- Summary
- Topic
- Metadata

## Config‑Driven Architecture
All settings are stored in `config.yaml`.

## UV‑Managed Environment
Fast, reproducible dependency management.

---

# Project Structure

```
youtube-agent/
│
├── app.py                     # Streamlit UI
├── summarizer_agent.py        # Summarization + topic extraction pipeline
├── config.yaml                # User configuration
├── config_loader.py           # Loads config.yaml
├── pyproject.toml             # UV project definition
└── README.md                  # This file
```

---

# Installation

## 1. Install UV

```
pip install uv
```

## 2. Create the environment

```
uv sync
```

# LLM Setup

## Option A — Local LLM via Ollama (recommended)

Install Ollama:

```
https://ollama.com/download
```

Pull a model:

```
ollama pull qwen3:1.7b
```

Configure:

```
llm:
  provider: ollama
  model: qwen3:1.7b
  temperature: 0.2
```

## Option B — OpenAI

```
llm:
  provider: openai
  model: gpt-4o-mini
  temperature: 0.2
```

---

# Running the App

```
uv run streamlit run app.py
```

Open:

```
http://localhost:8501
```

Paste a YouTube URL and receive:

- Summary
- Topic
- Metadata

---

# How It Works

## 1. Load transcript

```
YoutubeLoader.from_youtube_url(url, add_video_info=False)
```

## 2. Fetch metadata

```
yt_dlp.YoutubeDL().extract_info(url)
```

## 3. Chunk transcript

```
RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=200,
)
```

## 4. Summarize each chunk

```
summary_prompt → llm
```

## 5. Combine summaries

A second LLM call merges partial summaries.

## 6. Extract topic

A lightweight LLM call extracts the main subject.

---

# Example Output

Summary:
- Key ideas
- Arguments
- Insights
- Structured sections

Topic:
- "AI model analysis and data trends"

---

# Configuration

`config.yaml` controls:

- LLM provider
- Model name
- Temperature
- Chunk sizes
- UI settings

Example:

```
llm:
  provider: ollama
  model: qwen3:1.7b
  temperature: 0.2

youtube:
  add_video_info: false
```

---

# Privacy

- All transcript processing happens locally.
- With Ollama, no data leaves your machine.
- OpenAI mode sends only transcript text to the API.

---

# License

MIT License.

---

# Acknowledgements

- LangChain 1.x
- Ollama
- yt‑dlp
- Streamlit
- youtube‑transcript‑api

