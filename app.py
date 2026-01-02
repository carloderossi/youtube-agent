import streamlit as st
from summarizer_agent import summarize_youtube, extract_topic
from deep_search_agent import deep_search
from config_loader import load_config

config = load_config()

st.title("🎬 YouTube Summarizer + Deep Search (LangChain 1.1.2)")

url = st.text_input("YouTube URL")
enable_deep = st.checkbox("Enable Deep Search", value=config.deep_search.enabled)

if st.button("Run"):
    if not url.strip():
        st.error("Please enter a valid URL.")
    else:
        with st.spinner("Summarizing..."):
            summary = summarize_youtube(url)

        st.subheader("Summary")
        st.write(summary)

        if enable_deep:
            with st.spinner("Running Deep Search..."):
                topic = extract_topic(summary)
                results = deep_search(topic)

            st.subheader(f"Deep Search Results for: {topic}")
            st.json(results)
