import yaml
from pydantic import BaseModel

class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float

class OpenAIConfig(BaseModel):
    model: str

class YouTubeConfig(BaseModel):
    add_video_info: bool

class DeepSearchConfig(BaseModel):
    enabled: bool
    max_results: int
    highlight_youtube: bool

class AppConfig(BaseModel):
    llm: LLMConfig
    openai: OpenAIConfig
    youtube: YouTubeConfig
    deep_search: DeepSearchConfig

def load_config(path="config.yaml") -> AppConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)
