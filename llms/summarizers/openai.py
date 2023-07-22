from .base import InstructTunedSummarizer
from ..models.openai import OpenAIChat


class OpenAISummarizer(OpenAIChat, InstructTunedSummarizer):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
