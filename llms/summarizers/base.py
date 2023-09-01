import logging

from ..models.base import PromptBasedLM

logger = logging.getLogger(__name__)


class PromptBasedSummarizer(PromptBasedLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def default_user_prompt(self):
        return "\nTL;DR:"


class InstructTunedSummarizer(PromptBasedLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def default_user_prompt(self):
        return (
            "\nWrite a summary of the article above in {budget} {budget_unit}."
        )
