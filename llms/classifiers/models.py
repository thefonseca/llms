from ..models.base import PromptBasedLM
from ..models.huggingface import (
    CausalLM,
    Alpaca,
    Vicuna,
    LlamaChat,
)
from ..models.openai import OpenAIChat


class InstructTunedClassifier(PromptBasedLM):
    def __init__(self, model_name, labels, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        self.labels = labels

    def get_prompt_args(self):
        labels = self.labels
        if isinstance(labels, list):
            labels = ", ".join(labels)
        return dict(labels=labels)

    def default_task_prompt(self):
        return "Classify the sentence above in one of the following labels: {labels}. Output only the label.\nAnswer:"


class InstructCausalLMClassifier(InstructTunedClassifier, CausalLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class AlpacaClassifier(Alpaca, InstructCausalLMClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class VicunaClassifier(Vicuna, InstructCausalLMClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class LlamaClassifier(LlamaChat, InstructCausalLMClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class OpenAIClassifier(OpenAIChat, InstructTunedClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
