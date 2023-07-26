import logging

from ..models.base import PromptBasedLM
from ..models.huggingface import (
    HFModel,
    CausalLM,
    Alpaca,
    Vicuna,
    LlamaChat,
)
from ..models.openai import OpenAIChat

logger = logging.getLogger(__name__)


class InstructTunedClassifier(PromptBasedLM):
    def __init__(self, model_name, labels, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        self.labels = labels

    def get_prompt_args(self):
        labels = self.labels
        if isinstance(labels, (list, tuple, dict)):
            labels = ", ".join(labels)
        return dict(labels=labels)

    def default_user_prompt(self):
        return (
            "Text: {input}\n\nClassify the text above into one of the following categories: {labels}. "
            "Be concise and only write the category name.\nCategory:"
        )

    def postprocess(self, output):
        output = super().postprocess(output)

        if isinstance(self.labels, (list, tuple, dict)):
            target_labels = [x.lower() for x in self.labels]
        else:
            target_labels = self.labels.lower()

        if output.lower() not in target_labels:
            logger.warning(f'Prediction "{output}" is not in labels: {self.labels}.')

        # Labels can be specified as a map from a language label to a symbol.
        # E.g., {"Positive":1, "Negative": 0}
        if isinstance(self.labels, dict):
            output = self.labels.get(output, output)
        return output


class HFClassifier(HFModel):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def process_generation_kwargs(self, **generation_kwargs):
        kwargs = super().process_generation_kwargs(**generation_kwargs)
        if "do_sample" not in kwargs:
            kwargs["do_sample"] = False
        return kwargs


class CausalLMClassifier(HFClassifier, CausalLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class InstructCausalLMClassifier(InstructTunedClassifier, CausalLMClassifier):
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

    def default_system_prompt(self):
        return None


class OpenAIClassifier(OpenAIChat, InstructTunedClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
