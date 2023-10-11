import re

from .base import InstructTunedSummarizer, PromptBasedSummarizer
from ..models.huggingface import (
    HFModel,
    Text2TextLM,
    CausalLM,
    InstructText2TextLM,
    Alpaca,
    Vicuna,
    Llama2,
    LlamaChat,
    FalconChat,
)


class HFSummarizer(HFModel):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def process_generation_kwargs(self, **generation_kwargs):
        generation_kwargs = super().process_generation_kwargs(**generation_kwargs)
        config = self.load_model_config()
        if hasattr(config, "task_specific_params"):
            task_params = config.task_specific_params
            if task_params and "summarization" in task_params:
                for key, param in task_params["summarization"].items():
                    if (
                        key not in ["prefix", "max_length"]
                        and key not in generation_kwargs
                    ):
                        generation_kwargs[key] = param
        return generation_kwargs


class Text2TextSummarizer(HFSummarizer, Text2TextLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class CausalLMSummarizer(HFSummarizer, CausalLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class InstructText2TextSummarizer(HFSummarizer, InstructText2TextLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def default_input_prompt(self):
        return "summarize: {input}"


class InstructCausalLMSummarizer(InstructTunedSummarizer, CausalLMSummarizer):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class PromptBasedCausalLMSummarizer(PromptBasedSummarizer, CausalLMSummarizer):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class AlpacaSummarizer(InstructCausalLMSummarizer, Alpaca):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class VicunaSummarizer(InstructCausalLMSummarizer, Vicuna):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class Llama2Summarizer(PromptBasedCausalLMSummarizer, Llama2):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class LlamaChatSummarizer(InstructCausalLMSummarizer, LlamaChat):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def default_system_prompt(self):
        return None

    def postprocess(self, output):
        output = re.sub(r"^\] ", "", output).strip()
        return super().postprocess(output)


class FalconChatSummarizer(InstructCausalLMSummarizer, FalconChat):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
