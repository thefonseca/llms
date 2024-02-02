import logging

import fire

from ..evaluation import evaluate_model
from ..inference import get_model_class, preprocess_kwargs
from .metrics import summarization_metrics

from .huggingface import (
    Text2TextSummarizer,
    CausalLMSummarizer,
    InstructText2TextSummarizer,
    InstructCausalLMSummarizer,
    AlpacaSummarizer,
    VicunaSummarizer,
    Llama2Summarizer,
    LlamaChatSummarizer,
    FalconChatSummarizer,
)
from .openai import OpenAISummarizer
from .cohere import CohereSummarizer

logger = logging.getLogger(__name__)


MODEL_MAP = {
    r"gpt-[-\d\w]*": OpenAISummarizer,
    r"facebook/opt-[\d\w]+": CausalLMSummarizer,
    r".*llama-?2.*chat.*": LlamaChatSummarizer,
    r".*llama-?2.*": Llama2Summarizer,
    r".*llama.*": CausalLMSummarizer,
    r"bigscience/T0[_\d\w]*": InstructText2TextSummarizer,
    r"google/flan-t5[-\d\w]+": InstructText2TextSummarizer,
    r"google/long-t5[-\d\w]+": InstructText2TextSummarizer,
    r".*alpaca.*": AlpacaSummarizer,
    r".*vicuna.*": VicunaSummarizer,
    r"summarize-((medium)|(xlarge))": CohereSummarizer,
    r"mosaicml/mpt[-\d\w]$": CausalLMSummarizer,
    r"tiiuae/falcon[-\d\w]+chat": FalconChatSummarizer,
    r"tiiuae/falcon[-\d\w]+instruct": InstructCausalLMSummarizer,
    r"tiiuae/falcon[-\d\w]$": CausalLMSummarizer,
    r"mosaicml/mpt[-\d\w]+instruct": AlpacaSummarizer,
}


def get_summarizer_model_class(
    model_name, model_map=MODEL_MAP, default_class=Text2TextSummarizer
):
    model_class = get_model_class(
        model_name, model_map=model_map, default_class=default_class
    )
    return model_class


def evaluate_summarizer(model_name=None, model_class=None, metrics=None, **kwargs):
    if model_name and model_class is None:
        model_class = get_summarizer_model_class(model_name)
    if metrics is None:
        metrics = []
    metrics.append(summarization_metrics)
    result = evaluate_model(
        model_name=model_name,
        model_class=model_class,
        metrics=metrics,
        **kwargs,
    )
    return result


def run(**kwargs):
    kwargs = preprocess_kwargs(kwargs)
    outputs = evaluate_summarizer(**kwargs)["predictions"]
    if len(outputs) == 1:
        print(f"> {kwargs['model_name']}:\n{outputs[0]}")


def main():
    fire.Fire(run)


if __name__ == "__main__":
    main()