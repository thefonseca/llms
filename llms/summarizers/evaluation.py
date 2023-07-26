import logging
import os

import fire

from ..evaluation import evaluate_model
from ..inference import get_model_class
from .metrics import summarization_metrics

from .huggingface import (
    Text2TextSummarizer,
    CausalLMSummarizer,
    InstructText2TextSummarizer,
    InstructCausalLMSummarizer,
    AlpacaSummarizer,
    VicunaSummarizer,
    LlamaSummarizer,
)
from .openai import OpenAISummarizer
from .cohere import CohereSummarizer

logger = logging.getLogger(__name__)


MODEL_MAP = {
    "gpt-[-\d\w]*": OpenAISummarizer,
    "facebook/opt-[\d\w]+": CausalLMSummarizer,
    ".*llama-?2.*chat.*": LlamaSummarizer,
    ".*llama.*": CausalLMSummarizer,
    "bigscience/T0[_\d\w]*": InstructText2TextSummarizer,
    "google/flan-t5[-\d\w]+": InstructText2TextSummarizer,
    "google/long-t5[-\d\w]+": InstructText2TextSummarizer,
    ".*alpaca.*": AlpacaSummarizer,
    ".*vicuna.*": VicunaSummarizer,
    "summarize-((medium)|(xlarge))": CohereSummarizer,
    "mosaicml/mpt[-\d\w]$": CausalLMSummarizer,
    "tiiuae/falcon[-\d\w]$": CausalLMSummarizer,
    "mosaicml/mpt[-\d\w]+instruct": AlpacaSummarizer,
    "tiiuae/falcon[-\d\w]+instruct": InstructCausalLMSummarizer,
}


def evaluate_summarizer(model_name=None, **kwargs):
    model_class = get_model_class(
        model_name, model_map=MODEL_MAP, default_class=Text2TextSummarizer
    )
    evaluate_model(
        model_name=model_name,
        model_class=model_class,
        metrics=summarization_metrics,
        **kwargs,
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fire.Fire(evaluate_summarizer)