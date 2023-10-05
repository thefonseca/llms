import logging
import os

import fire

from ..evaluation import evaluate_model
from ..inference import get_model_class
from ..metrics import generation_metrics

from .lmql import (
    LMQLInstructCausalLMClassifier,
    LMQLVicunaClassifier,
    LMQLLlamaChatClassifier,
)
from .models import (
    CausalLMClassifier,
    InstructCausalLMClassifier,
    AlpacaClassifier,
    VicunaClassifier,
    LlamaChatClassifier,
    FalconChatClassifier,
    MistralInsructClassifier,
    OpenAIClassifier,
)


logger = logging.getLogger(__name__)


MODEL_MAP = {
    "lmql:.*vicuna.*": LMQLVicunaClassifier,
    "lmql:.*llama-?2.*": LMQLLlamaChatClassifier,
    "lmql:.*": LMQLInstructCausalLMClassifier,
    "gpt-[-\d\w]*": OpenAIClassifier,
    ".*llama-?2.*chat.*": LlamaChatClassifier,
    "llama-?2.*": CausalLMClassifier,
    ".*alpaca.*": AlpacaClassifier,
    ".*vicuna.*": VicunaClassifier,
    "mosaicml/mpt[-\d\w]+instruct": AlpacaClassifier,
    "tiiuae/falcon[-\d\w]+chat": FalconChatClassifier,
    "tiiuae/falcon[-\d\w]+instruct": InstructCausalLMClassifier,
    "mistralai/Mistral[-\d\w]+Instruct.*": MistralInsructClassifier,
}


def classification_metrics(prediction, reference=None, source=None, parallelized=False):
    metrics = generation_metrics(
        prediction, reference=reference, source=source, parallelized=parallelized
    )

    if reference is not None:
        if str(reference) == "nan":
            reference = "None"

        if isinstance(prediction, str) or isinstance(reference, str):
            prediction = str(prediction)
            reference = str(reference)
            metrics["exact_match"] = prediction.lower() == reference.lower()
            metrics["partial_match"] = (reference.lower() in prediction.lower()) or (
                prediction.lower() in reference.lower()
            )
        else:
            metrics["exact_match"] = prediction == reference

    return metrics


def evaluate_classifier(model_name=None, **kwargs):
    model_class = kwargs.pop("model_class", None)
    if model_class is None:
        model_class = get_model_class(
            model_name, model_map=MODEL_MAP, default_class=InstructCausalLMClassifier
        )
    metrics = kwargs.pop("metrics", [])
    metrics.append(classification_metrics)
    result = evaluate_model(
        model_name=model_name, model_class=model_class, metrics=metrics, **kwargs
    )
    return result


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fire.Fire(evaluate_classifier)
