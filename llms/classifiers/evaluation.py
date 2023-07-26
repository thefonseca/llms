import logging
import os

import fire

from ..evaluation import evaluate_model
from ..inference import get_model_class
from ..metrics import generation_metrics

from .models import (
    InstructCausalLMClassifier,
    AlpacaClassifier,
    VicunaClassifier,
    LlamaClassifier,
    OpenAIClassifier,
)


logger = logging.getLogger(__name__)


MODEL_MAP = {
    "gpt-[-\d\w]*": OpenAIClassifier,
    ".*llama-?2.*chat.*": LlamaClassifier,
    ".*alpaca.*": AlpacaClassifier,
    ".*vicuna.*": VicunaClassifier,
    "mosaicml/mpt[-\d\w]+instruct": AlpacaClassifier,
    "tiiuae/falcon[-\d\w]+instruct": InstructCausalLMClassifier,
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
    model_class = get_model_class(
        model_name, model_map=MODEL_MAP, default_class=InstructCausalLMClassifier
    )
    evaluate_model(
        model_name=model_name,
        model_class=model_class,
        metrics=classification_metrics,
        **kwargs,
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    fire.Fire(evaluate_classifier)
