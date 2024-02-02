import logging

import fire

from ..evaluation import evaluate_model
from ..inference import get_model_class, preprocess_kwargs
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
    Llama2Classifier,
    LlamaChatClassifier,
    FalconChatClassifier,
    MistralInsructClassifier,
    OpenAIClassifier,
)


logger = logging.getLogger(__name__)


MODEL_MAP = {
    r"lmql:.*vicuna.*": LMQLVicunaClassifier,
    r"lmql:.*llama-?2.*": LMQLLlamaChatClassifier,
    r"lmql:.*": LMQLInstructCausalLMClassifier,
    r"gpt-[-\d\w]*": OpenAIClassifier,
    r".*llama-?2.*chat.*": LlamaChatClassifier,
    r"llama-?2.*": Llama2Classifier,
    r"llama.*": CausalLMClassifier,
    r".*alpaca.*": AlpacaClassifier,
    r".*vicuna.*": VicunaClassifier,
    r"mosaicml/mpt[-\d\w]+instruct": AlpacaClassifier,
    r"tiiuae/falcon[-\d\w]+chat": FalconChatClassifier,
    r"tiiuae/falcon[-\d\w]+instruct": InstructCausalLMClassifier,
    r"mistralai/Mistral[-\d\w]+Instruct.*": MistralInsructClassifier,
}


def classification_metrics(
    prediction, reference=None, source=None, parallelized=False, index=None, **kwargs
):
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


def evaluate_classifier(model_name=None, metrics=None, **kwargs):
    model_class = kwargs.pop("model_class", None)
    if model_class is None:
        model_class = get_model_class(
            model_name, model_map=MODEL_MAP, default_class=InstructCausalLMClassifier
        )
    if metrics is None:
        metrics = []
    metrics.append(classification_metrics)
    result = evaluate_model(
        model_name=model_name, model_class=model_class, metrics=metrics, **kwargs
    )
    return result


def run(**kwargs):
    kwargs = preprocess_kwargs(kwargs)
    outputs = evaluate_classifier(**kwargs)["predictions"]
    if len(outputs) == 1:
        print(f"> {kwargs['model_name']}: {outputs[0]}")


def main():
    fire.Fire(run)


if __name__ == "__main__":
    main()
