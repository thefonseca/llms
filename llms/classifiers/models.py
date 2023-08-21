import logging
import re

import numpy as np
import torch
import torch.nn.functional as F

from ..models.base import BaseLM
from ..models.huggingface import (
    HFModel,
    InstructCausalLM,
    Alpaca,
    Vicuna,
    LlamaChat,
)
from ..models.openai import OpenAIChat
from ..utils.memoizer import memoize

logger = logging.getLogger(__name__)


class BaseClassifier(BaseLM):
    def __init__(self, model_name, labels, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        self.labels = labels

    def postprocess(self, output):
        output = super().postprocess(output)
        # Labels can be specified as a map from a language label to a symbol.
        # E.g., {"Positive":1, "Negative": 0}
        if isinstance(self.labels, dict):
            output = self.labels.get(output, output)
        return output
    

class HFClassifier(BaseClassifier, HFModel):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def process_generation_kwargs(self, **generation_kwargs):
        kwargs = super().process_generation_kwargs(**generation_kwargs)
        if "do_sample" not in kwargs:
            kwargs["do_sample"] = False
        return kwargs


class InstructTunedClassifier(BaseClassifier):
    def __init__(self, model_name, labels, **kwargs) -> None:
        super().__init__(model_name, labels, **kwargs)
        
    def get_prompt_args(self):
        labels = self.labels
        if isinstance(labels, (list, tuple, dict)):
            labels = [f"- {label}\n" for label in labels]
        return dict(labels=labels)

    def default_user_prompt(self):
        return (
            "Text: {input}\n\nClassify the text above into one of the following categories:\n{labels}\n"
            "Be concise and only write the category name."
            "\nCategory:"
        )

    def fix_prediction(self, output):
        if isinstance(self.labels, (list, tuple, dict)):
            target_labels = [x.lower() for x in self.labels]
        else:
            target_labels = self.labels.lower()

        if output == "":
            logger.warning(f"Prediction is empty")

        elif output.lower() not in target_labels:
            output = re.sub(r'(C|c)ategory:\s*', "", output)
            output = output.strip()

            if output.lower() not in target_labels:
                for label in self.labels:
                    # some models output truncated labels
                    if output.lower() == label.lower()[: len(output)]:
                        logger.warning(f"Fixing prediction: {output} => {label}")
                        output = label
                        break
                else:
                    logger.warning(
                        f'Prediction "{output}" is not in labels: {self.labels}.'
                    )

        return output

    def postprocess(self, output):
        output = self.fix_prediction(output)
        output = super().postprocess(output)
        return output


class MLECausalLMClassifier(HFClassifier, InstructCausalLM):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def default_user_prompt(self):
        return "Text: {input}\nCategory:"

    def get_scores_for_labels(self, model_input, labels, model, tokenizer):
        model_input = model_input.strip() + " "
        # random.shuffle(labels)
        _labels = labels
        inputs = [f"{model_input}{l}" for l in _labels]

        # Get encodings
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        input_enc = tokenizer([model_input], return_tensors="pt")
        inputs_enc = tokenizer(inputs, return_tensors="pt", padding="longest")

        # print(223, [tokenizer.convert_ids_to_tokens(ids) for ids in inputs_enc.input_ids])
        target_enc = inputs_enc.input_ids.clone()
        target_enc = (
            target_enc
            - (100 + tokenizer.pad_token_id)
            * (target_enc == tokenizer.pad_token_id).long()
        )

        with torch.no_grad():
            inputs_enc = inputs_enc.to(model.device)
            model_output = model(
                attention_mask=inputs_enc.attention_mask,
                input_ids=inputs_enc.input_ids,
                return_dict=True,
            )

        logits = model_output.logits[:, :-1]
        target_enc = target_enc[:, 1:]
        logits = logits.flatten(0, 1)
        target_enc = target_enc.flatten(0, 1)

        # Compute the log probabilities associated with each of the labels
        logits = logits.to(target_enc.device)
        labels_log_probs = F.cross_entropy(logits, target_enc, reduction="none")

        # Sum log probs for each of the (input, label) pair
        labels_scores = labels_log_probs.view(len(labels), -1)
        input_len = input_enc.input_ids.shape[1]
        
        labels_scores = labels_scores[:, input_len - 2 :]
        labels_scores = labels_scores.sum(dim=-1)
        # Note: Label log probabilities are positive (due to the internals of pytorch's
        # cross entropy). To obtain the "logits", we need to multiply by -1.
        labels_scores = labels_scores * -1
        labels_scores = labels_scores.exp().detach().numpy()
        # labels_probs = labels_scores / labels_scores.sum()
        best_label = labels[np.argmax(labels_scores)]
        return best_label

    @memoize(ignore_kwargs=["model_path"])
    def generate_cached(
        self, model_name, model_input, memoizer_ignore_cache=False, **kwargs
    ):
        model_kwargs = self.get_model_kwargs()
        model = self.load_model(**model_kwargs)
        tokenizer = self.load_tokenizer()
        labels = self.labels
        return self.get_scores_for_labels(model_input, labels, model, tokenizer)


class InstructCausalLMClassifier(InstructTunedClassifier, InstructCausalLM):
    def __init__(self, model_name, labels, **kwargs) -> None:
        super().__init__(model_name, labels, **kwargs)


class AlpacaClassifier(Alpaca, InstructCausalLMClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class VicunaClassifier(Vicuna, InstructCausalLMClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)


class LlamaChatClassifier(LlamaChat, InstructCausalLMClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)

    def default_system_prompt(self):
        return None


class OpenAIClassifier(OpenAIChat, InstructTunedClassifier):
    def __init__(self, model_name, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
